import torch.multiprocessing as mp
import gymnasium as gym
import numpy as np
import sys
import os
import time
import random
from typing import Union
from itertools import product
from experiments.gigpo_diag import science_contract as contract


def _receive(remote):
    result = remote.recv()
    if isinstance(result, dict) and 'scienceworld_error' in result:
        raise RuntimeError('ScienceWorld native worker failed: ' + result['scienceworld_error'])
    return result

def compute_reward(info, multi_modal=False):
    reward = 10.0 * float(info['won'])
    return reward

def _worker(remote, seed, task_nums, simplifications_preset, env_step_limit, jar_path, split=None, variations_idx=None, worker_slot=0):
    from scienceworld import ScienceWorldEnv
    env = ScienceWorldEnv("", jar_path, envStepLimit=env_step_limit)
    prev_score = 0
    try:
        taskNames = env.get_task_names()
        random.seed(seed)
        remote.send(taskNames)  # Already-read canonical routing table, no new JVM call.
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                action = data
                observation, reward, done, info = env.step(action)
                valid_actions = env.get_possible_actions()
                valid_objs = env.get_possible_objects()
                contract.native_action_inputs(valid_actions, valid_objs)
                valid_action_strs = f"Valid_actions: {valid_actions}, OBJ needs to be replaced with one of the following objects: {valid_objs}\n example: <action>focus on door</action>"
                info['available_actions'] = valid_action_strs
                info['observation_text'] = observation
                info["possible_actions"] = env.get_valid_action_object_combinations()
                contract.required_inputs(observation, info)
                info['task_score'] = info['score']
                isCompleted = done
                prev_score = info['score']
                info["won"] = isCompleted and info["score"] > 0
                contract.optional_strict_label(env, info)
                reward = compute_reward(info)
                remote.send((observation, reward, isCompleted, info))
            elif cmd == 'reset':
                if not isinstance(data, dict):
                    raise ValueError("ScienceWorld reset requires explicit grouped assignment")
                task_id, task_variation = int(data["task_id"]), int(data["variation_id"])
                if (task_id, task_variation) not in {tuple(x) for x in variations_idx}:
                    raise ValueError("Reset assignment is outside this split's task pool")
                task_num = task_id
                taskName = taskNames[task_id]
                random.seed(int(data["reset_seed"]))
                simplification_str = simplifications_preset if simplifications_preset else ""
                env.load(taskName, task_variation, simplification_str)
                observation, info = env.reset()
                receipt = contract.load_receipt(env, data, taskNames, task_id, task_variation,
                                                simplification_str, env_step_limit, worker_slot)
                task_description = env.get_task_description()
                info['task_description'] = task_description
                valid_actions = env.get_possible_actions()
                valid_objs = env.get_possible_objects()
                contract.native_action_inputs(valid_actions, valid_objs)
                valid_action_strs = f"Valid_actions: {valid_actions}, OBJ needs to be replaced with one of the following objects: {valid_objs}\n example: <action>focus on door</action>"
                info['available_actions'] = valid_action_strs
                info['observation_text'] = observation
                info["possible_actions"] = env.get_valid_action_object_combinations()
                info['won'] = False
                info['task_num'] = task_num
                info['task_id'] = taskName
                info['variation_id'] = task_variation
                info['reset_seed'] = int(data['reset_seed'])
                info['score'] = info.get('score', None)
                info['load_receipt'] = receipt
                info['simplifications_requested'] = simplification_str
                contract.required_inputs(observation, info, reset=True)
                from experiments.gigpo_diag import diagnostics as diag
                diag.capture_science_reset(info, observation, valid_objs)
                prev_score = 0
                remote.send((observation, info))
            elif cmd == 'close':
                remote.close()
                break
            else:
                raise NotImplementedError(f"Unknown command sent to worker: {cmd}")
    except Exception as exc:
        remote.send({'scienceworld_error': f'{type(exc).__name__}: {exc}'})
        raise
    finally:
        env.close()
        remote.close()

class SciWorldMultiProcessEnv(gym.Env):
    def __init__(
        self,
        seed: int = 0,
        env_num: int = 1,
        group_n: int = 1,
        task_nums: list = [1],
        split: str = "train",
        simplifications_preset: str = "",
        env_step_limit: int = 100,
        jar_path: str = None,
        variations_idx: list = None
    ) -> None:
        super().__init__()
        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.split = split
        self.task_nums = task_nums
        self.variations_idx = variations_idx
        self.simplifications_preset = simplifications_preset
        self.env_step_limit = env_step_limit
        self.jar_path = jar_path
        self._rng = np.random.RandomState(seed)
        self._reset_count = 0
        self._assignment_prefix = f"{os.environ.get('GIGPO_RUN_ID', 'local')}/{split}"
        self._confirmed_violations = contract.confirmed_violations()
        self._parent_remotes: list[mp.connection.Connection] = []
        self._workers: list[mp.Process] = []
        ctx = mp.get_context('spawn')
        for i in range(self.num_processes):
            parent_remote, child_remote = ctx.Pipe()
            seed_i = seed + i
            worker = ctx.Process(
                target=_worker,
                args=(child_remote, seed_i, self.task_nums, self.simplifications_preset,
                      self.env_step_limit, self.jar_path, self.split, self.variations_idx, i),
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)
            self._parent_remotes.append(parent_remote)
            child_remote.close()
        tables = [_receive(remote) for remote in self._parent_remotes]
        if not tables or any(table != tables[0] for table in tables):
            contract.fail('assignment_mismatch', 'native task routing tables disagree')
        self._task_names = tables[0]
        self.prev_available_actions = [[] for _ in range(self.num_processes)]
        self.prev_possible_actions = [[] for _ in range(self.num_processes)]

    def step(self, actions: list[str]):
        if len(actions) != self.num_processes:
            raise ValueError(
                f'Expected {self.num_processes} actions, got {len(actions)}',
            )
        for remote, action in zip(self._parent_remotes, actions):
            remote.send(('step', action))
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for i, remote in enumerate(self._parent_remotes):
            obs, reward, done, info = _receive(remote)
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
            self.prev_available_actions[i] = info['available_actions']
            self.prev_possible_actions[i] = info["possible_actions"]
        return obs_list, reward_list, done_list, info_list

    def reset(self):
        # Parent-only task RNG; replacement preserves independent group draws.
        selected = self._rng.choice(len(self.variations_idx), size=self.env_num, replace=True)
        seeds = self._rng.randint(0, 2**31 - 1, size=self.env_num)
        assignments = []
        for group_slot, (pool_index, reset_seed) in enumerate(zip(selected, seeds)):
            task_id, variation_id = self.variations_idx[int(pool_index)]
            assignment = dict(task_id=int(task_id), variation_id=int(variation_id), reset_seed=int(reset_seed))
            assignment.update(task_name=self._task_names[int(task_id)], group_slot=group_slot,
                              assignment_token=f'{self._assignment_prefix}/{self._reset_count + 1}/{group_slot}',
                              reset_seq=self._reset_count + 1,
                              simplifications=self.simplifications_preset or '', env_step_limit=self.env_step_limit,
                              contract_policy=contract.POLICY)
            assignments.extend([{**assignment, 'worker_slot': group_slot * self.group_n + j}
                                for j in range(self.group_n)])
        self._reset_count += 1
        contract.enforce_confirmed_violations(assignments, self._confirmed_violations)
        self.expected_assignments = assignments
        for remote, assignment in zip(self._parent_remotes, assignments):
            remote.send(('reset', assignment))
        obs_list, info_list = [], []
        for i, remote in enumerate(self._parent_remotes):
            obs, info = _receive(remote)
            obs_list.append(obs)
            info_list.append(info)
            self.prev_available_actions[i] = info['available_actions']
            self.prev_possible_actions[i] = info["possible_actions"]
        contract.validate_assignment_receipts(assignments, [info.get('load_receipt') for info in info_list])
        return obs_list, info_list

    def sampler_state(self):
        name, keys, pos, gaussian, cached = self._rng.get_state()
        return dict(name=name, keys=keys.tolist(), pos=pos, gaussian=gaussian, cached=cached, resets=self._reset_count)

    def restore_sampler_state(self, state):
        self._rng.set_state((state['name'], np.array(state['keys'], dtype=np.uint32), state['pos'], state['gaussian'], state['cached']))
        self._reset_count = state['resets']

    @property
    def get_available_actions(self):
        return self.prev_available_actions

    @property
    def get_admissible_commands(self):
        return self.prev_available_actions

    @property
    def get_possible_actions(self):
        return self.prev_possible_actions

    def close(self):
        if getattr(self, '_closed', False):
            return
        for remote in self._parent_remotes:
            remote.send(('close', None))
        for worker in self._workers:
            worker.join()
        self._closed = True

    def __del__(self):
        self.close()

def build_sciworld_envs(
    seed: int = 0,
    env_num: int = 1,
    group_n: int = 1,
    task_nums: Union[int, list] = 1,
    split: str = "train",
    simplifications_preset: str = "",
    env_step_limit: int = 100,
    jar_path: str = None,
    variations_idx: list = None
):
    return SciWorldMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        task_nums=task_nums,
        split=split,
        simplifications_preset=simplifications_preset,
        env_step_limit=env_step_limit,
        jar_path=jar_path,
        variations_idx=variations_idx
    )
