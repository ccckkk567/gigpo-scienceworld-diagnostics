import torch.multiprocessing as mp
import gymnasium as gym
import numpy as np
import sys
import os
import time
import random
from typing import Union
from itertools import product

def compute_reward(info, multi_modal=False):
    reward = 10.0 * float(info['won'])
    return reward

def _worker(remote, seed, task_nums, simplifications_preset, env_step_limit, jar_path, split=None, variations_idx=None):
    from scienceworld import ScienceWorldEnv
    env = ScienceWorldEnv("", jar_path, envStepLimit=env_step_limit)
    taskNames = env.get_task_names()
    random.seed(seed)
    prev_score = 0
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                action = data
                observation, reward, done, info = env.step(action)
                valid_actions = env.get_possible_actions()
                valid_objs = env.get_possible_objects()
                valid_action_strs = f"Valid_actions: {valid_actions}, OBJ needs to be replaced with one of the following objects: {valid_objs}\n example: <action>focus on door</action>"
                info['available_actions'] = valid_action_strs
                info['observation_text'] = observation
                info["possible_actions"] = env.get_valid_action_object_combinations()
                info['score'] = info.get('score', 0.0)
                info['task_score'] = info['score']
                isCompleted = done
                prev_score = info['score']
                info["won"] = isCompleted and info["score"] > 0
                info["native_score_unrounded"] = float(env.server.getScore())
                info["native_completed"] = bool(env.server.getCompleted())
                info["strict_success"] = info["native_completed"] and info["native_score_unrounded"] >= 1.0
                info["strict_definition"] = "ScienceWorld 1.2.3 Java completed and raw score >= 1; read-only, no reward override"
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
                task_description = env.get_task_description()
                info['task_description'] = task_description
                valid_actions = env.get_possible_actions()
                # Same native alias generator, one enumeration in place of the
                # string-only API. Keep its names/order for the policy prompt.
                object_records = env.get_possible_object_referent_types_LUT()
                from experiments.gigpo_diag.diagnostics import science_object_evidence
                valid_objs, _ = science_object_evidence(object_records)
                valid_action_strs = f"Valid_actions: {valid_actions}, OBJ needs to be replaced with one of the following objects: {valid_objs}\n example: <action>focus on door</action>"
                info['available_actions'] = valid_action_strs
                info['observation_text'] = observation
                # One native action enumeration, as before. The JSON API also
                # exposes object IDs; a second enumeration would draw new aliases.
                action_records = env.get_valid_action_object_combinations_with_templates()
                info["possible_actions"] = sorted(row["action"] for row in action_records)
                info['won'] = False
                info['task_num'] = task_num
                info['task_id'] = taskName
                info['variation_id'] = task_variation
                info['reset_seed'] = int(data['reset_seed'])
                info['score'] = info.get('score', None)
                from experiments.gigpo_diag.diagnostics import digest
                info['initial_state_components'] = {'observation': observation, 'objects': valid_objs,
                                                    'possible_object_referents': object_records,
                                                    'possible_actions': sorted(info['possible_actions']),
                                                    'possible_action_templates': action_records,
                                                    'score': info['score']}
                info['reset_evidence_version'] = 3
                info['initial_state_hash'] = digest(info['initial_state_components'])
                info['simplifications_requested'] = simplification_str
                info['initialization_semantics'] = 'Java task variation controls world; Python seed recorded separately'
                prev_score = 0
                remote.send((observation, info))
            elif cmd == 'close':
                remote.close()
                break
            else:
                raise NotImplementedError(f"Unknown command sent to worker: {cmd}")
    finally:
        env.close()

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
        self._parent_remotes: list[mp.connection.Connection] = []
        self._workers: list[mp.Process] = []
        ctx = mp.get_context('spawn')
        for i in range(self.num_processes):
            parent_remote, child_remote = ctx.Pipe()
            seed_i = seed + i
            worker = ctx.Process(
                target=_worker,
                args=(child_remote, seed_i, self.task_nums, self.simplifications_preset,
                      self.env_step_limit, self.jar_path, self.split, self.variations_idx),
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)
            self._parent_remotes.append(parent_remote)
            child_remote.close()
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
            obs, reward, done, info = remote.recv()
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
        for pool_index, reset_seed in zip(selected, seeds):
            task_id, variation_id = self.variations_idx[int(pool_index)]
            assignment = dict(task_id=int(task_id), variation_id=int(variation_id), reset_seed=int(reset_seed))
            assignments.extend([assignment.copy() for _ in range(self.group_n)])
        self._reset_count += 1
        for remote, assignment in zip(self._parent_remotes, assignments):
            remote.send(('reset', assignment))
        obs_list, info_list = [], []
        for i, remote in enumerate(self._parent_remotes):
            obs, info = remote.recv()
            obs_list.append(obs)
            info_list.append(info)
            self.prev_available_actions[i] = info['available_actions']
            self.prev_possible_actions[i] = info["possible_actions"]
        from experiments.gigpo_diag.diagnostics import verify_groups
        verify_groups({'anchor': obs_list}, info_list, [i // self.group_n for i in range(self.num_processes)], environment='scienceworld')
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
