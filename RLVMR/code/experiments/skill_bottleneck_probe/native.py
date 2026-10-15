"""One explicit study assignment, through the unchanged training worker and manager."""
import multiprocessing as mp


class NativeEnv:
    def __init__(self, protocol, variation, reset_seed, token):
        from agent_system.environments.env_package.sciworld.envs import _worker
        from experiments.gigpo_diag import science_contract as contract
        self.protocol=protocol; self.remote=None; self.process=None
        ctx=mp.get_context('spawn'); parent,child=ctx.Pipe()
        self.remote=parent
        self.process=ctx.Process(target=_worker, args=(child,reset_seed,[protocol['task_num']],
            protocol['simplifications'],protocol['env_step_limit'],None,'train',
            [(protocol['task_num'],variation)],0), daemon=True)
        self.process.start(); child.close()
        try:
            names=self.receive()
            if names[protocol['task_num']] != protocol['task_id']: raise ValueError('Native task routing mismatch')
            self.assignment=dict(task_id=protocol['task_num'],task_name=protocol['task_id'],variation_id=variation,
                reset_seed=reset_seed,assignment_token=token,group_slot=0,reset_seq=1,worker_slot=0,
                simplifications=protocol['simplifications'],env_step_limit=protocol['env_step_limit'],contract_policy=contract.POLICY)
            contract.enforce_confirmed_violations([self.assignment],contract.confirmed_violations())
        except BaseException:
            self.close(); raise

    def receive(self):
        from agent_system.environments.env_package.sciworld.envs import _receive
        if not self.remote.poll(90): raise TimeoutError('Study worker did not respond within 90 seconds')
        return _receive(self.remote)

    def reset(self):
        from experiments.gigpo_diag.science_contract import validate_assignment_receipts
        self.remote.send(('reset',self.assignment)); obs,info=self.receive()
        validate_assignment_receipts([self.assignment],[info.get('load_receipt')])
        self.info=info
        return [obs],[info]

    def step(self, actions):
        if len(actions)!=1: raise ValueError('Study worker expects one action')
        self.remote.send(('step',actions[0])); obs,reward,done,info=self.receive(); self.info=info
        return [obs],[reward],[done],[info]

    @property
    def get_possible_actions(self): return [self.info['possible_actions']]

    def close(self):
        if self.remote:
            try: self.remote.send(('close',None))
            except (OSError,EOFError): pass
        if self.process:
            self.process.join(10)
            if self.process.is_alive(): self.process.terminate(); self.process.join(5)
        if self.remote: self.remote.close()


def manager(protocol, variation, reset_seed, token):
    from omegaconf import OmegaConf
    from agent_system.environments.env_manager import SciWorldEnvironmentManager
    from agent_system.environments.env_package.sciworld.projection import sciworld_projection
    env=NativeEnv(protocol,variation,reset_seed,token)
    try:
        return SciWorldEnvironmentManager(env,sciworld_projection,'sciworld',
            OmegaConf.create({'env':{'sciworld':{'meta_think':False}}})),env
    except BaseException: env.close(); raise


class Renderer:
    def __init__(self, tokenizer, protocol, placeholder_path):
        import pyarrow.parquet as pq
        from types import SimpleNamespace
        from omegaconf import OmegaConf
        from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
        rows=pq.read_table(placeholder_path,columns=['prompt','data_source']).to_pylist()
        if not rows or any(r['prompt'] != rows[0]['prompt'] for r in rows):
            raise ValueError('Placeholder prompts are missing or heterogeneous')
        self.tokenizer=tokenizer; self.placeholder=rows[0]
        self.batch=SimpleNamespace(non_tensor_batch={'raw_prompt':[rows[0]['prompt']], 'data_source':[rows[0]['data_source']]})
        config=OmegaConf.create({'data':{'max_prompt_length':protocol['max_prompt_tokens'],'return_raw_chat':True}})
        self.collector=TrajectoryCollector(config,tokenizer)

    def render(self, obs, hint=None):
        obs={**obs,'text':list(obs['text'])}
        if hint: obs['text'][0]+='\n\n'+hint
        row=self.collector.preprocess_single_sample(0,self.batch,obs)
        prompt=self.tokenizer.apply_chat_template(row['raw_prompt'],add_generation_prompt=True,tokenize=False)
        ids=row['raw_prompt_ids']
        if ids!=self.tokenizer.encode(prompt,add_special_tokens=False): raise ValueError('Prompt/token alignment mismatch')
        return prompt,ids
