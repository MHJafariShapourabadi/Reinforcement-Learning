import multiprocessing as mp
import numpy as np
from multiprocessing import shared_memory

# Use "spawn" start method to be safe on all platforms.
mp.set_start_method("spawn", force=True)

# =============================================================================
# Worker Process Function with Shared Memory for obs, act, rew, term, trunc
# =============================================================================
def worker(remote, parent_remote, env_fn, env_fn_args,
           shm_obs_name, obs_shape, obs_dtype,
           shm_act_name, act_shape, act_dtype,
           shm_rew_name, rew_shape, rew_dtype,
           shm_term_name, term_shape, term_dtype,
           shm_trunc_name, trunc_shape, trunc_dtype,
           shm_I_name, I_shape, I_dtype,
           idx):
    """
    Worker process that runs an environment instance and communicates via a pipe.
    It writes its latest observation into shared memory at index idx, reads its action
    from shared memory, and writes the resulting reward, terminated, and truncated values
    into shared memory.
    """
    parent_remote.close()
    
    # Open shared memory blocks.
    shm_obs = shared_memory.SharedMemory(name=shm_obs_name)
    obs_buf = np.ndarray(obs_shape, dtype=obs_dtype, buffer=shm_obs.buf)
    
    shm_act = shared_memory.SharedMemory(name=shm_act_name)
    act_buf = np.ndarray(act_shape, dtype=act_dtype, buffer=shm_act.buf)
    
    shm_rew = shared_memory.SharedMemory(name=shm_rew_name)
    rew_buf = np.ndarray(rew_shape, dtype=rew_dtype, buffer=shm_rew.buf)
    
    shm_term = shared_memory.SharedMemory(name=shm_term_name)
    term_buf = np.ndarray(term_shape, dtype=term_dtype, buffer=shm_term.buf)
    
    shm_trunc = shared_memory.SharedMemory(name=shm_trunc_name)
    trunc_buf = np.ndarray(trunc_shape, dtype=trunc_dtype, buffer=shm_trunc.buf)

    shm_I = shared_memory.SharedMemory(name=shm_I_name)
    I_buf = np.ndarray(I_shape, dtype=I_dtype, buffer=shm_I.buf)
    
    env = env_fn(*env_fn_args)
    
    try:
        while True:
            cmd = remote.recv()
            if cmd == "reset":
                obs, info = env.reset()
                # print(f"worker: {idx} obs: {obs.shape} obs_buf: {obs_buf.shape}")
                done = False
                np.copyto(obs_buf[idx], np.array(obs, dtype=obs_dtype))
                I_buf[idx] = 0
                episode_steps = 0
                episode_reward = 0
                info['episode_steps'] = episode_steps
                info['episode_reward'] = episode_reward
                info['done'] = done
                remote.send(info)
            elif cmd == "step":               
                act = act_buf[idx]
                obs, rew, terminated, truncated, info = env.step(act)
                # print(f"worker: {idx} obs: {obs.shape} obs_buf: {obs_buf.shape}")
                done = terminated or truncated
                np.copyto(obs_buf[idx], np.array(obs, dtype=obs_dtype))
                rew_buf[idx] = rew
                term_buf[idx] = terminated
                trunc_buf[idx] = truncated
                I_buf[idx] = I_buf[idx] + 1
                episode_steps += 1
                episode_reward += rew
                info['episode_steps'] = episode_steps
                info['episode_reward'] = episode_reward
                info['done'] = done
                remote.send(info)
            elif cmd == "close":
                remote.close()
                break
            else:
                raise NotImplementedError(f"Unknown command: {cmd}")
    except KeyboardInterrupt:
        print(f"Worker {idx}: KeyboardInterrupt")
    finally:
        env.close()
        shm_obs.close()
        shm_act.close()
        shm_rew.close()
        shm_term.close()
        shm_trunc.close()
        shm_I.close()

# =============================================================================
# Custom Async Vectorized Environment with Shared Memory
# =============================================================================
class AsyncVectorEnv:
    """
    A custom asynchronous vectorized environment using standard multiprocessing,
    pipes, and shared memory for obs, act, rew, term, and trunc.
    """
    def __init__(self, env_fns, env_fns_args, obs_dtype=np.float32, act_dtype=np.float32):
        assert len(env_fns) == len(env_fns_args), "Number of functions doesn't match number of function args"
        self.n_envs = len(env_fns)
        self.env_fns = env_fns
        self.env_fns_args = env_fns_args
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.n_envs)])
        self.processes = []
        
        # Determine observation shape.
        self.dummy_env = dummy = env_fns[0](*env_fns_args[0])
        self.observation_space = dummy.observation_space
        self.action_space = dummy.action_space
        self.obs_shape = dummy.observation_space.shape
        self.act_shape = dummy.action_space.shape
        dummy.close()
        
        # Shared memory for observations.
        self.shm_obs_shape = (self.n_envs,) + self.obs_shape
        self.obs_dtype = np.dtype(obs_dtype)
        nbytes_obs = np.prod(self.shm_obs_shape) * self.obs_dtype.itemsize
        self.shm_obs = shared_memory.SharedMemory(create=True, size=int(nbytes_obs))
        self.obs_buf = np.ndarray(self.shm_obs_shape, dtype=self.obs_dtype, buffer=self.shm_obs.buf)
        
        # Shared memory for actions.
        self.shm_act_shape = (self.n_envs,) + self.act_shape
        self.act_dtype = np.dtype(act_dtype)
        nbytes_act = np.prod(self.shm_act_shape) * self.act_dtype.itemsize
        self.shm_act = shared_memory.SharedMemory(create=True, size=int(nbytes_act))
        self.act_buf = np.ndarray(self.shm_act_shape, dtype=self.act_dtype, buffer=self.shm_act.buf)
        
        # Shared memory for rewards.
        self.rew_shape = (self.n_envs,)
        self.rew_dtype = np.float32
        nbytes_rew = np.prod(self.rew_shape) * np.dtype(self.rew_dtype).itemsize
        self.shm_rew = shared_memory.SharedMemory(create=True, size=int(nbytes_rew))
        self.rew_buf = np.ndarray(self.rew_shape, dtype=self.rew_dtype, buffer=self.shm_rew.buf)
        
        # Shared memory for terminated flags.
        self.term_shape = (self.n_envs,)
        self.term_dtype = np.bool_
        nbytes_term = np.prod(self.term_shape) * np.dtype(self.term_dtype).itemsize
        self.shm_term = shared_memory.SharedMemory(create=True, size=int(nbytes_term))
        self.term_buf = np.ndarray(self.term_shape, dtype=self.term_dtype, buffer=self.shm_term.buf)
        
        # Shared memory for truncated flags.
        self.trunc_shape = (self.n_envs,)
        self.trunc_dtype = np.bool_
        nbytes_trunc = np.prod(self.trunc_shape) * np.dtype(self.trunc_dtype).itemsize
        self.shm_trunc = shared_memory.SharedMemory(create=True, size=int(nbytes_trunc))
        self.trunc_buf = np.ndarray(self.trunc_shape, dtype=self.trunc_dtype, buffer=self.shm_trunc.buf)

        # Shared memory for Is.
        self.I_shape = (self.n_envs,)
        self.I_dtype = np.float32
        nbytes_I = np.prod(self.I_shape) * np.dtype(self.I_dtype).itemsize
        self.shm_I = shared_memory.SharedMemory(create=True, size=int(nbytes_I))
        self.I_buf = np.ndarray(self.I_shape, dtype=self.I_dtype, buffer=self.shm_I.buf)
        
        # Spawn worker processes.
        for worker_id, (w_remote, remote, fn, args) in enumerate(zip(self.work_remotes, self.remotes, env_fns, env_fns_args)):
            args = (w_remote, remote, fn, args,
                    self.shm_obs.name, self.shm_obs_shape, self.obs_dtype,
                    self.shm_act.name, self.shm_act_shape, self.act_dtype,
                    self.shm_rew.name, self.rew_shape, self.rew_dtype,
                    self.shm_term.name, self.term_shape, self.term_dtype,
                    self.shm_trunc.name, self.trunc_shape, self.trunc_dtype,
                    self.shm_I.name, self.I_shape, self.I_dtype,
                    worker_id)
            p = mp.Process(target=worker, args=args)
            p.daemon = True
            p.start()
            w_remote.close()
            self.processes.append(p)
    
    def reset(self, worker_id=None):
        if worker_id is None:
            for remote in self.remotes:
                remote.send("reset")
            infos = [remote.recv() for remote in self.remotes]
            # print(f"I_buf: {self.I_buf.shape} obs_buf: {self.obs_buf.shape}")
            return self.I_buf.copy(), self.obs_buf.copy(), infos
        else:
            remote = self.remotes[worker_id]
            remote.send("reset")
            info = remote.recv()
            # print(f"I_buf: {self.I_buf.shape} obs_buf: {self.obs_buf.shape}")
            return self.I_buf.copy(), self.obs_buf.copy(), info

    def step(self, acts):
        # Write actions to shared memory.
        self.act_buf[:] = np.array(acts, dtype=self.act_dtype)
        for remote in self.remotes:
            remote.send("step")
        infos = [remote.recv() for remote in self.remotes]
        # Read step results from shared memory.
        I = self.I_buf.copy()
        obs = self.obs_buf.copy()
        rews = self.rew_buf.copy()
        terms = self.term_buf.copy()
        truncts = self.trunc_buf.copy()
        # print(f"I_buf: {self.I_buf.shape} obs_buf: {self.obs_buf.shape} rew_buf: {self.rew_buf.shape} term_buf: {self.term_buf.shape} trunc_buf: {self.trunc_buf.shape}")
        return I, obs, rews, terms, truncts, infos

    def close(self):
        for remote in self.remotes:
            remote.send("close")
        for p in self.processes:
            p.join()
        self.shm_obs.close()
        self.shm_obs.unlink()
        self.shm_act.close()
        self.shm_act.unlink()
        self.shm_rew.close()
        self.shm_rew.unlink()
        self.shm_term.close()
        self.shm_term.unlink()
        self.shm_trunc.close()
        self.shm_trunc.unlink()
        self.shm_I.close()
        self.shm_I.unlink()