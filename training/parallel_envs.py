"""
training/parallel_envs.py
==========================
Parallel environment management for rollout collection.

Runs 4 environment instances on CPU in separate processes.
Batches their observations for a single GPU forward pass.
Uses Python multiprocessing with shared memory pipes.

Architecture (from design doc §12):
    Process 1: env_1 -> obs -> pipe
    Process 2: env_2 -> obs -> pipe
    Process 3: env_3 -> obs -> pipe
    Process 4: env_4 -> obs -> pipe
                              |
                    batch graphs (CPU)
                    PyG Batch.from_data_list
                              |
                    GPU forward pass (TGIN)
                              |
                    actions back to processes

Usage:
    vec_env = VecManufacturingEnv(config, n_envs=4)
    obs_list, info_list = vec_env.reset()
    obs_list, rewards, dones, infos = vec_env.step(actions)
"""

import multiprocessing as mp
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from environments.mfg_env import ManufacturingEnv
from utils.seeding import make_env_seed


def _worker(
    env_idx:     int,
    config:      dict,
    seed:        int,
    conn:        mp.connection.Connection,
) -> None:
    """
    Worker process: runs one environment instance.
    Listens for commands over a pipe connection.

    Commands:
        ('reset', None)         -> sends (obs, info)
        ('step', action)        -> sends (obs, r1, r2, term, trunc, info)
        ('close', None)         -> exits
    """
    env = ManufacturingEnv(config)
    obs, info = env.reset(seed=seed)

    while True:
        try:
            cmd, data = conn.recv()

            if cmd == "reset":
                obs, info = env.reset(seed=seed)
                conn.send((obs, info))

            elif cmd == "step":
                # data = (action1, action2)
                action1, action2 = data

                # Agent 1 half-step
                env.agent_selection = "pdm_agent"
                env.step(action1)

                # Agent 2 half-step + physics
                env.agent_selection = "jobshop_agent"
                env.step(action2)

                obs1 = env.observe("pdm_agent")
                obs2 = env.observe("jobshop_agent")
                r1   = env.rewards["pdm_agent"]
                r2   = env.rewards["jobshop_agent"]
                term = env.terminations["pdm_agent"]
                trunc = env.truncations["pdm_agent"]
                info = env.infos["pdm_agent"]

                conn.send(((obs1, obs2), r1, r2, term, trunc, info))

                # Auto-reset on episode end
                if term or trunc:
                    obs, info = env.reset(seed=seed)

            elif cmd == "close":
                env.close()
                conn.close()
                break

        except EOFError:
            break


class VecManufacturingEnv:
    """
    Vectorised wrapper running n_envs environment instances in parallel.

    Each env runs in its own process.
    Main process batches observations and dispatches actions.
    """

    def __init__(self, config: dict, n_envs: int = 4):
        self.config  = config
        self.n_envs  = n_envs
        base_seed    = config.get("seed", 42)

        # Create pipes and worker processes
        self.parent_conns = []
        self.processes    = []

        for i in range(n_envs):
            parent_conn, child_conn = mp.Pipe()
            seed = make_env_seed(base_seed, i)

            p = mp.Process(
                target=_worker,
                args=(i, config, seed, child_conn),
                daemon=True,
            )
            p.start()
            child_conn.close()  # close child end in parent process

            self.parent_conns.append(parent_conn)
            self.processes.append(p)


    def reset(self) -> Tuple[List[dict], List[dict]]:
        """
        Resets all environments and returns initial observations.

        Returns:
            (obs1_list, obs2_list)  — one obs per env for each agent
        """
        for conn in self.parent_conns:
            conn.send(("reset", None))

        results = [conn.recv() for conn in self.parent_conns]
        obs1_list = [r[0]["pdm_agent"]     for r in results]
        obs2_list = [r[0]["jobshop_agent"] for r in results]

        return obs1_list, obs2_list


    def step(
        self,
        actions1: List[dict],                     # Agent 1 actions per env
        actions2: List[Optional[Tuple]],          # Agent 2 actions per env
    ) -> Tuple[List, List, List[float], List[float], List[bool], List[bool], List[dict]]:
        """
        Steps all environments simultaneously.

        Args:
            actions1: [n_envs] Agent 1 action dicts
            actions2: [n_envs] Agent 2 action indices (int or None=WAIT)

        Returns:
            (obs1_list, obs2_list, r1_list, r2_list, term_list, trunc_list, info_list)
        """
        # Send actions to all workers
        for conn, a1, a2 in zip(self.parent_conns, actions1, actions2):
            conn.send(("step", (a1, a2)))

        # Collect results
        results = [conn.recv() for conn in self.parent_conns]

        obs_list  = [r[0] for r in results]
        obs1_list = [o[0] for o in obs_list]
        obs2_list = [o[1] for o in obs_list]
        r1_list   = [r[1] for r in results]
        r2_list   = [r[2] for r in results]
        term_list = [r[3] for r in results]
        trunc_list = [r[4] for r in results]
        info_list = [r[5] for r in results]

        return obs1_list, obs2_list, r1_list, r2_list, term_list, trunc_list, info_list


    def close(self) -> None:
        for conn in self.parent_conns:
            conn.send(("close", None))
        for p in self.processes:
            p.join(timeout=5)
