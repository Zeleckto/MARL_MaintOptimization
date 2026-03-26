"""
run_integration_tests.py
=========================
Full integration test suite — runs after unit tests pass.
Tests environment, models, agents, and training pipeline end-to-end.
Place in project ROOT and run with: python run_integration_tests.py

Does NOT require pygame display for rendering test (uses headless check).
"""

import sys
import os
import time
import traceback
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

DIVIDER  = "=" * 70
DIVIDER2 = "-" * 70

results = []   # list of (test_name, passed, message, duration)


def run_test(name, fn):
    """Runs one test function, catches exceptions, records result."""
    print(f"\n{DIVIDER2}")
    print(f"  TEST: {name}")
    print(DIVIDER2)
    t0 = time.time()
    try:
        msg = fn()
        dur = time.time() - t0
        results.append((name, True, msg or "OK", dur))
        print(f"  RESULT: PASS ({dur:.2f}s)")
        if msg:
            print(f"  {msg}")
        return True
    except Exception as e:
        dur = time.time() - t0
        tb = traceback.format_exc()
        results.append((name, False, str(e), dur))
        print(f"  RESULT: FAIL ({dur:.2f}s)")
        print(f"  ERROR: {e}")
        print(f"  TRACEBACK:\n{tb}")
        return False


# =============================================================================
# TEST 1: Config loading
# =============================================================================
def test_config_loading():
    import yaml
    with open("configs/base.yaml") as f:
        config = yaml.safe_load(f)
    assert "machines" in config, "No machines in config"
    assert len(config["machines"]) == 5, f"Expected 5 machines, got {len(config['machines'])}"
    assert "resources" in config
    assert "mappo" in config
    return (f"Config OK: {len(config['machines'])} machines, "
            f"stoch_level={config['stochasticity_level']}")


# =============================================================================
# TEST 2: Environment reset
# =============================================================================
def test_env_reset():
    import yaml
    from environments.mfg_env import ManufacturingEnv, AGENT_PDM, AGENT_JOBSHOP
    with open("configs/base.yaml") as f:
        config = yaml.safe_load(f)
    env = ManufacturingEnv(config)
    obs_dict, info = env.reset(seed=42)
    from environments.mfg_env import AGENT_PDM, AGENT_JOBSHOP
    obs1 = obs_dict[AGENT_PDM]
    assert obs1 is not None, "obs1 is None"
    assert len(obs1) > 0, "obs1 is empty"
    assert len(env.machine_states) == 5, "Wrong machine count"
    assert len(env.jobs) == config["jobs"]["n_jobs_train"]
    obs2 = env._build_agent2_obs()
    assert "op_features" in obs2
    assert "valid_pairs" in obs2
    return (f"Env reset OK: obs1.shape={obs1.shape}, "
            f"jobs={len(env.jobs)}, valid_pairs={len(obs2['valid_pairs'])}")



# =============================================================================
# TEST 3: Environment step — random policy full episode
# =============================================================================
def test_env_random_episode():
    import yaml
    from environments.mfg_env import ManufacturingEnv, AGENT_PDM, AGENT_JOBSHOP
    with open("configs/base.yaml") as f:
        config = yaml.safe_load(f)
    config["episode"]["t_max_train"] = 50
    config["jobs"]["n_jobs_train"] = 5

    env = ManufacturingEnv(config)
    env.reset(seed=42)

    steps = 0
    total_r1 = total_r2 = 0.0
    done = False

    while not done and steps < 50:
        # Agent 1 half-step
        action1 = {
            "maintenance": np.zeros(5, dtype=int),
            "reorder":     np.zeros(3, dtype=float),
        }
        env._step_agent1(action1)

        # Agent 2 half-step + physics
        env._step_agent2(len(env._valid_pairs))  # WAIT
        env._resolve_physics()
        env._compute_rewards()

        r1 = env.rewards[AGENT_PDM]
        r2 = env.rewards[AGENT_JOBSHOP]
        total_r1 += r1
        total_r2 += r2

        term  = env.terminations[AGENT_PDM]
        trunc = env.truncations[AGENT_PDM]
        done  = term or trunc
        steps += 1

    assert steps > 0, "Episode ran 0 steps"
    return (f"Random episode: {steps} steps, "
            f"r1_total={total_r1:.2f}, r2_total={total_r2:.2f}, "
            f"failures={env._episode_failures}, "
            f"completions={env._episode_completions}")


# =============================================================================
# TEST 4: Machine degradation visible in episode
# =============================================================================
def test_machine_degradation_in_episode():
    import yaml
    from environments.mfg_env import ManufacturingEnv
    with open("configs/base.yaml") as f:
        config = yaml.safe_load(f)
    config["episode"]["t_max_train"] = 30
    config["jobs"]["n_jobs_train"] = 3

    env = ManufacturingEnv(config)
    env.reset(seed=42)

    initial_health = [s.health for s in env.machine_states]

    for _ in range(20):
        action1 = {"maintenance": np.zeros(5, dtype=int), "reorder": np.zeros(3)}
        env._step_agent1(action1)
        env._step_agent2(0)
        env._resolve_physics()
        env._compute_rewards()

        done = env.terminations["pdm_agent"] or env.truncations["pdm_agent"]
        if done:
            break

    final_health = [s.health for s in env.machine_states]
    degraded = sum(1 for i, f in zip(initial_health, final_health) if f < i)

    assert degraded > 0, "No machines degraded — check Weibull + busy logic"
    return (f"Degradation visible: {degraded}/5 machines degraded, "
            f"health change: {[f'{i:.1f}->{f:.1f}' for i,f in zip(initial_health, final_health)]}")


# =============================================================================
# TEST 5: GPU availability
# =============================================================================
def test_gpu():
    import torch
    assert torch.cuda.is_available(), "CUDA not available"
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    ver  = torch.version.cuda
    return f"GPU: {name}, VRAM: {vram:.1f}GB, CUDA: {ver}"


# =============================================================================
# TEST 6: Model instantiation on GPU
# =============================================================================
def test_model_instantiation():
    import yaml, torch
    with open("configs/base.yaml") as f:
        config = yaml.safe_load(f)
    config["device"] = "cuda"

    from agents.pdm_agent import PDMAgent
    from agents.jobshop_agent import JobShopAgent
    from models.critic import CentralizedCritic

    agent1 = PDMAgent(config, device="cuda")
    agent2 = JobShopAgent(config, device="cuda")
    critic = CentralizedCritic(config).to("cuda")

    assert agent1.policy is not None
    assert agent2.tgin is not None
    assert agent2.action_scorer is not None

    p1 = sum(p.numel() for p in agent1.policy.parameters())
    p2 = (sum(p.numel() for p in agent2.tgin.parameters()) +
          sum(p.numel() for p in agent2.action_scorer.parameters()))
    p3 = sum(p.numel() for p in critic.parameters())

    return (f"Agent1(MLP)={p1:,} params | "
            f"Agent2(TGIN)={p2:,} params | "
            f"Critic={p3:,} params | "
            f"Total={p1+p2+p3:,}")


# =============================================================================
# TEST 7: TGIN forward pass
# =============================================================================
def test_tgin_forward_pass():
    import yaml, torch
    with open("configs/base.yaml") as f:
        config = yaml.safe_load(f)

    from environments.mfg_env import ManufacturingEnv
    from models.tgin.graph_builder import GraphBuilder
    from models.tgin.tgin import TGIN

    env = ManufacturingEnv(config)
    env.reset(seed=42)
    obs2 = env._build_agent2_obs()

    builder = GraphBuilder(config)
    graph   = builder.build(obs2, device="cuda")
    tgin    = TGIN(config).to("cuda")

    with torch.no_grad():
        embeddings = tgin(graph)

    assert "op" in embeddings
    assert "machine" in embeddings
    assert "job" in embeddings

    hidden = config["tgin"]["hidden_dim"]
    assert embeddings["machine"].shape == (5, hidden), \
        f"Machine emb shape wrong: {embeddings['machine'].shape}"

    return (f"TGIN forward pass OK: "
            f"op={embeddings['op'].shape}, "
            f"machine={embeddings['machine'].shape}, "
            f"job={embeddings['job'].shape}")


# =============================================================================
# TEST 8: Action scorer
# =============================================================================
def test_action_scorer():
    import yaml, torch
    with open("configs/base.yaml") as f:
        config = yaml.safe_load(f)

    from environments.mfg_env import ManufacturingEnv
    from models.tgin.graph_builder import GraphBuilder
    from models.tgin.tgin import TGIN
    from models.tgin.action_scorer import ActionScorer

    env = ManufacturingEnv(config)
    env.reset(seed=42)
    obs2 = env._build_agent2_obs()
    valid_pairs = obs2["valid_pairs"]

    builder = GraphBuilder(config)
    graph   = builder.build(obs2, device="cuda")
    tgin    = TGIN(config).to("cuda")
    scorer  = ActionScorer(config).to("cuda")

    # Build op_id_map
    op_id_map = {(j, k): i for i, (j, k, m) in enumerate(valid_pairs)}

    with torch.no_grad():
        embeddings = tgin(graph)
        dist, logits = scorer(embeddings, valid_pairs, op_id_map)
        action = dist.sample()
        logprob = dist.log_prob(action)

    assert logits.shape[0] == len(valid_pairs) + 1  # +1 for WAIT
    assert not torch.isnan(logits).any(), "NaN in logits"
    assert not torch.isinf(logits).any(), "Inf in logits"

    return (f"Action scorer OK: {len(valid_pairs)} valid pairs + WAIT, "
            f"sampled action={action.item()}, logprob={logprob.item():.4f}")


# =============================================================================
# TEST 9: Agent 1 full act() call
# =============================================================================
def test_agent1_act():
    import yaml
    with open("configs/base.yaml") as f:
        config = yaml.safe_load(f)

    from environments.mfg_env import ManufacturingEnv
    from agents.pdm_agent import PDMAgent

    env = ManufacturingEnv(config)
    obs_dict, _ = env.reset(seed=42)
    from environments.mfg_env import AGENT_PDM
    obs1 = obs_dict[AGENT_PDM]
    agent1 = PDMAgent(config, device="cuda")

    action1, logp1, ent1 = agent1.act(
        obs_np=obs1,
        machine_states=env.machine_states,
        machine_busy=env.machine_busy,
        resource_state=env.resource_state,
        rho_PM=env.rho_PM,
        rho_CM=env.rho_CM,
    )

    assert "maintenance" in action1
    assert "reorder" in action1
    assert len(action1["maintenance"]) == 5
    assert len(action1["reorder"]) == 3
    assert not np.isnan(logp1), "NaN log prob"

    # Verify masking — no invalid actions
    for i, a in enumerate(action1["maintenance"]):
        s = env.machine_states[i]
        from environments.transitions.degradation import MachineStatus
        if a == 1:  # PM
            assert s.status == MachineStatus.OP, f"PM on non-OP machine {i}"
        if a == 2:  # CM
            assert s.status == MachineStatus.FAIL, f"CM on non-FAIL machine {i}"

    return (f"Agent1 act OK: maintenance={action1['maintenance'].tolist()}, "
            f"reorder={action1['reorder'].tolist()}, logp={logp1:.4f}, ent={ent1:.4f}")


# =============================================================================
# TEST 10: Agent 2 full act() call
# =============================================================================
def test_agent2_act():
    import yaml
    with open("configs/base.yaml") as f:
        config = yaml.safe_load(f)

    from environments.mfg_env import ManufacturingEnv
    from agents.jobshop_agent import JobShopAgent

    env = ManufacturingEnv(config)
    env.reset(seed=42)
    obs2 = env._build_agent2_obs()
    agent2 = JobShopAgent(config, device="cuda")

    sem2, idx2, logp2, ent2 = agent2.act(obs2, env._valid_pairs)

    assert idx2 <= len(env._valid_pairs), "Action index out of range"
    if sem2 is not None:
        j, k, m = sem2
        # Verify it's a valid assignment
        from environments.transitions.job_dynamics import OpStatus
        job = next(jb for jb in env.jobs if jb.job_id == j)
        assert job.operations[k].status == OpStatus.READY

    return (f"Agent2 act OK: action={sem2}, idx={idx2}, "
            f"logp={logp2:.4f}, ent={ent2:.4f}")


# =============================================================================
# TEST 11: Full agent loop (5 complete timesteps)
# =============================================================================
def test_full_agent_loop():
    import yaml
    with open("configs/base.yaml") as f:
        config = yaml.safe_load(f)

    from environments.mfg_env import ManufacturingEnv, AGENT_PDM, AGENT_JOBSHOP
    from agents.pdm_agent import PDMAgent
    from agents.jobshop_agent import JobShopAgent

    env    = ManufacturingEnv(config)
    agent1 = PDMAgent(config, device="cuda")
    agent2 = JobShopAgent(config, device="cuda")

    obs_dict, _ = env.reset(seed=42)
    from environments.mfg_env import AGENT_PDM as _PDM
    obs1 = obs_dict[_PDM]
    step_log = []

    for step in range(5):
        obs2 = env._build_agent2_obs()

        action1, logp1, ent1 = agent1.act(
            obs_np=obs1,
            machine_states=env.machine_states,
            machine_busy=env.machine_busy,
            resource_state=env.resource_state,
            rho_PM=env.rho_PM,
            rho_CM=env.rho_CM,
        )
        env._step_agent1(action1)
        obs2 = env._build_agent2_obs()

        sem2, idx2, logp2, ent2 = agent2.act(obs2, env._valid_pairs)
        env._step_agent2(idx2)
        env._resolve_physics()
        env._compute_rewards()

        obs1 = env.observe(AGENT_PDM)
        r1   = env.rewards[AGENT_PDM]
        r2   = env.rewards[AGENT_JOBSHOP]
        step_log.append(f"s{step+1}: r1={r1:.2f} r2={r2:.2f} act={sem2}")

        print(f"    Step {step+1}: r1={r1:.3f} r2={r2:.3f} | "
              f"action={sem2} | failures={env._episode_failures}")

    return f"Full loop OK: {' | '.join(step_log)}"


# =============================================================================
# TEST 12: Rollout buffer + GAE
# =============================================================================
def test_rollout_buffer():
    import yaml
    with open("configs/base.yaml") as f:
        config = yaml.safe_load(f)
    config["mappo"]["rollout_steps"] = 20

    from training.rollout_buffer import RolloutBuffer
    buf = RolloutBuffer(config)

    rng = np.random.default_rng(42)
    for i in range(20):
        buf.add(
            obs1=np.zeros(10), action1={"maintenance": np.zeros(5), "reorder": np.zeros(3)},
            logp1=float(rng.normal()),  r1=float(rng.normal()),  v1=0.5,
            obs2={},                    action2=0,
            logp2=float(rng.normal()),  r2=float(rng.normal()),  v2=0.5,
            done=(i == 19),
        )

    buf.compute_gae(0.0, 0.0, gamma=0.99, gae_lambda=0.95)

    assert len(buf.buffer1.advantages) == 20
    assert len(buf.buffer2.advantages) == 20
    assert not np.any(np.isnan(buf.buffer1.advantages))
    assert not np.any(np.isnan(buf.buffer2.advantages))

    return (f"Buffer OK: 20 transitions, "
            f"adv1 mean={buf.buffer1.advantages.mean():.3f}, "
            f"adv2 mean={buf.buffer2.advantages.mean():.3f}")


# =============================================================================
# TEST 13: Pygame import check (headless)
# =============================================================================
def test_pygame_import():
    import pygame
    ver = pygame.__version__
    # Don't actually open a window — just verify import works
    # Use pygame.display.init() only if display is available
    return f"Pygame {ver} importable OK (window test: run render_episode.py manually)"


# =============================================================================
# TEST 14: Reward function end-to-end
# =============================================================================
def test_reward_fn_end_to_end():
    import yaml
    with open("configs/base.yaml") as f:
        config = yaml.safe_load(f)

    from environments.mfg_env import ManufacturingEnv, AGENT_PDM, AGENT_JOBSHOP
    env = ManufacturingEnv(config)
    env.reset(seed=42)

    action1 = {"maintenance": np.zeros(5, dtype=int), "reorder": np.zeros(3)}
    env._step_agent1(action1)
    env._step_agent2(len(env._valid_pairs))
    env._resolve_physics()
    env._compute_rewards()

    r1 = env.rewards[AGENT_PDM]
    r2 = env.rewards[AGENT_JOBSHOP]

    assert not np.isnan(r1), "r1 is NaN"
    assert not np.isnan(r2), "r2 is NaN"
    assert r1 != 0.0, "r1 is zero every step — dense signal missing"

    return f"Reward fn OK: r1={r1:.4f} (should be ~2.0 avail bonus), r2={r2:.4f}"


# =============================================================================
# TEST 15: Benchmark instance config loading
# =============================================================================
def test_benchmark_configs():
    import yaml
    instances = [
        "configs/benchmark_instances/small_3m_5j.yaml",
        "configs/benchmark_instances/medium_5m_10j.yaml",
        "configs/benchmark_instances/large_5m_20j.yaml",
    ]
    results_inner = []
    for path in instances:
        with open("configs/base.yaml") as f:
            config = yaml.safe_load(f)
        with open(path) as f:
            override = yaml.safe_load(f)
        if override:
            config.update(override)
        n_jobs = config["jobs"]["n_jobs_train"]
        results_inner.append(f"{os.path.basename(path)}: {n_jobs} jobs")

    return " | ".join(results_inner)


# =============================================================================
# MAIN
# =============================================================================
def main():
    print()
    print(DIVIDER)
    print("  MANUFACTURING MARL — INTEGRATION TEST SUITE")
    print(DIVIDER)
    print()

    tests = [
        ("01. Config Loading",              test_config_loading),
        ("02. Environment Reset",           test_env_reset),
        ("03. Random Episode (50 steps)",   test_env_random_episode),
        ("04. Machine Degradation",         test_machine_degradation_in_episode),
        ("05. GPU Availability",            test_gpu),
        ("06. Model Instantiation",         test_model_instantiation),
        ("07. TGIN Forward Pass",           test_tgin_forward_pass),
        ("08. Action Scorer",               test_action_scorer),
        ("09. Agent 1 act()",               test_agent1_act),
        ("10. Agent 2 act()",               test_agent2_act),
        ("11. Full Agent Loop (5 steps)",   test_full_agent_loop),
        ("12. Rollout Buffer + GAE",        test_rollout_buffer),
        ("13. Pygame Import",               test_pygame_import),
        ("14. Reward Function E2E",         test_reward_fn_end_to_end),
        ("15. Benchmark Configs",           test_benchmark_configs),
    ]

    for name, fn in tests:
        run_test(name, fn)

    # Summary
    print()
    print(DIVIDER)
    print("  INTEGRATION TEST SUMMARY")
    print(DIVIDER)

    passed = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]

    for name, ok, msg, dur in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}]  {name:<35} ({dur:.2f}s)")
        if not ok:
            print(f"         -> {msg[:80]}")

    print()
    print(f"  Passed: {len(passed)}/{len(results)}")
    print(f"  Failed: {len(failed)}/{len(results)}")
    print()

    if not failed:
        print("  ALL INTEGRATION TESTS PASSED")
        print("  Next step: python scripts/train.py --config configs/phase1.yaml --timesteps 50000")
    else:
        print("  FAILURES DETECTED — paste this output for debugging")

    print(DIVIDER)
    print()


if __name__ == "__main__":
    main()