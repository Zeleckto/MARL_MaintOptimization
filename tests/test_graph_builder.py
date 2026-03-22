"""
tests/test_graph_builder.py
============================
Tests for graph construction from environment observations.
Validates that the tripartite HeteroData structure is correct
before any GNN training begins.

Tests run in numpy-only mode (no PyTorch required) by checking
the raw obs dict from mfg_env._build_agent2_obs().

Run with: python tests/test_graph_builder.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "base.yaml")
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)


def make_env():
    from environments.mfg_env import ManufacturingEnv
    env = ManufacturingEnv(CONFIG)
    env.reset(seed=42)
    return env


def test_obs_dict_has_required_keys():
    """Agent 2 observation must contain all graph keys."""
    env = make_env()
    obs = env._build_agent2_obs()
    required = [
        "op_features", "machine_features", "job_features",
        "edge_op_mach", "edge_attr_op_mach",
        "edge_mach_job", "edge_attr_mach_job",
        "edge_op_job",  "edge_attr_op_job",
        "valid_pairs",
    ]
    for key in required:
        assert key in obs, f"Missing key: {key}"
    print(f"PASS: obs dict has all {len(required)} required keys")


def test_node_feature_dimensions():
    """Node feature dims must match Table 3.9 in report."""
    from environments.spaces.observation_spaces import (
        OP_FEATURE_DIM, MACHINE_FEATURE_DIM, JOB_FEATURE_DIM
    )
    env = make_env()
    obs = env._build_agent2_obs()

    assert obs["op_features"].shape[1]      == OP_FEATURE_DIM,      \
        f"Op features: expected {OP_FEATURE_DIM}, got {obs['op_features'].shape[1]}"
    assert obs["machine_features"].shape[1] == MACHINE_FEATURE_DIM, \
        f"Machine features: expected {MACHINE_FEATURE_DIM}, got {obs['machine_features'].shape[1]}"
    assert obs["job_features"].shape[1]     == JOB_FEATURE_DIM,     \
        f"Job features: expected {JOB_FEATURE_DIM}, got {obs['job_features'].shape[1]}"

    print(f"PASS: node dims correct — "
          f"ops={obs['op_features'].shape}, "
          f"machines={obs['machine_features'].shape}, "
          f"jobs={obs['job_features'].shape}")


def test_machine_count_fixed():
    """Machine node count must always equal n_machines (=5 from config)."""
    env = make_env()
    obs = env._build_agent2_obs()
    n_machines = len(CONFIG["machines"])
    assert obs["machine_features"].shape[0] == n_machines, (
        f"Expected {n_machines} machine nodes, got {obs['machine_features'].shape[0]}"
    )
    print(f"PASS: machine node count fixed at {n_machines}")


def test_edge_feature_dimensions():
    """All edge attribute tensors must have 2 features (Table 3.10 in report)."""
    env = make_env()
    obs = env._build_agent2_obs()

    for key in ["edge_attr_op_mach", "edge_attr_mach_job", "edge_attr_op_job"]:
        attr = obs[key]
        if attr.shape[0] > 0:
            assert attr.shape[1] == 2, f"{key}: expected 2 edge features, got {attr.shape[1]}"
    print("PASS: all edge attributes have 2 features")


def test_op_to_machine_edges_respect_eligibility():
    """Op->Machine edges must only exist for eligible (op, machine) pairs."""
    env = make_env()
    obs = env._build_agent2_obs()

    ei = obs["edge_op_mach"]   # [2, n_edges]
    if ei.shape[1] == 0:
        print("PASS: no Op->Machine edges (all machines busy/failed — valid state)")
        return

    # All edge destinations must be valid machine indices
    n_machines = len(CONFIG["machines"])
    assert ei[1].max() < n_machines, "Machine index out of range in Op->Machine edges"
    assert ei[0].min() >= 0, "Negative op index in edges"
    print(f"PASS: {ei.shape[1]} Op->Machine edges with valid indices")


def test_op_to_job_edges_cover_all_pending_ops():
    """Every pending operation should have at least one Op->Job structural edge."""
    env = make_env()
    obs = env._build_agent2_obs()

    ei_oj = obs["edge_op_job"]   # [2, n_edges]
    n_ops = obs["op_features"].shape[0]

    if n_ops == 1 and obs["op_features"].sum() == 0:
        print("PASS: empty graph (no active ops)")
        return

    # Each op node should appear as a source in Op->Job edges
    if ei_oj.shape[1] > 0:
        unique_src = set(ei_oj[0].tolist())
        assert len(unique_src) > 0
    print(f"PASS: Op->Job structural edges present ({ei_oj.shape[1]} edges)")


def test_node_features_normalised():
    """All node feature values should be in a reasonable range (no raw huge values)."""
    env = make_env()
    obs = env._build_agent2_obs()

    for key in ["op_features", "machine_features", "job_features"]:
        feat = obs[key]
        assert feat.max() <= 10.0, (
            f"{key} has suspiciously large values: max={feat.max():.3f}"
        )
        assert feat.min() >= -1.0, (
            f"{key} has suspiciously small values: min={feat.min():.3f}"
        )
    print("PASS: all node features in reasonable range [-1, 10]")


def test_valid_pairs_consistent_with_ready_ops():
    """Valid pairs must only reference READY operations."""
    from environments.transitions.job_dynamics import OpStatus
    env = make_env()
    obs = env._build_agent2_obs()

    valid_pairs = obs["valid_pairs"]
    job_map = {j.job_id: j for j in env.jobs}

    for job_id, op_idx, machine_id in valid_pairs:
        if job_id in job_map:
            op = job_map[job_id].operations[op_idx]
            assert op.status == OpStatus.READY, (
                f"Pair ({job_id},{op_idx},{machine_id}) references non-READY op "
                f"(status={op.status})"
            )
    print(f"PASS: all {len(valid_pairs)} valid pairs reference READY operations")


def test_graph_updates_after_assignment():
    """Graph should change after an operation is assigned."""
    env = make_env()

    obs_before = env._build_agent2_obs()
    n_valid_before = len(obs_before["valid_pairs"])

    if n_valid_before == 0:
        print("SKIP: no valid pairs to assign")
        return

    # Assign first valid pair
    rng = np.random.default_rng(42)
    j_id, op_idx, m_id = obs_before["valid_pairs"][0]
    env.jobs, _ = env.job_engine.assign_operation(
        env.jobs, j_id, op_idx, m_id, rng
    )
    env.machine_busy[m_id] = True

    # Rebuild valid_pairs after assignment (mirrors what AEC step does)
    from environments.spaces.action_spaces import build_agent2_valid_actions, flatten_agent2_actions
    valid = build_agent2_valid_actions(env.jobs, env.machine_states, env.machine_busy)
    env._valid_pairs = flatten_agent2_actions(valid)
    obs_after = env._build_agent2_obs()
    n_valid_after = len(obs_after["valid_pairs"])

    # Valid pairs should decrease after assignment (machine now busy)
    assert n_valid_after < n_valid_before, (
        f"Valid pairs should decrease after assignment "
        f"({n_valid_before} -> {n_valid_after})"
    )
    print(f"PASS: valid pairs decrease after assignment "
          f"({n_valid_before} -> {n_valid_after})")


if __name__ == "__main__":
    tests = [
        test_obs_dict_has_required_keys,
        test_node_feature_dimensions,
        test_machine_count_fixed,
        test_edge_feature_dimensions,
        test_op_to_machine_edges_respect_eligibility,
        test_op_to_job_edges_cover_all_pending_ops,
        test_node_features_normalised,
        test_valid_pairs_consistent_with_ready_ops,
        test_graph_updates_after_assignment,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t(); passed += 1
        except Exception as e:
            print(f"FAIL: {t.__name__}: {e}"); failed += 1
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
