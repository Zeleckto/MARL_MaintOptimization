"""
tests/test_gae.py
==================
Unit tests for GAE (Generalised Advantage Estimation) computation.

The most critical test here is the truncation vs termination bootstrap
distinction. Getting this wrong causes incorrect advantage estimates
that silently degrade training performance.

Reference: architecture doc §7.3

Run with: python tests/test_gae.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from training.rollout_buffer import AgentBuffer, RolloutBuffer


def make_buffer(rewards, values, dones, truncated=None):
    """Helper: fills buffer with given sequence."""
    buf = AgentBuffer(capacity=len(rewards))
    if truncated is None:
        truncated = [False] * len(rewards)
    for r, v, d, tr in zip(rewards, values, dones, truncated):
        buf.add(obs=None, action=None, log_prob=0.0,
                reward=r, value=v, done=d, truncated=tr)
    return buf


# ─── Basic GAE correctness ────────────────────────────────────────────────────

def test_single_step_gae_terminated():
    """
    Single step, terminated episode.
    delta = r + gamma*0 - V = r - V
    A = delta (lambda term is 0 since T=1)
    """
    buf = make_buffer([1.0], [0.5], [True])
    buf.compute_gae(last_value=0.0, gamma=0.99, gae_lambda=0.95)
    expected_adv = 1.0 - 0.5   # r - V (no future, bootstrap=0)
    assert abs(buf.advantages[0] - expected_adv) < 1e-5, (
        f"Expected {expected_adv}, got {buf.advantages[0]}"
    )
    print(f"PASS: single step terminated GAE = {buf.advantages[0]:.4f} (expected {expected_adv:.4f})")


def test_single_step_gae_truncated():
    """
    Single step, truncated episode (hit T_max).
    Bootstrap value = V(s_T) != 0.
    delta = r + gamma*last_value - V
    """
    buf = make_buffer([1.0], [0.5], [True], truncated=[True])
    last_value = 2.0
    buf.compute_gae(last_value=last_value, gamma=0.99, gae_lambda=0.95)
    expected_adv = 1.0 + 0.99 * last_value - 0.5
    assert abs(buf.advantages[0] - expected_adv) < 1e-5, (
        f"Expected {expected_adv:.4f}, got {buf.advantages[0]:.4f}"
    )
    print(f"PASS: single step truncated GAE = {buf.advantages[0]:.4f} (expected {expected_adv:.4f})")


def test_terminated_uses_zero_bootstrap():
    """
    CRITICAL: terminated episode must use bootstrap=0, not V(s_T).
    Passing last_value != 0 to a terminated episode would overestimate future returns.
    The buffer must override last_value=0 when done=True and truncated=False.
    """
    buf_term  = make_buffer([1.0], [0.5], [True],  truncated=[False])
    buf_trunc = make_buffer([1.0], [0.5], [True],  truncated=[True])

    buf_term.compute_gae(last_value=5.0, gamma=0.99, gae_lambda=0.95)
    buf_trunc.compute_gae(last_value=5.0, gamma=0.99, gae_lambda=0.95)

    # Terminated should ignore last_value=5.0 and use 0
    expected_term  = 1.0 - 0.5              # r - V, bootstrap=0
    expected_trunc = 1.0 + 0.99*5.0 - 0.5  # r + gamma*last_value - V

    assert abs(buf_term.advantages[0]  - expected_term)  < 1e-5
    assert abs(buf_trunc.advantages[0] - expected_trunc) < 1e-5
    assert buf_trunc.advantages[0] > buf_term.advantages[0]

    print(f"PASS: terminated bootstrap=0 ({buf_term.advantages[0]:.4f}) "
          f"!= truncated bootstrap ({buf_trunc.advantages[0]:.4f})")


def test_gae_multi_step():
    """
    3-step episode, not done until last step.
    Manually compute GAE backwards and verify.
    gamma=0.9, lambda=0.8
    """
    gamma, lam = 0.9, 0.8
    rewards = [1.0, 0.5, 2.0]
    values  = [0.8, 0.6, 0.4]
    dones   = [False, False, True]

    buf = make_buffer(rewards, values, dones)
    buf.compute_gae(last_value=0.0, gamma=gamma, gae_lambda=lam)

    # Manual computation:
    # t=2: delta2 = 2.0 + gamma*0 - 0.4 = 1.6      gae2 = 1.6
    # t=1: delta1 = 0.5 + gamma*0.4 - 0.6 = 0.26   gae1 = 0.26 + gamma*lam*1.6 = 0.26 + 1.152 = 1.412
    # t=0: delta0 = 1.0 + gamma*0.6 - 0.8 = 0.74   gae0 = 0.74 + gamma*lam*1.412 = 0.74 + 1.0167 = 1.7567

    expected = [1.7567, 1.412, 1.6]
    for t in range(3):
        assert abs(buf.advantages[t] - expected[t]) < 1e-3, (
            f"Step {t}: expected {expected[t]:.4f}, got {buf.advantages[t]:.4f}"
        )
    print(f"PASS: 3-step GAE correct: {[f'{a:.4f}' for a in buf.advantages]}")


def test_returns_equal_advantages_plus_values():
    """Returns = advantages + values (by definition)."""
    buf = make_buffer([1.0, 0.5, 2.0], [0.8, 0.6, 0.4], [False, False, True])
    buf.compute_gae(last_value=0.0, gamma=0.99, gae_lambda=0.95)

    for t in range(3):
        expected_return = buf.advantages[t] + buf.values[t]
        assert abs(buf.returns[t] - expected_return) < 1e-5, (
            f"Step {t}: returns ({buf.returns[t]:.4f}) != adv+val ({expected_return:.4f})"
        )
    print("PASS: returns = advantages + values at all steps")


def test_advantages_normalised_in_minibatch():
    """
    Advantages should be normalised to mean~0, std~1 before PPO update.
    Test that the buffer contains raw advantages (normalisation done in ppo_update.py).
    """
    rewards = list(np.random.default_rng(42).normal(0, 10, 50))
    values  = list(np.random.default_rng(43).normal(0, 10, 50))
    dones   = [False] * 49 + [True]
    buf = make_buffer(rewards, values, dones)
    buf.compute_gae(last_value=0.0, gamma=0.99, gae_lambda=0.95)

    # Raw advantages should NOT be normalised (that happens in PPO update)
    adv_std = np.std(buf.advantages)
    assert adv_std > 0.1, "Buffer should store raw un-normalised advantages"
    print(f"PASS: raw advantages stored (std={adv_std:.3f}, normalisation in PPO update)")


def test_minibatch_generator_covers_all_data():
    """All transitions must appear in at least one minibatch."""
    T = 100
    buf = make_buffer(
        list(range(T)),
        [0.0]*T,
        [False]*(T-1) + [True]
    )
    buf.compute_gae(0.0, 0.99, 0.95)

    seen = set()
    for mb in buf.get_minibatches(minibatch_size=32, shuffle=False):
        # Actions are None here — use rewards as proxy
        for r in mb["advantages"]:
            seen.add(round(float(r), 3))

    assert len(seen) == T, f"Expected {T} unique transitions, got {len(seen)}"
    print(f"PASS: minibatch generator covers all {T} transitions")


def test_rollout_buffer_combined():
    """RolloutBuffer (both agents) computes GAE independently per agent."""
    from training.rollout_buffer import RolloutBuffer
    config = {"mappo": {"rollout_steps": 10}}
    rb = RolloutBuffer(config)

    for i in range(10):
        rb.add(
            obs1=None, action1=None, logp1=0.0, r1=float(i),   v1=0.5,
            obs2=None, action2=None, logp2=0.0, r2=float(i*2), v2=0.5,
            done=(i == 9),
        )

    rb.compute_gae(0.0, 0.0, gamma=0.99, gae_lambda=0.95)
    assert len(rb.buffer1.advantages) == 10
    assert len(rb.buffer2.advantages) == 10
    assert not np.allclose(rb.buffer1.advantages, rb.buffer2.advantages), (
        "Agent 1 and 2 advantages should differ (different rewards)"
    )
    print("PASS: RolloutBuffer computes GAE independently per agent")


if __name__ == "__main__":
    tests = [
        test_single_step_gae_terminated,
        test_single_step_gae_truncated,
        test_terminated_uses_zero_bootstrap,
        test_gae_multi_step,
        test_returns_equal_advantages_plus_values,
        test_advantages_normalised_in_minibatch,
        test_minibatch_generator_covers_all_data,
        test_rollout_buffer_combined,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t(); passed += 1
        except Exception as e:
            print(f"FAIL: {t.__name__}: {e}"); failed += 1
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
