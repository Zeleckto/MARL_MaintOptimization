"""
utils/distributions.py
=======================
SINGLE SOURCE OF TRUTH for all stochastic sampling in the project.

Every random draw anywhere in the codebase must go through this file.
This ensures:
    1. Switching stochasticity phases only requires changing calls here
    2. All distributions are unit-testable in isolation
    3. No hidden np.random calls scattered across transition files

Architecture Decision (from design doc):
    - Phase 1: only sample_weibull_failure() is stochastic
    - Phase 2: adds sample_repair_effectiveness(), sample_processing_time()
    - Phase 3: adds sample_job_arrivals(), sample_lead_time(), sample_resource_requirement()

All functions take explicit rng: np.random.Generator for reproducibility.
Never use np.random.xxx() directly — always pass an rng object.

Usage:
    rng = np.random.default_rng(seed=42)
    failed = sample_weibull_failure(virtual_age=150.0, beta=2.8, eta=3000.0, dt=8.0, rng=rng)
"""

import numpy as np
from typing import Optional


# =============================================================================
# WEIBULL FAILURE SAMPLING
# =============================================================================

def sample_weibull_failure(
    virtual_age: float,
    beta: float,
    eta: float,
    dt: float,
    rng: np.random.Generator,
) -> bool:
    """
    Samples whether a machine fails during one timestep [t, t+dt].

    Method: conditional probability given the machine survived to virtual_age.
    This is the correct way to sample Weibull failures incrementally —
    NOT sampling a full lifetime and comparing to current age.

    Weibull survival function: R(t) = exp(-(t/eta)^beta)
    Conditional failure probability in [t, t+dt]:
        P(fail | survived to t) = 1 - R(t+dt) / R(t)

    Args:
        virtual_age: Kijima virtual age v_m (hours equivalent). NOT raw time.
                     This accounts for imperfect repair history.
        beta:        Weibull shape parameter (beta > 1 = wear-out, our case)
        eta:         Weibull scale parameter (characteristic life in hours)
        dt:          Timestep duration in hours (1 shift = 8 hours)
        rng:         NumPy random Generator (seeded, passed in — never created here)

    Returns:
        True if machine fails this timestep, False otherwise.

    Example (M1, CNC Mill):
        failed = sample_weibull_failure(
            virtual_age=2500.0, beta=2.8, eta=3000.0, dt=8.0, rng=rng
        )
    """
    # Survival probability at current virtual age
    r_t = np.exp(-((virtual_age / eta) ** beta))

    # Survival probability at end of timestep
    r_t_dt = np.exp(-(((virtual_age + dt) / eta) ** beta))

    # Avoid division by zero if machine is already at end of life
    if r_t < 1e-10:
        return True  # machine is effectively dead

    # Conditional probability of failure in this interval
    p_fail = 1.0 - (r_t_dt / r_t)

    # Clip to [0, 1] for numerical safety
    p_fail = float(np.clip(p_fail, 0.0, 1.0))

    return bool(rng.random() < p_fail)


def compute_weibull_hazard_rate(virtual_age: float, beta: float, eta: float) -> float:
    """
    Computes instantaneous Weibull hazard rate lambda(t) = (beta/eta) * (t/eta)^(beta-1).
    Used as a state feature in Agent 1's observation — tells the agent how dangerous
    the current machine age is.

    Args:
        virtual_age: Current Kijima virtual age (hours)
        beta:        Weibull shape parameter
        eta:         Weibull scale parameter (hours)

    Returns:
        Hazard rate (failures per hour). Higher = more dangerous.
    """
    if virtual_age <= 0:
        return 0.0
    return float((beta / eta) * ((virtual_age / eta) ** (beta - 1)))


def compute_weibull_rul(virtual_age: float, beta: float, eta: float) -> float:
    """
    Estimates Remaining Useful Life (RUL) from current virtual age.
    Uses the mean residual life integral approximation:
        E[T - t | T > t] = integral from t to inf of R(s)/R(t) ds

    Approximated numerically over a reasonable horizon.

    Args:
        virtual_age: Current Kijima virtual age (hours)
        beta:        Weibull shape parameter
        eta:         Weibull scale parameter (hours)

    Returns:
        Estimated remaining useful life in hours.
    """
    # Survival at current age
    r_t = np.exp(-((virtual_age / eta) ** beta))

    if r_t < 1e-10:
        return 0.0  # already at end of life

    # Numerical integration: sample survival function from current age onwards
    # Integrate up to 5x eta (covers >99.9% of remaining life distribution)
    horizon = max(5.0 * eta, virtual_age + 1000.0)
    t_grid = np.linspace(virtual_age, horizon, num=500)
    r_grid = np.exp(-((t_grid / eta) ** beta))

    # Trapezoidal integration of R(s)/R(t) from t to horizon
    integrand = r_grid / r_t
    rul = float(np.trapezoid(integrand, t_grid))

    return max(rul, 0.0)


def classify_bathtub_phase(beta: float) -> int:
    """
    Maps Weibull shape parameter to bathtub curve phase.
    Used as a discrete state feature for Agent 1.

    Args:
        beta: Weibull shape parameter

    Returns:
        0 = infant mortality (beta < 1)
        1 = useful life / random failures (beta == 1)
        2 = wear-out (beta > 1)

    Our machines all have beta > 1, so they'll always return 2.
    Keeping this generic for extensibility.
    """
    if beta < 1.0:
        return 0  # infant mortality: decreasing hazard rate
    elif beta == 1.0:
        return 1  # random failures: constant hazard rate (exponential)
    else:
        return 2  # wear-out: increasing hazard rate (our case)


# =============================================================================
# KIJIMA REPAIR EFFECTIVENESS SAMPLING (Phase 2+)
# =============================================================================

def sample_repair_effectiveness(
    alpha_q: float,
    beta_q: float,
    rng: np.random.Generator,
) -> float:
    """
    Samples repair effectiveness q from Beta(alpha_q, beta_q).
    Used in Kijima Type I model: v_new = v_old + q * X_n

    Beta distribution is ideal because:
        - Bounded in [0, 1] (q=0 is perfect repair, q=1 is minimal repair)
        - Flexible shape with just two parameters
        - Mean = alpha_q / (alpha_q + beta_q)

    Phase 1: Use fixed q = 0.5 (call with deterministic=True or just pass 0.5 directly)
    Phase 2+: Sample from Beta

    Args:
        alpha_q: Beta distribution alpha parameter (controls mean and shape)
        beta_q:  Beta distribution beta parameter
        rng:     NumPy random Generator

    Returns:
        Repair effectiveness q in (0, 1)

    Typical values for realistic repair:
        alpha_q=5, beta_q=5 => mean=0.5, moderate uncertainty
        alpha_q=2, beta_q=8 => mean=0.2, mostly good repairs
    """
    q = float(rng.beta(alpha_q, beta_q))
    # Clip away extremes to avoid numerical issues in Kijima update
    return float(np.clip(q, 0.01, 0.99))


# =============================================================================
# PROCESSING TIME SAMPLING (Phase 2+)
# =============================================================================

def sample_processing_time(
    nominal_time: float,
    sigma_log: float,
    rng: np.random.Generator,
) -> float:
    """
    Samples actual processing time from LogNormal(mu, sigma^2).
    LogNormal is standard for processing times because:
        - Always positive
        - Right-skewed (rare long delays, common near-nominal times)
        - Parameterised naturally from nominal time

    We parameterise by nominal (deterministic) time and log-space std deviation.
    mu_log = log(nominal) - sigma_log^2 / 2  so that E[X] = nominal_time

    Phase 1: Returns nominal_time directly (no sampling)
    Phase 2+: Returns LogNormal sample

    Args:
        nominal_time: Deterministic processing time from problem instance (hours)
        sigma_log:    Log-space standard deviation. 0.1 = ~10% CV, 0.2 = ~20% CV
        rng:          NumPy random Generator

    Returns:
        Sampled processing time in hours (always positive)
    """
    # Parameterise so mean equals nominal_time
    mu_log = np.log(nominal_time) - 0.5 * (sigma_log ** 2)
    sampled = float(rng.lognormal(mean=mu_log, sigma=sigma_log))

    # Hard floor at 50% of nominal — extreme values destabilise training
    return max(sampled, 0.5 * nominal_time)


# =============================================================================
# JOB ARRIVAL SAMPLING (Phase 3+)
# =============================================================================

def sample_job_arrivals(lambda_arr: float, rng: np.random.Generator) -> int:
    """
    Samples number of new jobs arriving this timestep from Poisson(lambda_arr).
    Poisson is appropriate for job arrivals because:
        - Counts non-negative integers
        - Memoryless (arrivals independent between shifts)
        - Single parameter lambda = mean arrival rate

    Phase 1+2: Returns 0 (no dynamic arrivals — batch released at episode start)
    Phase 3:   Returns Poisson sample

    Args:
        lambda_arr: Mean jobs per timestep (e.g., 0.5 = avg 1 job every 2 shifts)
        rng:        NumPy random Generator

    Returns:
        Number of new jobs arriving this timestep (0, 1, 2, ...)
    """
    return int(rng.poisson(lam=lambda_arr))


# =============================================================================
# LEAD TIME SAMPLING (Phase 3+)
# =============================================================================

def sample_lead_time(
    nominal_lead: float,
    sigma_log: float,
    rng: np.random.Generator,
) -> int:
    """
    Samples actual replenishment lead time from LogNormal.
    Same reasoning as processing times: always positive, right-skewed delivery delays.

    Rounds to nearest integer (lead times are in whole timesteps).

    Phase 1+2: Returns int(nominal_lead) directly
    Phase 3:   Returns rounded LogNormal sample

    Args:
        nominal_lead: Nominal lead time in timesteps
        sigma_log:    Log-space std deviation
        rng:          NumPy random Generator

    Returns:
        Actual lead time in timesteps (integer >= 1)
    """
    mu_log = np.log(nominal_lead) - 0.5 * (sigma_log ** 2)
    sampled = float(rng.lognormal(mean=mu_log, sigma=sigma_log))
    return max(1, int(round(sampled)))  # minimum 1 timestep lead time


# =============================================================================
# RESOURCE REQUIREMENT SAMPLING (Phase 3+)
# =============================================================================

def sample_resource_requirement(
    nominal_requirement: float,
    rng: np.random.Generator,
) -> int:
    """
    Samples actual resource requirement from Poisson(nominal_requirement).
    Small perturbations around nominal — technician or part count varies slightly.

    Phase 1+2: Returns int(nominal_requirement) directly
    Phase 3:   Returns Poisson sample

    Args:
        nominal_requirement: Nominal resource units required (e.g., 2 technicians)
        rng:                 NumPy random Generator

    Returns:
        Actual resource units required (integer >= 0)
    """
    return int(rng.poisson(lam=max(nominal_requirement, 0.01)))


# =============================================================================
# MAINTENANCE DURATION SAMPLING (future use)
# =============================================================================

def sample_maintenance_duration(
    nominal_duration: float,
    delta: float,
    rng: np.random.Generator,
) -> float:
    """
    Samples PM/CM duration from Uniform(nominal - delta, nominal + delta).
    Bounded perturbation around nominal maintenance time.

    Currently not used in Phase 1-3 but included for completeness.

    Args:
        nominal_duration: Nominal maintenance duration in hours
        delta:            Half-width of uniform distribution (hours)
        rng:              NumPy random Generator

    Returns:
        Sampled duration in hours (always > 0)
    """
    low = max(0.5 * nominal_duration, nominal_duration - delta)
    high = nominal_duration + delta
    return float(rng.uniform(low=low, high=high))
