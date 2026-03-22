"""
tier1/cp_sat_solver.py
========================
PLACEHOLDER — OR-Tools CP-SAT solver for Tier 1 exact optimization.
Implements all 38 constraints from the report (Appendix A).
Will be implemented after Tier 3 training pipeline is validated.

When implemented, will:
    - Accept benchmark instance config (M<=5, J<=10)
    - Solve to optimality using CP-SAT Branch and Bound
    - Return solution in same format as Tier 3 for benchmarks/evaluate.py comparison
"""
# TODO: implement Tier 1

def solve(config: dict, time_limit_sec: int = 300) -> dict:
    raise NotImplementedError(
        "Tier 1 CP-SAT solver not yet implemented. "
        "Focus first on Tier 3 training pipeline validation."
    )
