"""
run_all_tests.py
================
Runs all test suites and produces a clean summary.
Place this in the project ROOT (manufacturing_marl/).

Run with:
    python run_all_tests.py

Output shows:
    - Pass/fail per test
    - Full traceback for any failure
    - Final summary count
    - Any missing packages
"""

import subprocess
import sys
import os

# Make sure we are in the right directory
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

TEST_FILES = [
    "tests/test_degradation.py",
    "tests/test_kijima.py",
    "tests/test_inventory.py",
    "tests/test_action_masking.py",
    "tests/test_reward_components.py",
    "tests/test_gae.py",
    "tests/test_graph_builder.py",
]

DIVIDER = "=" * 70

def check_packages():
    """Check all required packages are installed."""
    print(DIVIDER)
    print("PACKAGE CHECK")
    print(DIVIDER)
    packages = {
        "numpy":           "numpy",
        "torch":           "torch",
        "torch_geometric": "torch_geometric",
        "pettingzoo":      "pettingzoo",
        "gymnasium":       "gymnasium",
        "ortools":         "ortools",
        "pygame":          "pygame",
        "yaml":            "pyyaml",
        "tensorboard":     "tensorboard",
        "scipy":           "scipy",
    }
    missing = []
    for import_name, pkg_name in packages.items():
        try:
            __import__(import_name)
            print(f"  OK  {pkg_name}")
        except ImportError:
            print(f"  MISSING  {pkg_name}  <-- pip install {pkg_name}")
            missing.append(pkg_name)

    if missing:
        print(f"\n  ACTION REQUIRED: pip install {' '.join(missing)}")
    else:
        print("\n  All packages installed.")
    print()


def run_test(test_file):
    """Runs a single test file via pytest and returns (passed, failed, output)."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short", "--no-header"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    output = result.stdout + result.stderr

    # Count passed/failed from pytest output
    passed = output.count(" PASSED")
    failed = output.count(" FAILED")
    errors = output.count(" ERROR")

    return passed, failed + errors, output


def main():
    print()
    print(DIVIDER)
    print("  MANUFACTURING MARL — FULL TEST SUITE")
    print(DIVIDER)
    print()

    # 1. Package check
    check_packages()

    # 2. Run each test file
    total_passed = 0
    total_failed = 0
    failures = []

    print(DIVIDER)
    print("TEST RESULTS")
    print(DIVIDER)

    for test_file in TEST_FILES:
        name = test_file.replace("tests/", "").replace(".py", "")

        if not os.path.exists(test_file):
            print(f"  MISSING  {name} (file not found)")
            continue

        passed, failed, output = run_test(test_file)
        total_passed += passed
        total_failed += failed

        status = "PASS" if failed == 0 else "FAIL"
        bar    = "✓" * passed + "✗" * failed
        print(f"  [{status}]  {name:<30}  {passed} passed  {failed} failed  {bar}")

        if failed > 0:
            failures.append((name, output))

    # 3. Print full output for failures
    if failures:
        print()
        print(DIVIDER)
        print("FAILURE DETAILS")
        print(DIVIDER)
        for name, output in failures:
            print(f"\n--- {name} ---")
            # Print only the relevant failure lines
            lines = output.split("\n")
            in_failure = False
            for line in lines:
                if "FAILED" in line or "ERROR" in line or "AssertionError" in line:
                    in_failure = True
                if in_failure:
                    print(line)
                if line.strip() == "" and in_failure:
                    in_failure = False

    # 4. Summary
    print()
    print(DIVIDER)
    print("SUMMARY")
    print(DIVIDER)
    print(f"  Total passed: {total_passed}")
    print(f"  Total failed: {total_failed}")
    print()

    if total_failed == 0:
        print("  ALL TESTS PASSED — ready for training")
    else:
        print("  FAILURES DETECTED — share this output for debugging")

    print(DIVIDER)
    print()


if __name__ == "__main__":
    main()
