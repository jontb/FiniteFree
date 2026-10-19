"""Check a fixed set of representative wrong implementations against real contracts.

Each baseline and injected fault runs in a fresh process. No source file is
modified. Passing requires a clean baseline and a genuine assertion failure in
the selected contract, rather than a collection/import error or timeout.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAULTS = {
    "cumulant_normalization": (
        "tests/test_cumulant_prefix_contracts.py::test_cumulants_against_independent_definitions_and_closed_forms",
        """
import math
import finitefree
import finitefree.transforms as transforms
original = finitefree.FiniteRTransform
def wrong(*args, **kwargs):
    return [value * math.factorial(index)**2 for index, value in enumerate(original(*args, **kwargs))]
finitefree.FiniteRTransform = transforms.FiniteRTransform = wrong
""",
    ),
    "crt_bound": (
        "tests/test_diagonal_specialization_contracts.py::test_modular_agreement_cannot_hide_specialization_coefficients",
        """
import finitefree.hyperbolic.pencils as pencils
pencils._determinant_coefficient_bound = lambda matrices: 0
""",
    ),
    "stale_hvp_direction": (
        "tests/test_adversarial_pencils.py::test_prepared_derivatives_against_symbolic_determinants_across_scales",
        """
from finitefree.hyperbolic.prepared import PencilFactorization
original = PencilFactorization.logabsdet_hvp
def wrong(self, directions):
    if not hasattr(self, '_audit_hvp'):
        self._audit_hvp = original(self, directions)
    return self._audit_hvp.copy()
PencilFactorization.logabsdet_hvp = wrong
""",
    ),
    "deterministic_haar": (
        "tests/test_ensembles.py::test_empirical_conjugation_coefficient_laws",
        """
import numpy as np
import finitefree
finitefree.sample_haar_unitary = lambda dimension: np.eye(dimension, dtype=complex)
""",
    ),
}


def audit(both_backends: bool) -> list[dict[str, str]]:
    results = []
    for fallback in (False, True) if both_backends else (False,):
        backend = "fallback" if fallback else "compiled"
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("COV_")
            and key
            not in ("PYTEST_CURRENT_TEST", "PYTEST_PLUGINS", "PYFFP_DISABLE_CYTHON")
        }
        env["PYTHONPATH"] = str(ROOT)
        if fallback:
            env["PYFFP_DISABLE_CYTHON"] = "1"
        for name, (target, mutation) in FAULTS.items():
            for injected in (False, True):
                program = (
                    (mutation if injected else "")
                    + "\nimport pytest\nraise SystemExit(pytest.main(['-q', '--import-mode=importlib', "
                    + repr(target)
                    + "]))\n"
                )
                completed = subprocess.run(
                    [sys.executable, "-c", program],
                    cwd=ROOT,
                    env=env,
                    text=True,
                    capture_output=True,
                    timeout=180,
                )
                output = completed.stdout + completed.stderr
                success = completed.returncode == (1 if injected else 0)
                if injected:
                    success = (
                        success
                        and "AssertionError" in output
                        and "FAILED " + target in output
                    )
                if not success:
                    raise RuntimeError(
                        f"{backend}/{name}: {'fault survived or failed outside assertions' if injected else 'baseline failed'}\n{output}"
                    )
            results.append({"backend": backend, "fault": name, "result": "detected"})
            print(f"{backend}: detected {name}", flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--both-backends", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = audit(args.both_backends)
    if args.output:
        args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(
        f"All {len(results)} representative injected faults detected after clean controls"
    )


if __name__ == "__main__":
    main()
