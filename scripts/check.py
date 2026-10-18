"""Run contributor checks with the active interpreter, without shell syntax."""

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ["finitefree", "tests", "scripts", "examples"]


@dataclass(frozen=True)
class Check:
    label: str
    command: Sequence[str]
    fallback: bool = False


def build_checks(full: bool, installed_wheel: bool) -> list[Check]:
    python = sys.executable
    extension = (
        "from setuptools import setup, Extension; "
        "from Cython.Build import cythonize; import numpy as np; "
        "setup(ext_modules=cythonize(Extension("
        "'finitefree.utils.modular_fast', ['finitefree/utils/modular_fast.pyx'], "
        "include_dirs=[np.get_include()])), "
        "script_args=['build_ext', '--inplace'])"
    )
    tests = [
        python,
        "-m",
        "pytest",
        "--import-mode=importlib",
        "tests/",
        "scripts/tests/",
    ]
    checks = [
        Check("Ruff lint", [python, "-m", "ruff", "check", *SOURCES]),
        Check("Ruff formatting", [python, "-m", "ruff", "format", "--check", *SOURCES]),
        Check("Strict typing", [python, "-m", "mypy", *SOURCES]),
        Check("Build source extension", [python, "-c", extension]),
        Check("Source and tooling tests", tests),
    ]
    if full:
        synchronization = (
            "from pathlib import Path; from scripts.sync_docs import convert_readme; "
            "assert convert_readme(Path('README.md').read_text(encoding='utf-8')) "
            "== Path('docs/index.md').read_text(encoding='utf-8'), "
            "'Run python scripts/sync_docs.py and commit docs/index.md'"
        )
        checks.extend(
            [
                Check("Fallback tests", tests, fallback=True),
                Check("README/index synchronization", [python, "-c", synchronization]),
                Check(
                    "Documentation examples",
                    [python, "scripts/check_readme_examples.py"],
                ),
                Check("Showcase", [python, "examples/showcase.py"]),
                Check(
                    "Strict documentation build",
                    [python, "-m", "mkdocs", "build", "--strict"],
                ),
            ]
        )
    if installed_wheel:
        checks.append(
            Check("Installed-wheel tests", [python, "scripts/test_installed_wheel.py"])
        )
    return checks


def run_checks(checks: Sequence[Check], root: Path) -> int:
    for check in checks:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root)
        if check.fallback:
            env["PYFFP_DISABLE_CYTHON"] = "1"
        else:
            env.pop("PYFFP_DISABLE_CYTHON", None)
        print(f"\n== {check.label} ==", flush=True)
        try:
            result = subprocess.run(check.command, cwd=root, env=env, check=False)
        except OSError as error:
            print(f"Could not start {check.label}: {error}", file=sys.stderr)
            return 1
        if result.returncode:
            print(f"Failed: {check.label} (exit {result.returncode})", file=sys.stderr)
            return result.returncode
    print("\nAll requested checks passed.", flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Also run fallback tests, examples and strict docs",
    )
    parser.add_argument(
        "--installed-wheel",
        action="store_true",
        help="Also test the standard-installed candidate outside this checkout",
    )
    args = parser.parse_args(argv)
    return run_checks(build_checks(args.full, args.installed_wheel), ROOT)


if __name__ == "__main__":
    sys.exit(main())
