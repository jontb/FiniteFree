"""Run the regression suite against the installed package, away from source."""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    with tempfile.TemporaryDirectory(prefix="finitefree-wheel-") as directory:
        shutil.copytree(root / "tests", Path(directory) / "tests")
        # This also fails if the installed wheel omitted the compiled extension.
        probe = (
            "from pathlib import Path; import finitefree; "
            "import finitefree.utils.modular_fast; "
            f"assert not Path(finitefree.__file__).resolve().is_relative_to({str(root)!r}), "
            "'Imported source checkout instead of installed wheel'; "
            "print('Installed package:', finitefree.__file__)"
        )
        subprocess.run(
            [sys.executable, "-c", probe], cwd=directory, env=env, check=True
        )
        return subprocess.call(
            [sys.executable, "-m", "pytest", "--import-mode=importlib", "tests/"],
            cwd=directory,
            env=env,
        )


if __name__ == "__main__":
    sys.exit(main())
