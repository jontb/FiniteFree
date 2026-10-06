"""Reject wheels containing stale compiled extensions from another Python build."""

import sys
import zipfile
from pathlib import Path


def check_wheels(directory: Path) -> int:
    wheels = sorted(directory.glob("*.whl"))
    if not wheels:
        raise ValueError(f"No wheels found in {directory}")
    for wheel in wheels:
        python_tag = wheel.name.split("-")[2]
        with zipfile.ZipFile(wheel) as archive:
            extensions = [
                name
                for name in archive.namelist()
                if name.startswith("finitefree/utils/modular_fast.")
                and Path(name).suffix in (".so", ".pyd")
            ]
        if len(extensions) != 1:
            raise ValueError(f"{wheel.name}: expected one extension, got {extensions}")
        version = python_tag[2:]
        extension = extensions[0]
        if not python_tag.startswith("cp") or not (
            f".cpython-{version}-" in extension or f".cp{version}-" in extension
        ):
            raise ValueError(f"{wheel.name}: extension does not match {python_tag}")
    return len(wheels)


def main() -> None:
    print(f"Verified {check_wheels(Path(sys.argv[1]))} wheel extension inventories")


if __name__ == "__main__":
    main()
