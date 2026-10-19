"""Validate static release metadata and select an explicit publishing destination."""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from packaging.version import Version

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def project_metadata(root: Path) -> dict[str, Any]:
    with (root / "pyproject.toml").open("rb") as stream:
        project: dict[str, Any] = tomllib.load(stream)["project"]
    return project


def release_destination(
    version: str, ref: str, event: str, testpypi: bool = True
) -> str:
    parsed = Version(version)
    if (
        str(parsed) != version
        or len(parsed.release) != 3
        or parsed.epoch
        or parsed.local
        or parsed.is_postrelease
    ):
        raise ValueError(
            "Use a canonical X.Y.Z version, optionally with a/b/rc/dev suffix"
        )
    if ref.startswith("refs/tags/"):
        if ref != f"refs/tags/v{version}":
            raise ValueError(
                "Release tag must exactly match the static package version"
            )
    elif event == "push":
        raise ValueError("Publishing pushes must reference a matching version tag")
    if event == "workflow_dispatch":
        if testpypi:
            return "testpypi"
        if parsed.is_prerelease or parsed.is_devrelease:
            raise ValueError("Production publication requires a stable X.Y.Z version")
        return "pypi"
    if event != "push":
        raise ValueError(f"Unsupported publishing event: {event}")
    return "pypi" if re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version) else "testpypi"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parent.parent
    )
    parser.add_argument("--event", choices=("push", "workflow_dispatch"), required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--testpypi", choices=("true", "false"), default="true")
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()
    version = str(project_metadata(args.root)["version"])
    destination = release_destination(
        version, args.ref, args.event, args.testpypi == "true"
    )
    result = {
        "version": version,
        "destination": destination,
        "repository_url": "https://test.pypi.org/legacy/"
        if destination == "testpypi"
        else "https://upload.pypi.org/legacy/",
    }
    if args.github_output:
        with args.github_output.open("a", encoding="utf-8") as output:
            output.write("".join(f"{key}={value}\n" for key, value in result.items()))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
