"""Validate release archives against the checkout and preserve their SHA-256 hashes."""

import argparse
import email
import hashlib
import json
import tarfile
import zipfile
from pathlib import Path
from typing import Any

from packaging.requirements import Requirement
from packaging.utils import (
    canonicalize_name,
    parse_sdist_filename,
    parse_wheel_filename,
)

from scripts.check_wheel_contents import check_wheels
from scripts.release_preflight import project_metadata


def runtime_requirements(values: list[str]) -> set[str]:
    return {
        str(Requirement(value))
        for value in values
        if Requirement(value).marker is None
        or "extra" not in str(Requirement(value).marker)
    }


def check_metadata(data: bytes, project: dict[str, Any]) -> None:
    metadata = email.message_from_bytes(data)
    if (
        canonicalize_name(str(metadata["Name"])) != canonicalize_name(project["name"])
        or metadata["Version"] != project["version"]
        or metadata["Requires-Python"] != project["requires-python"]
        or runtime_requirements(metadata.get_all("Requires-Dist", []))
        != runtime_requirements(project["dependencies"])
    ):
        raise ValueError(
            "Archive metadata differs from the source version or dependency policy"
        )


def validate_archives(
    directory: Path,
    root: Path,
    expected_wheels: int | None = None,
    require_sdist: bool = False,
) -> list[dict[str, Any]]:
    project = project_metadata(root)
    sources = {
        path.relative_to(root).as_posix(): path.read_bytes().replace(b"\r\n", b"\n")
        for path in (root / "finitefree").rglob("*")
        if path.suffix in (".py", ".pyx")
    }
    archives = sorted([*directory.rglob("*.whl"), *directory.rglob("*.tar.gz")])
    if not archives or len({path.name for path in archives}) != len(archives):
        raise ValueError("Require nonempty, uniquely named distribution archives")
    wheels = [path for path in archives if path.suffix == ".whl"]
    sdists = [path for path in archives if path.name.endswith(".tar.gz")]
    if expected_wheels is not None and len(wheels) != expected_wheels:
        raise ValueError(f"Expected {expected_wheels} wheels, found {len(wheels)}")
    if len(sdists) > 1 or (require_sdist and len(sdists) != 1):
        raise ValueError("Require exactly one source archive")
    for parent in {path.parent for path in wheels}:
        check_wheels(parent)
    manifest = []
    for path in archives:
        if path.suffix == ".whl":
            name, version, _, _ = parse_wheel_filename(path.name)
            with zipfile.ZipFile(path) as wheel:
                entries = wheel.namelist()
                if len(entries) != len(set(entries)):
                    raise ValueError(f"Duplicate wheel entries: {path.name}")
                metadata = [
                    entry for entry in entries if entry.endswith(".dist-info/METADATA")
                ]
                if len(metadata) != 1:
                    raise ValueError(f"Expected one metadata file: {path.name}")
                check_metadata(wheel.read(metadata[0]), project)
                contents = {
                    entry: wheel.read(entry).replace(b"\r\n", b"\n")
                    for entry in entries
                    if entry.startswith("finitefree/")
                    and Path(entry).suffix in (".py", ".pyx")
                }
        else:
            name, version = parse_sdist_filename(path.name)
            with tarfile.open(path) as sdist:
                entries = sdist.getnames()
                if len(entries) != len(set(entries)):
                    raise ValueError(f"Duplicate source entries: {path.name}")
                prefix = entries[0].split("/")[0] + "/"
                if any(
                    not entry.startswith(prefix) and entry != prefix[:-1]
                    for entry in entries
                ):
                    raise ValueError("Source archive must have one top-level directory")
                if any(
                    entry.endswith((".whl", ".so", ".pyd", ".pyc"))
                    or "/wheelhouse/" in entry
                    for entry in entries
                ):
                    raise ValueError("Source archive contains stale build artifacts")

                def read(entry: str) -> bytes:
                    stream = sdist.extractfile(entry)
                    if stream is None:
                        raise ValueError(f"Expected a regular source file: {entry}")
                    return stream.read()

                check_metadata(read(prefix + "PKG-INFO"), project)
                if read(prefix + "pyproject.toml").replace(b"\r\n", b"\n") != (
                    root / "pyproject.toml"
                ).read_bytes().replace(b"\r\n", b"\n"):
                    raise ValueError(
                        "Source archive build policy differs from the checkout"
                    )
                contents = {
                    entry[len(prefix) :]: read(entry).replace(b"\r\n", b"\n")
                    for entry in entries
                    if entry.startswith(prefix + "finitefree/")
                    and Path(entry).suffix in (".py", ".pyx")
                }
        if (
            canonicalize_name(name) != canonicalize_name(project["name"])
            or str(version) != project["version"]
        ):
            raise ValueError(
                f"Distribution filename version/name mismatch: {path.name}"
            )
        if contents != sources:
            raise ValueError(
                f"Runtime source inventory/content differs from checkout: {path.name}"
            )
        manifest.append(
            {
                "filename": path.name,
                "size": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return manifest


def verify_manifest(directory: Path, manifest: list[dict[str, Any]]) -> None:
    archives = [*directory.rglob("*.whl"), *directory.rglob("*.tar.gz")]
    actual = {path.name: path for path in archives}
    expected = {entry["filename"]: entry for entry in manifest}
    if (
        len(actual) != len(archives)
        or len(expected) != len(manifest)
        or actual.keys() != expected.keys()
    ):
        raise ValueError("Archive inventory differs from the reviewed manifest")
    for name, path in actual.items():
        if (
            path.stat().st_size != expected[name]["size"]
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected[name]["sha256"]
        ):
            raise ValueError(f"Archive bytes differ from reviewed manifest: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parent.parent
    )
    parser.add_argument("--expected-wheels", type=int)
    parser.add_argument("--require-sdist", action="store_true")
    parser.add_argument(
        "--verify",
        type=Path,
        help="Compare downloaded archives to a previously reviewed manifest",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Write the validated inventory and SHA256SUMS next to this path",
    )
    args = parser.parse_args()
    if args.verify:
        verify_manifest(
            args.directory, json.loads(args.verify.read_text(encoding="utf-8"))
        )
    manifest = validate_archives(
        args.directory, args.root, args.expected_wheels, args.require_sdist
    )
    if args.manifest:
        args.manifest.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        args.manifest.with_name("SHA256SUMS").write_text(
            "".join(f"{entry['sha256']}  {entry['filename']}\n" for entry in manifest),
            encoding="utf-8",
        )
    print(f"Verified {len(manifest)} release archives against the checkout")


if __name__ == "__main__":
    main()
