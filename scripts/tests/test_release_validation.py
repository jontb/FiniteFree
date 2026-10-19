"""Release routing and archive integrity must reject plausible wrong releases."""

import io
import json
import tarfile
import zipfile
from pathlib import Path

import pytest

from scripts.check_release_artifacts import validate_archives, verify_manifest
from scripts.release_preflight import release_destination


def test_release_destination_preserves_manual_default_and_recognizes_prereleases() -> (
    None
):
    assert release_destination("0.4.0", "refs/tags/v0.4.0", "push") == "pypi"
    assert (
        release_destination("0.4.0", "refs/heads/develop", "workflow_dispatch")
        == "testpypi"
    )
    assert (
        release_destination("0.4.0", "refs/heads/develop", "workflow_dispatch", False)
        == "pypi"
    )
    for version in ("0.4.0a1", "0.4.0b2", "0.4.0rc1", "0.4.0.dev2", "0.4.0rc1.dev2"):
        assert (
            release_destination(version, f"refs/tags/v{version}", "push", False)
            == "testpypi"
        )
        with pytest.raises(ValueError, match="stable"):
            release_destination(
                version, "refs/heads/develop", "workflow_dispatch", False
            )
    for version, ref, event in (
        ("0.4.0", "refs/tags/v0.3.0", "push"),
        ("0.4.0", "refs/tags/v0.3.0", "workflow_dispatch"),
        ("0.4.0", "refs/heads/develop", "push"),
        ("0.4.0+local", "refs/tags/v0.4.0+local", "push"),
        ("0.4.0.post1", "refs/tags/v0.4.0.post1", "push"),
        ("0.4", "refs/tags/v0.4", "push"),
        ("1!0.4.0", "refs/tags/v1!0.4.0", "push"),
        ("0.4.0", "refs/tags/v0.4.0", "pull_request"),
    ):
        with pytest.raises(ValueError):
            release_destination(version, ref, event)


def distribution_fixture(root: Path, defect: str = "") -> Path:
    package = root / "finitefree"
    package.mkdir(parents=True)
    source = b"VERSION = '0.4.0'\n"
    (package / "__init__.py").write_bytes(source)
    (root / "pyproject.toml").write_text(
        '[project]\nname = "finitefree"\nversion = "0.4.0"\nrequires-python = ">=3.10"\ndependencies = ["numpy>=1.24"]\n',
        encoding="utf-8",
    )
    metadata = "Metadata-Version: 2.1\nName: finitefree\nVersion: 0.4.0\nRequires-Python: >=3.10\nRequires-Dist: numpy>=1.24\nRequires-Dist: pytest; extra == 'dev'\n\nDescription\n"
    if defect == "version":
        metadata = metadata.replace("Version: 0.4.0", "Version: 0.3.0")
    if defect == "python":
        metadata = metadata.replace(">=3.10", ">=3.11")
    if defect == "dependency":
        metadata = metadata.replace("numpy>=1.24", "numpy>=2")
    entries = {
        "finitefree/__init__.py": source
        if defect != "source"
        else b"VERSION = 'wrong'\n",
        "finitefree/utils/modular_fast.cpython-310-x86_64-linux-gnu.so": b"native-inventory-fixture",
        "finitefree-0.4.0.dist-info/METADATA": metadata.encode(),
    }
    if defect == "extension":
        entries["finitefree/utils/modular_fast.cpython-311-x86_64-linux-gnu.so"] = (
            b"stale"
        )
    if defect == "missing_source":
        del entries["finitefree/__init__.py"]
    directory = root / "dist"
    directory.mkdir()
    with zipfile.ZipFile(
        directory / "finitefree-0.4.0-cp310-cp310-manylinux_2_17_x86_64.whl", "w"
    ) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    with tarfile.open(directory / "finitefree-0.4.0.tar.gz", "w:gz") as archive:
        members = {
            "finitefree-0.4.0/PKG-INFO": metadata.encode(),
            "finitefree-0.4.0/pyproject.toml": (root / "pyproject.toml").read_bytes()
            if defect != "build_policy"
            else b'[project]\nversion = "wrong"\n',
            "finitefree-0.4.0/finitefree/__init__.py": source,
        }
        if defect == "stale_sdist":
            members["finitefree-0.4.0/wheelhouse/stale.whl"] = b"stale"
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return directory


def test_archive_policy_rejects_wrong_metadata_source_and_extensions(
    tmp_path: Path,
) -> None:
    for defect in (
        "",
        "version",
        "python",
        "dependency",
        "source",
        "missing_source",
        "extension",
        "stale_sdist",
        "build_policy",
    ):
        root = tmp_path / (defect or "valid")
        directory = distribution_fixture(root, defect)
        if defect:
            with pytest.raises(ValueError):
                validate_archives(
                    directory, root, expected_wheels=1, require_sdist=True
                )
        else:
            manifest = validate_archives(
                directory, root, expected_wheels=1, require_sdist=True
            )
            assert len(manifest) == 2
            verify_manifest(directory, manifest)
            with pytest.raises(ValueError, match="Expected 16"):
                validate_archives(
                    directory, root, expected_wheels=16, require_sdist=True
                )


def test_manifest_detects_changed_bytes_missing_extra_and_duplicate_files(
    tmp_path: Path,
) -> None:
    directory = distribution_fixture(tmp_path / "source")
    manifest = validate_archives(
        directory, directory.parent, expected_wheels=1, require_sdist=True
    )
    # Persist/reload the reviewed manifest just as a later publishing/download step does.
    reviewed = json.loads(json.dumps(manifest))
    wheel = next(directory.glob("*.whl"))
    original = wheel.read_bytes()
    wheel.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with pytest.raises(ValueError, match="bytes differ"):
        verify_manifest(directory, reviewed)
    wheel.write_bytes(original)
    for bad in (reviewed[:-1], reviewed + [reviewed[0]]):
        with pytest.raises(ValueError, match="inventory"):
            verify_manifest(directory, bad)
    duplicate = directory / "nested" / wheel.name
    duplicate.parent.mkdir()
    duplicate.write_bytes(original)
    with pytest.raises(ValueError, match="inventory"):
        verify_manifest(directory, reviewed)
    duplicate.unlink()
    wheel.unlink()
    with pytest.raises(ValueError, match="inventory"):
        verify_manifest(directory, reviewed)
