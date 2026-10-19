import json
import os
import sys
from pathlib import Path

import pytest

from scripts.check import Check, build_checks, run_checks


def test_full_check_generates_docs_assets_before_strict_build() -> None:
    checks = build_checks(full=True, installed_wheel=False)
    commands = [list(check.command) for check in checks]
    strict = commands.index([sys.executable, "-m", "mkdocs", "build", "--strict"])
    for generator in ("interactive_geometry.py", "spectral_dashboards.py"):
        command = [
            sys.executable,
            f"examples/{generator}",
            "--output",
            "docs/visuals/interactive",
        ]
        assert commands.index(command) < strict
    ordinary = build_checks(full=False, installed_wheel=False)
    assert not any("Generate" in check.label for check in ordinary)


def test_checks_use_checkout_and_isolate_fallback_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "checkout with spaces"
    root.mkdir()
    monkeypatch.setenv("PYTHONPATH", "unrelated-path")
    monkeypatch.setenv("PYFFP_DISABLE_CYTHON", "inherited")
    child = (
        "import json, os; from pathlib import Path; "
        "Path({name!r}).write_text(json.dumps([os.getcwd(), "
        "os.environ.get('PYTHONPATH'), os.environ.get('PYFFP_DISABLE_CYTHON')]))"
    )
    checks = [
        Check(
            "fallback", [sys.executable, "-c", child.format(name="fallback.json")], True
        ),
        Check("ordinary", [sys.executable, "-c", child.format(name="ordinary.json")]),
    ]
    assert run_checks(checks, root) == 0
    assert json.loads((root / "fallback.json").read_text()) == [
        str(root),
        str(root),
        "1",
    ]
    assert json.loads((root / "ordinary.json").read_text()) == [
        str(root),
        str(root),
        None,
    ]
    assert os.environ["PYTHONPATH"] == "unrelated-path"
    assert os.environ["PYFFP_DISABLE_CYTHON"] == "inherited"


def test_failure_stops_checks_and_preserves_exit_status(tmp_path: Path) -> None:
    checks = [
        Check("failing check", [sys.executable, "-c", "raise SystemExit(7)"]),
        Check(
            "must not run",
            [
                sys.executable,
                "-c",
                "from pathlib import Path; Path('unexpected').touch()",
            ],
        ),
    ]
    assert run_checks(checks, tmp_path) == 7
    assert not (tmp_path / "unexpected").exists()


def test_missing_program_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        run_checks(
            [Check("missing tool", [str(tmp_path / "absent-program")])], tmp_path
        )
        == 1
    )
    assert "Could not start missing tool" in capsys.readouterr().err
