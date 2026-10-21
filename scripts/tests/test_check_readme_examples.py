from pathlib import Path

import pytest

from scripts import check_readme_examples


def prepare_documents(
    root: Path, monkeypatch: pytest.MonkeyPatch, package_examples: str
) -> None:
    """Run the real checker against a miniature documentation tree."""
    (root / "scripts").mkdir()
    (root / "docs").mkdir()
    for name in ["README.md", "CHANGELOG.md"]:
        (root / name).write_text("# Documentation\n", encoding="utf-8")
    (root / "PYPI_README.md").write_text(package_examples, encoding="utf-8")
    monkeypatch.setattr(
        check_readme_examples,
        "__file__",
        str(root / "scripts/check_readme_examples.py"),
    )


def test_package_description_example_is_executed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    prepare_documents(
        tmp_path,
        monkeypatch,
        '```python\nassert __name__ == "__main__"\nprint("package example ran")\n```\n',
    )

    check_readme_examples.main()

    output = capsys.readouterr().out
    assert "package example ran" in output
    assert "Passed 1 independent Python examples in PYPI_README.md" in output


def test_invalid_package_description_example_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare_documents(
        tmp_path,
        monkeypatch,
        '```python\nraise RuntimeError("broken package example")\n```\n',
    )

    with pytest.raises(RuntimeError, match="broken package example"):
        check_readme_examples.main()


def test_package_description_fences_do_not_share_a_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    prepare_documents(
        tmp_path,
        monkeypatch,
        "```python\nprevious_example_value = 42\n```\n\n"
        '```python\nassert "previous_example_value" not in globals()\n```\n',
    )

    check_readme_examples.main()

    assert (
        "Passed 2 independent Python examples in PYPI_README.md"
        in capsys.readouterr().out
    )
