import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.sync_docs import convert_readme, sync


@pytest.mark.parametrize("fence", ["```", "~~~~"])
def test_fenced_examples_preserve_html_lists_and_documentation_links(
    fence: str,
) -> None:
    literal = (
        f"{fence}text\n"
        "<details>\n<summary>Literal example</summary>\n"
        "Intro\n- item\n[API](docs/api.md)\n</details>\n"
        f"{fence}\n"
    )
    assert convert_readme(literal) == literal


@pytest.mark.parametrize("closing", ["~~~", "```", "````python"])
def test_short_wrong_or_annotated_fences_do_not_close_a_code_example(
    closing: str,
) -> None:
    literal = f"````text\n{closing}\n<details>\n[API](docs/api.md)\n`````\n"
    assert convert_readme(literal) == literal


def test_code_inside_details_retains_literal_content_and_list_spacing() -> None:
    source = (
        "<details>\n<summary><b>Example</b></summary>\n\n"
        "```text\nIntro\n- literal item\n</details>\n[API](docs/api.md)\n```\n"
        "</details>\n"
    )
    expected = (
        '??? "Example"\n\n'
        "    ```text\n    Intro\n    - literal item\n    </details>\n"
        "    [API](docs/api.md)\n    ```\n\n"
    )
    assert convert_readme(source) == expected


def test_details_lists_tables_links_and_blank_summary_lines() -> None:
    source = (
        "# Title\n[API](docs/api.md#roots)\n"
        "<details open>\n\n<summary><b>More</b></summary>\n<br>\n\n"
        "Paragraph\n- item\n- next\n\n| a | b |\n| - | - |\n| 1 | 2 |\n"
        "\n<br />\n</details>\n"
    )
    expected = (
        '# Title\n[API](api.md#roots)\n??? "More"\n\n'
        "    Paragraph\n\n    - item\n    - next\n\n"
        "    | a | b |\n    | - | - |\n    | 1 | 2 |\n\n"
    )
    assert convert_readme(source) == expected


@pytest.mark.parametrize(
    "source, error",
    [
        ("<details>\n", "Missing <summary> after line 1"),
        ("<details>\n\n", "Missing <summary> after line 1"),
        ("<details>\nBody\n</details>\n", "Missing <summary> after line 1"),
        (
            "<details>\n<summary>Title</summary>\nBody\n",
            "Unclosed <details> block at line 1",
        ),
        (
            "<details>\n<summary>Title</summary>\n<details>\n",
            "Nested <details> block at line 3",
        ),
        ("</details>\n", "Unexpected </details> at line 1"),
        (
            "<details><summary>Title</summary>\n",
            "<details> must occupy its own line at 1",
        ),
    ],
)
def test_malformed_details_fail_with_useful_locations(source: str, error: str) -> None:
    with pytest.raises(ValueError) as failure:
        convert_readme(source)
    assert str(failure.value) == error


@pytest.mark.parametrize(
    "source", ["<details>\n", "<details>\n<summary>Title</summary>\nBody\n"]
)
def test_invalid_readme_cannot_overwrite_an_existing_index(
    tmp_path: Path, source: str
) -> None:
    (tmp_path / "README.md").write_text(source, encoding="utf-8")
    index = tmp_path / "docs" / "index.md"
    index.parent.mkdir()
    index.write_bytes(b"existing index\n")
    with pytest.raises(ValueError):
        sync(tmp_path)
    assert index.read_bytes() == b"existing index\n"


def test_missing_readme_is_an_error_and_preserves_the_index(tmp_path: Path) -> None:
    index = tmp_path / "docs" / "index.md"
    index.parent.mkdir()
    index.write_bytes(b"existing index\n")
    with pytest.raises(FileNotFoundError):
        sync(tmp_path)
    assert index.read_bytes() == b"existing index\n"


def test_sync_creates_parent_directories_and_writes_utf8(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Résumé\n", encoding="utf-8")
    sync(tmp_path)
    assert (tmp_path / "docs" / "index.md").read_text(encoding="utf-8") == "# Résumé\n"


@pytest.mark.parametrize("readme", [None, "<details>\n"])
def test_cli_returns_failure_without_overwriting_the_index(
    tmp_path: Path, readme: str | None
) -> None:
    root = Path(__file__).resolve().parents[2]
    script = tmp_path / "scripts" / "sync_docs.py"
    script.parent.mkdir()
    shutil.copyfile(root / "scripts" / "sync_docs.py", script)
    if readme is not None:
        (tmp_path / "README.md").write_text(readme, encoding="utf-8")
    index = tmp_path / "docs" / "index.md"
    index.parent.mkdir()
    index.write_bytes(b"existing index\n")
    result = subprocess.run([sys.executable, str(script)], capture_output=True)
    assert result.returncode != 0
    assert index.read_bytes() == b"existing index\n"


def test_current_readme_reproduces_the_committed_index() -> None:
    root = Path(__file__).resolve().parents[2]
    readme = (root / "README.md").read_text(encoding="utf-8")
    expected = (root / "docs" / "index.md").read_text(encoding="utf-8")
    assert convert_readme(readme) == expected
