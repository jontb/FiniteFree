#!/usr/bin/env python3
"""
Syncs README.md to docs/index.md, converting HTML details tags to MkDocs-Material
??? details blocks. Fenced examples are preserved verbatim; malformed details
blocks fail before the destination is written.
"""

import re
from pathlib import Path


def is_list_item(line: str) -> bool:
    stripped = line.strip()
    return bool(re.match(r"^[-*]\s+", stripped) or re.match(r"^\d+\.\s+", stripped))


def convert_readme(content: str) -> str:
    """Convert single-level, separate-line details blocks for the docs index.

    Summaries must occupy their own line; blank lines before them are allowed.
    This handles the README's format, rather than arbitrary HTML or Markdown.
    """
    lines = content.splitlines()
    new_lines: list[str] = []

    in_details = False
    current_details_content: list[str] = []
    title = ""
    details_line = 0
    fence: str | None = None

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        destination = current_details_content if in_details else new_lines

        # Inside a fence, HTML tags, list-looking lines and links are literals.
        if fence is not None:
            destination.append(line)
            closing = re.fullmatch(r"\s*(`+|~+)\s*", line)
            if closing and closing[1][0] == fence[0] and len(closing[1]) >= len(fence):
                fence = None
            i += 1
            continue
        opening = re.match(r"\s*(`{3,}|~{3,})", line)
        if opening:
            fence = opening[1]
            destination.append(line)
            i += 1
            continue

        if re.match(r"<details(?:\s|>)", stripped):
            if in_details:
                raise ValueError(f"Nested <details> block at line {i + 1}")
            if not re.fullmatch(r"<details(?:\s+[^>]*)?>", stripped):
                raise ValueError(f"<details> must occupy its own line at {i + 1}")
            in_details = True
            details_line = i + 1
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            # Extract title from summary line
            # e.g., <summary><b>1. Limiting Distributions of Free Convolutions</b></summary>
            summary = (
                re.fullmatch(r"<summary>(.*?)</summary>", lines[i].strip())
                if i < len(lines)
                else None
            )
            if summary is None:
                raise ValueError(f"Missing <summary> after line {details_line}")
            title = re.sub(r"<[^>]+>", "", summary[1]).strip()

            current_details_content = []
            i += 1

            # Skip any leading empty lines or <br> tags immediately following the summary
            while i < len(lines) and (
                lines[i].strip() == ""
                or re.match(r"^<br\s*/?>$", lines[i].strip(), re.IGNORECASE)
            ):
                i += 1
            continue

        elif stripped == "</details>":
            if not in_details:
                raise ValueError(f"Unexpected </details> at line {i + 1}")
            in_details = False
            # Write out the converted block
            new_lines.append(f'??? "{title}"')
            new_lines.append("")

            # Remove trailing empty lines or <br> tags from the end of the details block
            while current_details_content and (
                current_details_content[-1].strip() == ""
                or re.match(
                    r"^<br\s*/?>$", current_details_content[-1].strip(), re.IGNORECASE
                )
            ):
                current_details_content.pop()

            for d_line in current_details_content:
                if d_line.strip() == "":
                    new_lines.append("")
                else:
                    new_lines.append("    " + d_line)
            new_lines.append("")
            i += 1
            continue

        # README documentation links are relative to the repository root.
        line = line.replace("](docs/", "](")
        if in_details:
            if is_list_item(line) and current_details_content:
                prev = current_details_content[-1]
                if prev.strip() != "" and not is_list_item(prev):
                    current_details_content.append("")
            current_details_content.append(line)
        else:
            if is_list_item(line) and new_lines:
                prev = new_lines[-1]
                if prev.strip() != "" and not is_list_item(prev):
                    new_lines.append("")
            new_lines.append(line)

        i += 1

    if in_details:
        raise ValueError(f"Unclosed <details> block at line {details_line}")
    return "\n".join(new_lines) + "\n"


def sync(root_dir: Path | None = None) -> None:
    if root_dir is None:
        root_dir = Path(__file__).parent.parent.resolve()
    readme_path = root_dir / "README.md"
    index_path = root_dir / "docs" / "index.md"
    print(f"Reading {readme_path}...")
    # Read and validate the entire input before touching an existing index.
    converted = convert_readme(readme_path.read_text(encoding="utf-8"))
    print(f"Writing to {index_path}...")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(converted, encoding="utf-8")

    print("Sync complete.")


if __name__ == "__main__":
    sync()
