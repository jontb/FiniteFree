#!/usr/bin/env python3
"""
Syncs README.md to docs/index.md, converting HTML details tags to MkDocs-Material
??? details blocks to ensure perfect Markdown list and table rendering.
"""

import re
from pathlib import Path


def is_list_item(line):
    stripped = line.strip()
    return bool(re.match(r"^[-*]\s+", stripped) or re.match(r"^\d+\.\s+", stripped))


def sync():
    root_dir = Path(__file__).parent.parent.resolve()
    readme_path = root_dir / "README.md"
    index_path = root_dir / "docs" / "index.md"

    if not readme_path.exists():
        print(f"Error: README.md not found at {readme_path}")
        return

    print(f"Reading {readme_path}...")
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Convert details blocks
    lines = content.splitlines()
    new_lines = []

    in_details = False
    current_details_content = []
    title = ""

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("<details"):
            in_details = True
            # Read next line to find summary/title
            i += 1
            summary_line = lines[i]
            # Extract title from summary line
            # e.g., <summary><b>1. Limiting Distributions of Free Convolutions</b></summary>
            m = re.search(r"<summary>(.*?)</summary>", summary_line)
            if m:
                raw_title = m.group(1)
                # Clean title of <b> / </b> tags and backticks / formatting
                title = re.sub(r"<[^>]+>", "", raw_title).strip()
            else:
                title = "Details"

            current_details_content = []
            i += 1

            # Skip any leading empty lines or <br> tags immediately following the summary
            while i < len(lines) and (
                lines[i].strip() == ""
                or re.match(r"^<br\s*/?>$", lines[i].strip(), re.IGNORECASE)
            ):
                i += 1
            continue

        elif in_details and stripped == "</details>":
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

    print(f"Writing to {index_path}...")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines) + "\n")

    print("Sync complete.")


if __name__ == "__main__":
    sync()
