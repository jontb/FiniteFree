"""Execute each Python usage example independently to catch API drift."""

import re
import textwrap
from pathlib import Path
from typing import Any


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    paths = [
        root / "README.md",
        root / "CHANGELOG.md",
        *sorted((root / "docs").glob("*.md")),
    ]
    for path in paths:
        relative = path.relative_to(root)
        content = path.read_text(encoding="utf-8")
        blocks = re.findall(
            r"^[ \t]*```python[ \t]*\n(.*?)^[ \t]*```[ \t]*$",
            content,
            flags=re.MULTILINE | re.DOTALL,
        )
        for index, block in enumerate(blocks, start=1):
            namespace: dict[str, Any] = {"__name__": "__main__"}
            exec(
                compile(
                    textwrap.dedent(block), f"{relative} Python example {index}", "exec"
                ),
                namespace,
            )
        print(f"Passed {len(blocks)} independent Python examples in {relative}")


if __name__ == "__main__":
    main()
