"""Execute each Python usage example independently to catch API drift."""

import re
from pathlib import Path
from typing import Any


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    for relative in ("README.md", "docs/api.md", "CHANGELOG.md"):
        content = (root / relative).read_text(encoding="utf-8")
        blocks = re.findall(
            r"^```python\n(.*?)^```", content, flags=re.MULTILINE | re.DOTALL
        )
        if not blocks:
            raise ValueError(f"{relative} contains no Python examples")
        for index, block in enumerate(blocks, start=1):
            namespace: dict[str, Any] = {"__name__": "__main__"}
            exec(
                compile(block, f"{relative} Python example {index}", "exec"), namespace
            )
        print(f"Passed {len(blocks)} independent Python examples in {relative}")


if __name__ == "__main__":
    main()
