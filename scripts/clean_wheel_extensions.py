"""Remove previous wheel builds' extensions from a disposable build checkout."""

from pathlib import Path


def clean_extensions(directory: Path) -> int:
    removed = 0
    for pattern in ("modular_fast*.so", "modular_fast*.pyd"):
        for extension in directory.glob(pattern):
            extension.unlink()
            removed += 1
    return removed


def main() -> None:
    directory = Path(__file__).resolve().parents[1] / "finitefree" / "utils"
    print(f"Removed {clean_extensions(directory)} previous build extensions")


if __name__ == "__main__":
    main()
