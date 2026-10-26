"""Print the XAI package tree, then display the full source of every Python file."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT / "xai"

def print_tree(folder, prefix=""):
    entries = sorted(folder.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    for i, entry in enumerate(entries):
        last = i == len(entries) - 1
        print(prefix + ("└── " if last else "├── ") + entry.name + ("/" if entry.is_dir() else ""))
        if entry.is_dir():
            print_tree(entry, prefix + ("    " if last else "│   "))

def show_source(path):
    print("\n" + "=" * 78)
    print(f"SOURCE: {path.relative_to(ROOT)}")
    print("=" * 78)
    try:
        print(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as exc:
        print(f"Could not read source: {exc}")

if __name__ == "__main__":
    print("XAI PRACTICAL REFERENCE — FILE STRUCTURE")
    print("=" * 42)
    print("xai/")
    print_tree(PACKAGE, "    ")
    print("\nRoot helper files:")
    for name in ("help_all.py", "README.md", "requirements-tabular.txt", "requirements-image.txt", "requirements-text.txt"):
        print(f"    {name}")
    for source in sorted(PACKAGE.rglob("*.py")):
        show_source(source)
