"""Topic reference for GUIDED BACKPROPAGATION.

Run the full practical with:
    python -m xai.image.full_practical

Print the matching source section with:
    python -c "from xai.image.guided_backprop import show_source; show_source()"
"""
from pathlib import Path

def show_source():
    """Print the corresponding section from the complete practical source."""
    source_file = Path(__file__).with_name("full_practical.py")
    lines = source_file.read_text(encoding="utf-8").splitlines()
    heading = "GUIDED BACKPROPAGATION"
    start = next((i for i, line in enumerate(lines) if heading in line.upper()), None)
    if start is None:
        print(f"Section {heading!r} was not found in {source_file.name}.")
        return
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].startswith("# ") and ("====" in lines[i] or (i + 1 < len(lines) and lines[i + 1].startswith("#"))):
            # Stop at the next numbered section heading.
            if any(ch.isdigit() for ch in lines[i]):
                end = i
                break
    print("\n".join(lines[start:end]))

if __name__ == "__main__":
    show_source()
