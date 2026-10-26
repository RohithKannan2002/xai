"""Tree-interpreter optional dependency entry point."""
def run():
    """Check whether treeinterpreter is installed."""
    try:
        import treeinterpreter
    except ImportError as exc:
        raise ImportError("Install with: pip install treeinterpreter") from exc
    print("treeinterpreter is installed.")
