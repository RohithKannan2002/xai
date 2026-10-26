"""LIME tabular explanation entry point."""
def run():
    """Check LIME installation and provide a usage hint."""
    try:
        import lime
    except ImportError as exc:
        raise ImportError("Install LIME with: pip install lime") from exc
    print("LIME is installed. See help(lime) and the LIME Tabular Explainer API.")
