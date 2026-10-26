"""Integrated Gradients reference entry point for image models."""
def run():
    """Check Captum installation and show the relevant API."""
    try:
        import captum
    except ImportError as exc:
        raise ImportError("Install Captum with: pip install captum") from exc
    print("Captum is installed. See captum.attr.IntegratedGradients.")
