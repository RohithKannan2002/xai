"""SHAP explanation entry point."""
def run():
    """Check SHAP installation and provide a usage hint."""
    try:
        import shap
    except ImportError as exc:
        raise ImportError("Install SHAP with: pip install shap") from exc
    print("SHAP version:", shap.__version__)
    print("General starting point: shap.Explainer(model, background_data)")
