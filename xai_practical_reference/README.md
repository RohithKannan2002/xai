# XAI Practical Reference Library

Organized reference scripts from the uploaded notebook, grouped into tabular, image, and text explainability.

## Structure

```text
xai_practical_reference/
├── xai/
│   ├── tabular/  # PDP, ICE, ALE, PFI, SHAP, TreeInterpreter
│   ├── image/    # Integrated Gradients, XRAI, Grad-CAM, LIME, Guided Backprop, Guided Grad-CAM
│   └── text/     # Tokenization, embeddings, LIME, Gradient x Input, Layer Integrated Gradients
├── help_all.py
└── requirements-*.txt
```

Run `python help_all.py` to print the tree and full Python source. Run the complete practicals from this directory with `python -m xai.tabular.full_practical`, `python -m xai.image.full_practical`, or `python -m xai.text.full_practical`.

Topic-specific files are reference excerpts; full practical scripts contain shared imports and setup. Install the requirements for the topic you run. The image script expects `/content/cat.jpg`; change it to a local image path. The tabular script downloads California Housing data, and the text script downloads a pretrained DistilBERT model on first run.
