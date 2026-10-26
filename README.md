# XAI Learning Kit

A small importable package with tabular and image Explainable AI learning examples.

## Install
```bash
python -m pip install -e .
```

## Usage
```python
import xai
from xai.tabular.regression import linear, multiple, polynomial
from xai.tabular.classification import iris

help(xai)
help(linear)
help(linear.run)

linear.run()
# multiple.run()
# polynomial.run(degree=3)
# iris.run()
```

## Explainability modules
```python
from xai.tabular.explainers import lime, shap, ale, ice, pdp, tree_interpreter
from xai.image import integrated_gradients
```

Install optional dependencies as needed:
```bash
python -m pip install lime shap captum treeinterpreter
```

Some explainability modules are starter/reference entry points; use the original notebooks for complete implementations.
