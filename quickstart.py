import xai
from xai.tabular.regression import linear, multiple, polynomial
from xai.tabular.classification import iris

print("XAI version:", xai.__version__)
help(linear)
# Run one example:
# linear.run()
# multiple.run()
# polynomial.run(degree=3)
# iris.run()
