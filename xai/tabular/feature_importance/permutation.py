"""Permutation feature importance example."""
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance

def run():
    """Fit linear regression and print permutation importance scores."""
    data = load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    X_train, X_test, y_train, y_test = train_test_split(
        X, data.target, test_size=0.2, random_state=42
    )
    model = LinearRegression().fit(X_train, y_train)
    result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42,
        scoring="neg_mean_absolute_error"
    )
    print(sorted(zip(X.columns, result.importances_mean),
                 key=lambda item: item[1], reverse=True))
    return model, result
