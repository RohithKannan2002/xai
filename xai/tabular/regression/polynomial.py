"""Polynomial regression on the BMI feature of the Diabetes dataset."""
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def run(degree=3):
    """Fit polynomial regression and print the in-sample R² score."""
    data = load_diabetes()
    X, y = data.data[:, [2]], data.target
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)
    print(f"Polynomial degree: {degree}")
    print("R² Score:", r2_score(y, model.predict(X_poly)))
    return model, poly
