"""Simple linear regression on the Diabetes dataset."""
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def run():
    """Train a one-feature BMI linear regression model and print its R² score."""
    data = load_diabetes()
    X, y = data.data[:, [2]], data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression().fit(X_train, y_train)
    print("R² Score:", r2_score(y_test, model.predict(X_test)))
    print("Slope:", model.coef_[0])
    print("Intercept:", model.intercept_)
    return model
