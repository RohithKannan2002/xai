"""Multiple linear regression on the Diabetes dataset."""
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def run():
    """Train a multiple-feature linear regression model and print its score."""
    data = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    model = LinearRegression().fit(X_train, y_train)
    print("R² Score:", model.score(X_test, y_test))
    return model
