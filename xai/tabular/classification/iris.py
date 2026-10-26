"""Iris classification example."""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def run():
    """Train a scaled logistic-regression classifier on Iris and print accuracy."""
    iris_data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris_data.data, iris_data.target, test_size=0.2,
        random_state=42, stratify=iris_data.target
    )
    scaler = StandardScaler().fit(X_train)
    model = LogisticRegression(max_iter=200, random_state=42).fit(
        scaler.transform(X_train), y_train
    )
    print("Accuracy:", accuracy_score(y_test, model.predict(scaler.transform(X_test))))
    return model, scaler
