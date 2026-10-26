"""Complete tabular XAI practical from the uploaded notebook."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay, permutation_importance

data = fetch_california_housing(as_frame=True)
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
X_sample = X_test.sample(n=min(300, len(X_test)), random_state=42)
print("Model trained successfully!")

# 1. PDP
PartialDependenceDisplay.from_estimator(model, X_sample, ["MedInc"], kind="average")
plt.title("PDP - Average Effect of Median Income"); plt.tight_layout(); plt.show()

# 2. ICE
PartialDependenceDisplay.from_estimator(model, X_sample, ["MedInc"], kind="individual", subsample=30, random_state=42)
plt.title("ICE - Individual Effects of Median Income"); plt.tight_layout(); plt.show()

# 3. ALE (pip install PyALE)
from PyALE import ale
ale_result = ale(X=X_sample, model=model, feature=["MedInc"], grid_size=20, include_CI=False, plot=False)
plt.figure(figsize=(7, 4)); plt.plot(ale_result.index, ale_result["eff"], marker="."); plt.axhline(0, linestyle="--")
plt.xlabel("Median Income"); plt.ylabel("Accumulated Local Effect"); plt.title("ALE - Median Income"); plt.tight_layout(); plt.show()

# 4. Permutation Feature Importance
pfi = permutation_importance(model, X_sample, y_test.loc[X_sample.index], scoring="r2", n_repeats=5, random_state=42)
importance = pd.Series(pfi.importances_mean, index=X.columns).sort_values()
importance.tail(10).plot(kind="barh"); plt.xlabel("Mean decrease in R2"); plt.title("PFI - Feature Importance"); plt.tight_layout(); plt.show()

# 5. SHAP (pip install shap)
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)
shap.summary_plot(shap_values, X_sample)
shap.dependence_plot("MedInc", shap_values, X_sample)

# 6. TreeInterpreter (pip install treeinterpreter)
from treeinterpreter import treeinterpreter as ti
one_sample = X_test.iloc[[0]]
prediction, bias, contributions = ti.predict(model, one_sample)
contribution_series = pd.Series(np.asarray(contributions).reshape(-1), index=X.columns)
top_contributions = contribution_series.loc[contribution_series.abs().sort_values(ascending=False).head(10).index].sort_values()
top_contributions.plot(kind="barh"); plt.axvline(0, color="black", linewidth=0.8); plt.xlabel("Contribution to Prediction")
plt.title("TreeInterpreter - One Prediction"); plt.tight_layout(); plt.show()
print("\nActual value:", y_test.loc[one_sample.index[0]])
print("Predicted value:", prediction[0]); print("Baseline value:", np.asarray(bias).reshape(-1)[0])
print("\nFeature contributions:"); print(top_contributions.sort_values(key=abs, ascending=False))
