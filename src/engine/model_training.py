from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, DBSCAN

# XGBoost — optional, falls back gracefully if system library missing
try:
    from xgboost import XGBClassifier, XGBRegressor
    has_xgb = True
except Exception:
    has_xgb = False
    print("[A2ML] XGBoost not available. Skipping XGB models.")

import numpy as np


class MultiModelLearningEngine:
    """
    Instantiates multiple ML models according to problem type.

    Supported problem types:
        - regression    : Linear, Ridge, Lasso, SVR, Random Forest, XGBoost
        - classification: SVM, KNN, Decision Tree, Random Forest, Naive Bayes, XGBoost
        - clustering    : KMeans, DBSCAN
    """

    def __init__(self):
        self.models = {}

    def get_models(self, problem_type: str) -> dict:
        """Returns a dictionary of raw model instances based on problem type."""
        models_dict = {}

        if problem_type == "regression":
            models_dict = {
                "Linear Regression": LinearRegression(),
                "Ridge":             Ridge(),
                "Lasso":             Lasso(),
                "SVR":               SVR(),
                "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
                "Random Forest Regressor": RandomForestRegressor(random_state=42),
            }
            if has_xgb:
                models_dict["XGBoost Regressor"] = XGBRegressor(
                    random_state=42, n_estimators=100, eval_metric="rmse"
                )

        elif problem_type == "classification":
            models_dict = {
                "SVM":           SVC(probability=True, random_state=42),
                "KNN":           KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Random Forest": RandomForestClassifier(random_state=42),
                "Naive Bayes":   GaussianNB(),
            }
            if has_xgb:
                models_dict["XGBoost Classifier"] = XGBClassifier(
                    random_state=42, eval_metric="logloss"
                )

        elif problem_type == "clustering":
            models_dict = {
                "KMeans": KMeans(n_clusters=3, random_state=42, n_init="auto"),
                "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
            }

        else:
            raise ValueError(f"Unknown problem type: {problem_type}")

        self.models = models_dict
        return models_dict
