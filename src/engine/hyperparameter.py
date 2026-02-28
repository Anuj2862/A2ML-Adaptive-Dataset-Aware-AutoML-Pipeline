from sklearn.model_selection import GridSearchCV
import numpy as np


class HyperparameterOptimizationEngine:
    """
    Runs GridSearchCV for each model using predefined parameter grids.
    Falls back to the untuned model when no grid is defined.
    """

    def __init__(self):
        self.best_params = {}

    def get_default_param_grid(self, model_name: str) -> dict:
        """Returns the hyperparameter grid for a given model name."""
        grids = {
            # Regression
            "Linear Regression":       {},
            "Ridge":                   {"alpha": [0.01, 0.1, 1.0, 10.0]},
            "Lasso":                   {"alpha": [0.01, 0.1, 1.0, 10.0]},
            "SVR":                     {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]},
            "Decision Tree Regressor": {"max_depth": [None, 5, 10, 20]},
            "Random Forest Regressor": {"n_estimators": [50, 100], "max_depth": [None, 10]},
            "XGBoost Regressor":       {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1], "max_depth": [3, 5]},

            # Classification
            "SVM":           {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
            "KNN":           {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]},
            "Decision Tree": {"max_depth": [None, 5, 10, 20]},
            "Random Forest": {"n_estimators": [50, 100], "max_depth": [None, 10]},
            "Naive Bayes":   {},
            "XGBoost Classifier": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1], "max_depth": [3, 5]},

            # Clustering — no hyperparameter tuning via GridSearchCV
            "KMeans": {},
            "DBSCAN": {},
        }
        return grids.get(model_name, {})

    def optimize(self, model, X_train, y_train, model_name: str, cv: int = 3):
        """
        Runs GridSearchCV for the given model.
        Returns the best estimator (or original model if no grid defined).
        """
        param_grid = self.get_default_param_grid(model_name)

        if not param_grid:
            # No grid to search — fit directly
            model.fit(X_train, y_train)
            return model

        try:
            gs = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=cv,
                n_jobs=-1,
                error_score="raise",
            )
            gs.fit(X_train, y_train)
            self.best_params[model_name] = gs.best_params_
            return gs.best_estimator_
        except Exception as e:
            print(f"[HyperOpt] GridSearch failed for {model_name}: {e}. Using default params.")
            model.fit(X_train, y_train)
            return model
