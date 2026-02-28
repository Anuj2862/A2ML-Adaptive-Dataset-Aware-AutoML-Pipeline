import pandas as pd
import numpy as np


class DataInputEngine:
    """
    Handles CSV dataset loading, target column detection,
    and problem type identification.

    Supported problem types: regression | classification | clustering
    """

    def __init__(self):
        self.df = None
        self.target_col = None
        self.problem_type = None

    def load_data(self, file_path_or_buffer) -> pd.DataFrame:
        """Loads data from a CSV file path or uploaded file buffer."""
        self.df = pd.read_csv(file_path_or_buffer)
        return self.df

    def detect_target_column(self, target_col: str = None) -> str:
        """
        Sets the target column.
        If not provided, uses common naming heuristics to find a candidate.
        """
        if target_col:
            self.target_col = target_col
            return self.target_col

        # Heuristic: common target-like column names
        common_targets = [
            "target", "label", "class", "output", "y",
            "price", "salary", "diagnosis", "species",
            "medv", "result", "outcome"
        ]
        for col in self.df.columns:
            if col.lower() in common_targets:
                self.target_col = col
                return self.target_col

        # Fall back: last column
        self.target_col = self.df.columns[-1]
        return self.target_col

    def detect_problem_type(self) -> str:
        """
        Detects if the problem is regression, classification, or clustering.

        Rules (from official scope):
          - No target column     → clustering
          - Categorical target   → classification
          - Numeric target with few unique values (≤20 or <5 % of rows) → classification
          - Numeric target with many unique values → regression
        """
        if self.target_col is None:
            self.problem_type = "clustering"
            return self.problem_type

        target_series = self.df[self.target_col]

        if pd.api.types.is_numeric_dtype(target_series):
            unique_vals = target_series.nunique()
            if unique_vals <= 20 or unique_vals < 0.05 * len(target_series):
                self.problem_type = "classification"
            else:
                self.problem_type = "regression"
        else:
            self.problem_type = "classification"

        return self.problem_type

    def get_features_and_target(self):
        """
        Returns (X, y) split.
        For clustering, y is None.
        """
        if self.target_col is None or self.target_col not in self.df.columns:
            return self.df.copy(), None

        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        return X, y
