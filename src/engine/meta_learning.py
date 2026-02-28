import pandas as pd
import numpy as np
import scipy.stats as stats


class MetaLearningEngine:
    """
    Analyzes dataset characteristics and computes:
      - Dataset statistics (rows, columns, types, missing values)
      - Class imbalance check
      - Dataset Complexity Score (official formula)

    Complexity Formula (from project specification):
        complexity = 0.4 * (num_features / num_samples)
                   + 0.3 * (missing_ratio)
                   + 0.3 * (skewness_score)
    """

    def __init__(self):
        self.stats_report = {}

    # ─────────────────────────────────────────────
    def analyze_dataset(
        self, df: pd.DataFrame, target_col: str, problem_type: str
    ) -> dict:
        """Generates a comprehensive dataset analysis report."""

        num_samples = len(df)
        num_features = len(df.columns) - (1 if target_col and target_col in df.columns else 0)
        missing_ratio = df.isnull().sum().sum() / max(df.size, 1)
        skewness_score = self._compute_skewness_score(df, target_col)

        # ── Official complexity formula ──────────────────────────
        raw_complexity = (
            0.4 * (num_features / max(num_samples, 1))
            + 0.3 * missing_ratio
            + 0.3 * skewness_score
        )
        complexity_label = self._label_complexity(raw_complexity)
        # ────────────────────────────────────────────────────────
        
        # ── Advanced Similarity Metrics ────────────────────────
        mean_target_correlation = self._compute_target_correlation(df, target_col, problem_type)
        mean_feature_entropy = self._compute_mean_entropy(df, target_col)
        # ────────────────────────────────────────────────────────

        report = {
            "num_rows":          num_samples,
            "num_columns":       len(df.columns),
            "num_features":      num_features,
            "target_column":     target_col,
            "problem_type":      problem_type,
            "memory_usage_mb":   round(df.memory_usage(deep=True).sum() / (1024 ** 2), 3),
            "missing_ratio":     round(missing_ratio, 4),
            "skewness_score":    round(skewness_score, 4),
            "mean_target_correlation": round(mean_target_correlation, 4),
            "mean_feature_entropy":    round(mean_feature_entropy, 4),
            "raw_complexity":    round(raw_complexity, 4),
            "data_complexity_score": complexity_label,
            "missing_values_summary": self._analyze_missing_values(df),
            "data_types":        self._analyze_data_types(df),
            "class_imbalance":   None,
        }

        if problem_type == "classification" and target_col and target_col in df.columns:
            report["class_imbalance"] = self._check_class_imbalance(df[target_col])

        self.stats_report = report
        return report

    # ─────────────────────────────────────────────
    def _compute_skewness_score(self, df: pd.DataFrame, target_col: str) -> float:
        """
        Computes the mean absolute skewness of all numeric columns (excluding target).
        Normalises to a [0, 1] range using a cap at skewness = 10.
        """
        numeric_df = df.select_dtypes(include=[np.number])
        if target_col and target_col in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=[target_col])

        if numeric_df.empty:
            return 0.0

        abs_skews = numeric_df.apply(lambda col: abs(col.dropna().skew()) if len(col.dropna()) > 2 else 0)
        mean_skew = abs_skews.mean()
        return float(min(mean_skew / 10.0, 1.0))  # Cap at 1.0

    def _compute_target_correlation(self, df: pd.DataFrame, target_col: str, problem_type: str) -> float:
        """Computes mean absolute Pearson correlation of numeric features with the target (regression)."""
        if problem_type != "regression" or not target_col or target_col not in df.columns:
            return 0.0
        
        numeric_df = df.select_dtypes(include=[np.number])
        if target_col not in numeric_df.columns or len(numeric_df.columns) < 2:
            return 0.0
            
        corr_matrix = numeric_df.corr()
        target_corr = corr_matrix[target_col].drop(target_col, errors="ignore")
        return float(target_corr.abs().mean(skipna=True)) if not target_corr.isna().all() else 0.0
        
    def _compute_mean_entropy(self, df: pd.DataFrame, target_col: str) -> float:
        """Computes mean Shannon entropy of categorical columns as a proxy for info richness."""
        cat_df = df.select_dtypes(exclude=[np.number])
        if target_col and target_col in cat_df.columns:
            cat_df = cat_df.drop(columns=[target_col])
            
        if cat_df.empty:
            return 0.0
            
        entropies = []
        for col in cat_df.columns:
            value_counts = cat_df[col].value_counts(normalize=True)
            if not value_counts.empty:
                entropies.append(stats.entropy(value_counts))
                
        return float(np.mean(entropies)) if entropies else 0.0

    def _label_complexity(self, raw_score: float) -> str:
        """Maps numeric complexity score to a human-readable label."""
        if raw_score < 0.15:
            return "Low"
        elif raw_score < 0.35:
            return "Medium"
        else:
            return "High"

    def _analyze_missing_values(self, df: pd.DataFrame) -> dict:
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        missing_pct = (missing / len(df)) * 100
        return {
            "total_missing_cells":  int(df.isnull().sum().sum()),
            "columns_with_missing": int(len(missing)),
            "details": {
                col: {"count": int(cnt), "percentage": round(float(pct), 2)}
                for col, cnt, pct in zip(missing.index, missing, missing_pct)
            },
        }

    def _analyze_data_types(self, df: pd.DataFrame) -> dict:
        numerical_cols   = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        return {
            "numerical_columns":   numerical_cols,
            "categorical_columns": categorical_cols,
            "numerical_count":     len(numerical_cols),
            "categorical_count":   len(categorical_cols),
        }

    def _check_class_imbalance(self, target_series: pd.Series) -> dict:
        value_counts  = target_series.value_counts(normalize=True) * 100
        is_imbalanced = value_counts.min() < 10.0
        return {
            "is_imbalanced": is_imbalanced,
            "class_distribution_percentages": {str(k): round(v, 2) for k, v in value_counts.items()},
        }
