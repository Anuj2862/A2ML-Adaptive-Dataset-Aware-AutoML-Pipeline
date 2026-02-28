import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

class FeatureOptimizationEngine:
    """
    Advanced feature engineering: PCA, correlation threshold, or information scoring.
    """
    def __init__(self):
        self.pca = None
        self.selector = None
        
    def optimize_features(self, X: pd.DataFrame, y: pd.Series, problem_type: str, strategy: str = "auto") -> pd.DataFrame:
        """
        Applies feature optimization based on the chosen strategy.
        Strategies: 'auto', 'pca', 'mutual_info', 'interactions', 'none'
        """
        if strategy == "none" or X.shape[1] < 2:
            return X
            
        if strategy == "auto":
            # Heuristic: if high dimensionality relative to rows, use PCA. 
            # If low dimensionality (under 10), try adding interactions. Else mutual info.
            if X.shape[1] > 20 and len(X) < X.shape[1] * 10:
                strategy = "pca"
            elif X.shape[1] <= 10:
                strategy = "interactions"
            else:
                strategy = "mutual_info"
                
        if strategy == "interactions":
            # Select up to 5 highest variance numeric columns to interact to prevent explosion
            num_cols = X.select_dtypes(include=[np.number]).columns
            if len(num_cols) >= 2:
                top_cols = X[num_cols].var().nlargest(5).index
                poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
                poly_features = poly.fit_transform(X[top_cols])
                
                # Create DataFrame for new features
                feature_names = poly.get_feature_names_out(top_cols)
                X_interactions = pd.DataFrame(poly_features, columns=feature_names, index=X.index)
                
                # Avoid duplicate columns that already existed 
                new_cols = [c for c in X_interactions.columns if c not in X.columns]
                X = pd.concat([X, X_interactions[new_cols]], axis=1)
            return X
            
        elif strategy == "pca":
            n_components = min(X.shape[1] - 1, 10, len(X))
            if n_components <= 0:
                return X
            self.pca = PCA(n_components=n_components)
            X_pca = self.pca.fit_transform(X)
            columns = [f"pc_{i+1}" for i in range(n_components)]
            return pd.DataFrame(X_pca, columns=columns, index=X.index)
            
        elif strategy == "mutual_info" and y is not None:
            # Drop features with very low mutual information score
            score_func = mutual_info_classif if problem_type == "classification" else mutual_info_regression
            k = min(X.shape[1] - 1, int(X.shape[1] * 0.8)) # Keep top 80% features
            k = max(k, 1)
            
            try:
                self.selector = SelectKBest(score_func=score_func, k=k)
                X_selected = self.selector.fit_transform(X, y)
                selected_cols = X.columns[self.selector.get_support()]
                return pd.DataFrame(X_selected, columns=selected_cols, index=X.index)
            except Exception:
                # Fallback if scoring fails
                return X
        
        return X
