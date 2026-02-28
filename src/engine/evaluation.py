from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
import time

class ModelEvaluationEngine:
    """Evaluates trained models and handles auto-model selection using composite scoring."""
    def __init__(self):
        self.results_df = None
        self.best_model_name = None
        
    def evaluate_models(self, trained_models: dict, X_train, y_train, X_test, y_test, problem_type: str) -> pd.DataFrame:
        """Evaluates multiple models, applies overfitting penalty, and returns a benchmark dataframe."""
        results = []
        
        for name, model in trained_models.items():
            if problem_type == "classification":
                # Train metrics to compute overfitting
                train_pred = model.predict(X_train)
                train_f1 = f1_score(y_train, train_pred, average='weighted', zero_division=0)
                
                # Test metrics & Inference Time
                start_time = time.time()
                y_pred = model.predict(X_test)
                inf_time = time.time() - start_time
                
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                test_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                # Overfitting Penalty
                overfitting_penalty = max(0.0, train_f1 - test_f1)
                
                # Composite Score (higher is better)
                # Formula: Test_F1 - (0.5 * Overfit) - (0.01 * Log(Inference Time))
                # Using 0.01 * inf_time directly for simplicity
                composite_score = test_f1 - (0.5 * overfitting_penalty) - (0.01 * inf_time)
                
                roc_auc = "N/A"
                if hasattr(model, "predict_proba"):
                    try:
                        y_prob = model.predict_proba(X_test)
                        if len(np.unique(y_test)) == 2: # Binary
                            roc_auc = roc_auc_score(y_test, y_prob[:, 1])
                        else: # Multiclass
                            roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
                    except Exception:
                        pass
                        
                results.append({
                    "Model": name,
                    "Accuracy": acc,
                    "Precision": prec,
                    "Recall": rec,
                    "F1 Score": test_f1,
                    "ROC AUC": roc_auc,
                    "Train F1": train_f1,
                    "Overfit Penalty": round(overfitting_penalty, 4),
                    "Inference Time(s)": round(inf_time, 4),
                    "Composite Score": round(composite_score, 4)
                })
                
            elif problem_type == "regression":
                # Train metrics
                train_pred = model.predict(X_train)
                train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
                
                # Test metrics
                start_time = time.time()
                y_pred = model.predict(X_test)
                inf_time = time.time() - start_time
                
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Overfitting Penalty (RMSE is lower=better, so penalty if test is worse than train)
                overfitting_penalty = max(0.0, test_rmse - train_rmse)
                
                # Composite Score (lower is better)
                # Formula: Test_RMSE + (0.5 * Overfit) + (0.01 * Inference Time)
                composite_score = test_rmse + (0.5 * overfitting_penalty) + (0.01 * inf_time)
                
                results.append({
                    "Model": name,
                    "RMSE": test_rmse,
                    "MAE": mae,
                    "R² Score": r2,
                    "Train RMSE": train_rmse,
                    "Overfit Penalty": round(overfitting_penalty, 4),
                    "Inference Time(s)": round(inf_time, 4),
                    "Composite Score": round(composite_score, 4)
                })
                
            elif problem_type == "clustering":
                try:
                    start_time = time.time()
                    y_pred = model.fit_predict(X_test)
                    inf_time = time.time() - start_time
                    
                    if len(np.unique(y_pred)) > 1: # Requires at least 2 clusters
                        sil = silhouette_score(X_test, y_pred)
                    else:
                        sil = -1
                except Exception:
                    sil = -1
                    inf_time = 0.0
                    
                results.append({
                    "Model": name,
                    "Silhouette Score": sil,
                    "Inference Time(s)": round(inf_time, 4),
                    "Composite Score": sil  # Proxy
                })
                
        self.results_df = pd.DataFrame(results)
        return self.results_df
        
    def auto_select_best_model(self, problem_type: str) -> str:
        """Automatically chooses the best performing model based on composite scores."""
        if self.results_df is None or self.results_df.empty:
            raise ValueError("Models must be evaluated first.")
            
        if problem_type == "regression":
            # Lowest Composite Score (lower is better for RMSE-based)
            best_idx = self.results_df['Composite Score'].idxmin()
            
        elif problem_type == "classification":
            # Highest Composite Score
            best_idx = self.results_df['Composite Score'].idxmax()
            
        elif problem_type == "clustering":
            # Highest Silhouette
            best_idx = self.results_df.loc[self.results_df['Silhouette Score'] != -1, 'Composite Score'].idxmax()
            if pd.isna(best_idx): 
                # Fallback if no valid silhouette
                best_idx = self.results_df.index[0]
                
        self.best_model_name = self.results_df.loc[best_idx, 'Model']
        return self.best_model_name
