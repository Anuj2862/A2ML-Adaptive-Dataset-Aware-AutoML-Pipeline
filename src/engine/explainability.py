import shap
import pandas as pd
import numpy as np
from sklearn.inspection import partial_dependence

class ExplainableAIEngine:
    """Uses SHAP to explain best model predictions."""
    def __init__(self):
        self.explainer = None
        self.shap_values = None
        
    def explain_model(self, model, X: pd.DataFrame, max_display: int = 10) -> dict:
        """
        Generates generic feature importance using SHAP.
        Returns a dictionary mapping feature names to importance scores.
        """
        if X is None or X.shape[1] == 0:
            return {}
            
        # Due to complexity with differing model types and SHAP explainer compatibility:
        # Use TreeExplainer for Tree models, LinearExplainer for linear, KernelExplainer as fallback
        model_class = type(model).__name__
        
        try:
            sample_X = shap.sample(X, 100) if len(X) > 100 else X
            
            if 'Forest' in model_class or 'Tree' in model_class or 'XGB' in model_class:
                self.explainer = shap.TreeExplainer(model)
                self.shap_values = self.explainer.shap_values(sample_X)
            elif 'Linear' in model_class or 'Ridge' in model_class or 'Lasso' in model_class:
                self.explainer = shap.LinearExplainer(model, sample_X)
                self.shap_values = self.explainer.shap_values(sample_X)
            else:
                # KernelExplainer is slow, so we take a very small sample
                kernel_sample = shap.kmeans(X, 10) if len(X) > 10 else X
                predict_func = getattr(model, "predict_proba", model.predict)
                self.explainer = shap.KernelExplainer(predict_func, kernel_sample)
                self.shap_values = self.explainer.shap_values(shap.sample(X, 50))
                
        except Exception as e:
            print(f"SHAP Explainer Error for {model_class}: {e}")
            return {}
            
        # Compile importance dictionary
        try:
            if isinstance(self.shap_values, list):
                # Multiclass classification often returns a list of shap arrays
                importances = np.abs(self.shap_values[0]).mean(0)
            else:
                importances = np.abs(self.shap_values).mean(0)
                
            if len(importances.shape) > 1:
                importances = importances.mean(axis=1) # Flatten if 2D
                
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': importances
            })
            feature_importance = feature_importance.sort_values(by='Importance', ascending=False).head(max_display)
            
            top_feature = feature_importance['Feature'].iloc[0] if not feature_importance.empty else None
            pdp_result = None
            
            if top_feature:
                try:
                    # Attempt calculating partial dependence for the top feature
                    pd_res = partial_dependence(model, X, [top_feature], grid_resolution=50)
                    pdp_result = {
                        "feature": top_feature,
                        "values": pd_res["grid_values"][0].tolist(),
                        "dependence": pd_res["average"][0].tolist()
                    }
                except Exception as e:
                    print(f"Failed to compute PDP for {top_feature}: {e}")
            
            return {
                "importance": dict(zip(feature_importance['Feature'], feature_importance['Importance'])),
                "pdp": pdp_result
            }
            
        except Exception as e:
             print(f"Error computing SHAP aggregations: {e}")
             return {"importance": {}, "pdp": None}
