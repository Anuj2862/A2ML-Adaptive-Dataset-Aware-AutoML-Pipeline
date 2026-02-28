import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class AdaptivePreprocessingEngine:
    """
    Automatically decides and applies imputation, encoding, and scaling.
    """
    def __init__(self):
        self.imputers = {}
        self.label_encoders = {}
        self.scaler = None
        self.scaler_type = None
        
    def fit_transform(self, df: pd.DataFrame, target_col: str, problem_type: str) -> pd.DataFrame:
        """Analyzes dataframe attributes and applies adaptive preprocessing."""
        processed_df = df.copy()
        
        # 1. Handle target separately (if classification, encode it)
        target_series = None
        if target_col in processed_df.columns:
            target_series = processed_df.pop(target_col)
            if problem_type == "classification" and target_series.dtype == 'object':
                le = LabelEncoder()
                target_series = pd.Series(le.fit_transform(target_series), name=target_col)
                self.label_encoders['__target__'] = le
                
        # 2. Imputation
        numerical_cols = processed_df.select_dtypes(include=[np.number]).columns
        categorical_cols = processed_df.select_dtypes(exclude=[np.number]).columns
        
        if len(numerical_cols) > 0:
            num_imputer = SimpleImputer(strategy='mean')
            processed_df[numerical_cols] = num_imputer.fit_transform(processed_df[numerical_cols])
            self.imputers['numerical'] = num_imputer
            
        if len(categorical_cols) > 0:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            processed_df[categorical_cols] = cat_imputer.fit_transform(processed_df[categorical_cols])
            self.imputers['categorical'] = cat_imputer
            
        # 3. Encoding (One-Hot for low cardinality categorical, Label Encoding for high cardinality)
        encoded_dfs = [processed_df[numerical_cols]] if len(numerical_cols) > 0 else []
        for col in categorical_cols:
            if processed_df[col].nunique() <= 10:
                # OneHot manually via pandas get_dummies for simplicity
                dummies = pd.get_dummies(processed_df[col], prefix=col, drop_first=True)
                encoded_dfs.append(dummies)
            else:
                le = LabelEncoder()
                encoded_col = pd.DataFrame(le.fit_transform(processed_df[col]), columns=[col])
                encoded_dfs.append(encoded_col)
                self.label_encoders[col] = le
                
        if encoded_dfs:
            processed_df = pd.concat(encoded_dfs, axis=1)
            
        # 4. Scaling (StandardScaler generally safe, MinMaxScaler if skewed or strictly positive knowns)
        # Using StandardScaler as default policy.
        self.scaler = StandardScaler()
        self.scaler_type = "StandardScaler"
        processed_df_scaled = pd.DataFrame(
            self.scaler.fit_transform(processed_df), 
            columns=processed_df.columns
        )
        
        if target_series is not None:
             processed_df_scaled = pd.concat([processed_df_scaled, target_series], axis=1)
             
        return processed_df_scaled
