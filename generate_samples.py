import pandas as pd
from sklearn.datasets import load_iris, fetch_california_housing
import os

os.makedirs('data', exist_ok=True)

# 1. Classification Data
iris = load_iris()
df_class = pd.DataFrame(iris.data, columns=iris.feature_names)
df_class['target'] = iris.target
df_class.to_csv('data/iris.csv', index=False)

# 2. Regression Data
ca = fetch_california_housing()
df_reg = pd.DataFrame(ca.data, columns=ca.feature_names)
df_reg['target'] = ca.target
# Sample to speed up test
df_reg.sample(500, random_state=42).to_csv('data/housing.csv', index=False)

print("Sample datasets generated in 'data/' directory.")
