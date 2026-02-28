"""
Experiment 1: Pipeline vs Single Baseline Model
Compares A²ML's full pipeline results against a single default model.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from src.engine.pipeline import A2MLPipeline

DATASET = "data/iris.csv"
TARGET  = "target"

print("\n── Experiment 1: Pipeline vs Single Baseline ──")

# ── A²ML Full Pipeline ────────────────────────────────
pipeline = A2MLPipeline(DATASET)
results  = pipeline.run_pipeline(target_column=TARGET, apply_hyperopt=False)
bench    = results["benchmark_results"]
best_f1  = bench["Composite Score"].max() if "Composite Score" in bench.columns else None

print(f"\nA²ML Best Model  : {results['best_model_name']}")
print(f"A²ML Best F1     : {best_f1:.4f}" if best_f1 else "")
print(f"Complexity Score : {results['dataset_report']['data_complexity_score']}")

# ── Single Baseline (Random Forest, no hyperopt, no feature engineering) ──
df = pd.read_csv(DATASET)
X  = pd.get_dummies(df.drop(columns=[TARGET]))
y  = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

baseline = RandomForestClassifier(random_state=42)
baseline.fit(X_train, y_train)
preds    = baseline.predict(X_test)
baseline_f1 = f1_score(y_test, preds, average="weighted")

print(f"\nBaseline RF F1   : {baseline_f1:.4f}")
print(f"\nImprovement      : {((best_f1 - baseline_f1) / baseline_f1 * 100):.1f}%" if best_f1 else "")
print("\n── Done ──")
