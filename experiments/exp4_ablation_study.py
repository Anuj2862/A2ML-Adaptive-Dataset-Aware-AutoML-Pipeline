"""
Experiment 4 (Ablation Study): Without Preprocessing | Without Feature Eng | Full System
Compares the three configurations on the heart disease classification dataset.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from src.engine.pipeline import A2MLPipeline

DATASET = "data/heart_disease.csv"
TARGET  = "target"

print("\n── Experiment 4: Ablation Study ──")
print(f"  Dataset: {DATASET}  |  Target: {TARGET}\n")

results_table = []

# ── Config A: No Preprocessing, No Feature Eng ──────────────────────────
print("Config A: Baseline — no preprocessing, no feature engineering")
df = pd.read_csv(DATASET)
X  = df.drop(columns=[TARGET]).fillna(0)
y  = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
m  = RandomForestClassifier(random_state=42)
m.fit(X_train, y_train)
f1 = f1_score(y_test, m.predict(X_test), average="weighted")
acc= accuracy_score(y_test, m.predict(X_test))
results_table.append({"Config": "A — Baseline (no preprocessing)", "Accuracy": round(acc,4), "F1": round(f1,4)})
print(f"   Accuracy={acc:.4f}  F1={f1:.4f}")

# ── Config B: Preprocessing ON, Feature Eng OFF ─────────────────────────
print("Config B: Preprocessing ON, Feature Engineering OFF")
try:
    p = A2MLPipeline(DATASET)
    r = p.run_pipeline(target_column=TARGET, feature_opt_strategy="none", apply_hyperopt=False)
    bench = r["benchmark_results"]
    best = r["best_model_name"]
    f1  = bench.loc[bench["Model"]==best,"Composite Score"].values[0]
    acc = bench.loc[bench["Model"]==best,"Accuracy"].values[0]
    results_table.append({"Config": "B — Preprocessing only", "Accuracy": round(acc,4), "F1": round(f1,4)})
    print(f"   Accuracy={acc:.4f}  F1={f1:.4f}  Best={best}")
except Exception as e:
    print(f"   Error: {e}")

# ── Config C: Full System ───────────────────────────────────────────────
print("Config C: Full A²ML System (preprocessing + feature eng + hyperopt)")
try:
    p = A2MLPipeline(DATASET)
    r = p.run_pipeline(target_column=TARGET, feature_opt_strategy="auto", apply_hyperopt=True)
    bench = r["benchmark_results"]
    best = r["best_model_name"]
    f1  = bench.loc[bench["Model"]==best,"Composite Score"].values[0]
    acc = bench.loc[bench["Model"]==best,"Accuracy"].values[0]
    results_table.append({"Config": "C — Full A²ML System", "Accuracy": round(acc,4), "F1": round(f1,4)})
    print(f"   Accuracy={acc:.4f}  F1={f1:.4f}  Best={best}")
except Exception as e:
    print(f"   Error: {e}")

print("\n── Ablation Summary ──")
print(pd.DataFrame(results_table).to_string(index=False))
print("\n── Done ──")
