"""
Experiment 5: Statistical Significance Testing
Performs a T-test to determine if the best pipeline model is statistically significantly
better than the baseline model over multiple randomized train/test splits.
"""
import sys, os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.engine.pipeline import A2MLPipeline

DATASET = "data/heart_disease.csv"
TARGET = "target"
N_ITERATIONS = 5

print(f"\n── Experiment 5: Statistical Significance Testing (N={N_ITERATIONS}) ──")

pipeline_scores = []
baseline_scores = []

for i in range(N_ITERATIONS):
    print(f"Running iteration {i+1}...")
    
    # ── Pipeline Run ──
    try:
        p = A2MLPipeline(DATASET)
        r = p.run_pipeline(target_column=TARGET, apply_hyperopt=False)
        bench = r["benchmark_results"]
        # Use Test F1 from the new composite metrics output
        best_f1 = bench.loc[bench["Model"] == r["best_model_name"], "F1 Score"].values[0]
        pipeline_scores.append(best_f1)
    except Exception as e:
        print(f"Pipeline failed on iteration {i+1}: {e}")
        pipeline_scores.append(0.0)

    # ── Baseline Run ──
    df = pd.read_csv(DATASET)
    X = df.drop(columns=[TARGET]).fillna(0)
    y = df[TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i*42)
    m = RandomForestClassifier(random_state=i*42)
    m.fit(X_train, y_train)
    
    base_f1 = f1_score(y_test, m.predict(X_test), average="weighted", zero_division=0)
    baseline_scores.append(base_f1)

# ── T-Test ──
t_stat, p_value = stats.ttest_rel(pipeline_scores, baseline_scores)

print("\n── Results ──")
print(f"Pipeline Mean F1: {np.mean(pipeline_scores):.4f} ± {np.std(pipeline_scores):.4f}")
print(f"Baseline Mean F1: {np.mean(baseline_scores):.4f} ± {np.std(baseline_scores):.4f}")
print(f"T-Statistic     : {t_stat:.4f}")
print(f"P-Value         : {p_value:.4f}")

if p_value < 0.05:
    print("\n✅ Conclusion: The A²ML Pipeline is STATISTICALLY SIGNIFICANTLY better than the baseline (p < 0.05).")
else:
    print("\n❌ Conclusion: The performance difference is NOT statistically significant for this sample size/dataset.")

print("── Done ──")
