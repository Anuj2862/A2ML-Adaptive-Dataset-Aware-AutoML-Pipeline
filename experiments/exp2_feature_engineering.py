"""
Experiment 2: Effect of Feature Engineering
Runs the pipeline with feature engineering ON vs OFF.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.engine.pipeline import A2MLPipeline

DATASET = "data/air_quality.csv"
TARGET  = "pm25"

print("\n── Experiment 2: Effect of Feature Engineering ──")

for strategy in ["none", "mutual_info", "interactions", "pca"]:
    print(f"\n  Strategy: {strategy.upper()}")
    try:
        p = A2MLPipeline(DATASET)
        r = p.run_pipeline(target_column=TARGET, feature_opt_strategy=strategy, apply_hyperopt=False)
        bench = r["benchmark_results"]
        best  = r["best_model_name"]
        rmse  = bench.loc[bench["Model"] == best, "Composite Score"].values[0] if "Composite Score" in bench.columns else "N/A"
        print(f"  Best Model : {best}")
        print(f"  RMSE       : {rmse}")
        print(f"  Features   : {len(r['optimized_features'])}")
    except Exception as e:
        print(f"  Error: {e}")

print("\n── Done ──")
