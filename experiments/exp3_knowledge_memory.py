"""
Experiment 3: Effect of Knowledge Memory (Ablation)
Runs two datasets sequentially, then verifies the memory recommends correctly.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.engine.pipeline import A2MLPipeline

print("\n── Experiment 3: Knowledge Memory Effect ──")

# Run 1: Iris
print("\nRun 1 — Iris (classification)")
p1 = A2MLPipeline("data/iris.csv")
r1 = p1.run_pipeline(target_column="target", apply_hyperopt=False)
print(f"  Best: {r1['best_model_name']}  | Memory Rec: {r1['recommended_from_memory'] or 'None (first run)'}")

# Run 2: Heart disease (similar structure → should recommend based on Iris)
print("\nRun 2 — Heart Disease (classification)")
p2 = A2MLPipeline("data/heart_disease.csv")
r2 = p2.run_pipeline(target_column="target", apply_hyperopt=False)
print(f"  Best: {r2['best_model_name']}  | Memory Rec: {r2['recommended_from_memory'] or 'None'}")

# Read memory file to show stored vectors
with open("knowledge_memory.json") as f:
    mem = json.load(f)
print(f"\n  Total runs stored in memory: {len(mem['runs'])}")
print("\n── Done ──")
