import os
import sys

# Ensure src can be imported
sys.path.insert(0, os.path.abspath('.'))

from src.engine.pipeline import A2MLPipeline
import pandas as pd

def test_pipeline_classification():
    print("--- Testing Classification Pipeline on Iris Data ---")
    pipeline = A2MLPipeline("data/iris.csv")
    results = pipeline.run_pipeline(target_column="target", apply_hyperopt=False)
    print(f"Detected Problem Type: {results['problem_type']}")
    print(f"Target: {results['target']}")
    print(f"Best Model: {results['best_model_name']}")
    print("Benchmark Results:")
    print(results['benchmark_results'].to_string())
    print("Memory Recommendation:", results['recommended_from_memory'])
    print("\n--------------------------\n")

def test_pipeline_regression():
    print("--- Testing Regression Pipeline on Housing Data ---")
    pipeline = A2MLPipeline("data/housing.csv")
    results = pipeline.run_pipeline(target_column="target", apply_hyperopt=False)
    print(f"Detected Problem Type: {results['problem_type']}")
    print(f"Target: {results['target']}")
    print(f"Best Model: {results['best_model_name']}")
    print("Benchmark Results:")
    print(results['benchmark_results'].to_string())
    print("\n--------------------------\n")

if __name__ == "__main__":
    if os.path.exists("data/iris.csv") and os.path.exists("data/housing.csv"):
        test_pipeline_classification()
        test_pipeline_regression()
        print("ALL TESTS PASSED SUCCESSFULLY.")
    else:
        print("Missing data files. Generate samples first.")
