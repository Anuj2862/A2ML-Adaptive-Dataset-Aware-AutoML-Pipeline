#!/usr/bin/env python3
"""
A²ML — Adaptive Autonomous Machine Learning System
CLI Entry Point

Usage:
    python main.py --dataset data/iris.csv --target target
    python main.py --dataset data/housing.csv --target target
    python main.py --dataset data/customer_segmentation.csv   (clustering — no target)
    python main.py --dataset data/iris.csv --target target --no-hyperopt
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.engine.pipeline import A2MLPipeline
from src.utils.logger import get_logger

logger = get_logger("A2ML-CLI")


def parse_args():
    parser = argparse.ArgumentParser(
        description="A²ML — Adaptive Autonomous Machine Learning System"
    )
    parser.add_argument(
        "--dataset", "-d", required=True,
        help="Path to the CSV dataset file"
    )
    parser.add_argument(
        "--target", "-t", default=None,
        help="Target column name (leave blank for clustering)"
    )
    parser.add_argument(
        "--strategy", "-s", default="auto",
        choices=["auto", "pca", "mutual_info", "interactions", "none"],
        help="Feature engineering strategy (default: auto)"
    )
    parser.add_argument(
        "--no-hyperopt", action="store_true",
        help="Skip hyperparameter optimisation (faster)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("  A²ML — Adaptive Autonomous Machine Learning System")
    logger.info("=" * 60)
    logger.info(f"  Dataset  : {args.dataset}")
    logger.info(f"  Target   : {args.target or '(auto-clustering)'}")
    logger.info(f"  Strategy : {args.strategy}")
    logger.info(f"  HyperOpt : {not args.no_hyperopt}")
    logger.info("=" * 60)

    if not os.path.exists(args.dataset):
        logger.error(f"Dataset not found: {args.dataset}")
        sys.exit(1)

    start = time.time()
    dataset_name = os.path.basename(args.dataset)

    try:
        pipeline = A2MLPipeline(args.dataset)
        results  = pipeline.run_pipeline(
            target_column=args.target,
            feature_opt_strategy=args.strategy,
            apply_hyperopt=not args.no_hyperopt,
            dataset_name=dataset_name,
        )
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    elapsed = round(time.time() - start, 2)

    # ── Print Results ────────────────────────────────
    rep = results["dataset_report"]
    print("\n" + "─" * 55)
    print("  DATASET ANALYSIS")
    print("─" * 55)
    print(f"  Problem Type     : {results['problem_type'].upper()}")
    print(f"  Target Column    : {results['target']}")
    print(f"  Rows             : {rep['num_rows']}")
    print(f"  Features         : {rep['num_features']}")
    print(f"  Missing Ratio    : {rep['missing_ratio']}")
    print(f"  Skewness Score   : {rep['skewness_score']}")
    print(f"  Complexity Score : {rep['raw_complexity']} ({rep['data_complexity_score']})")

    if results["recommended_from_memory"]:
        print(f"\n  🧠 Memory Recommends : {results['recommended_from_memory']}")

    print("\n" + "─" * 55)
    print("  MODEL BENCHMARK RESULTS")
    print("─" * 55)
    print(results["benchmark_results"].to_string(index=False))

    print("\n" + "─" * 55)
    print(f"  🏆 Best Model Auto-Selected : {results['best_model_name']}")
    print("─" * 55)

    if results["explanation"]:
        print("\n  📊 Top Feature Importances (SHAP):")
        top_5 = sorted(results["explanation"].items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        for feat, val in top_5:
            print(f"     {feat:30s}  {val:+.4f}")

    print(f"\n  ✅ Pipeline completed in {elapsed}s")
    print("─" * 55)


if __name__ == "__main__":
    main()
