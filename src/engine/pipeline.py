import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import os
import yaml

from src.engine.data_input      import DataInputEngine
from src.engine.meta_learning   import MetaLearningEngine
from src.engine.preprocessing   import AdaptivePreprocessingEngine
from src.engine.feature_opt     import FeatureOptimizationEngine
from src.engine.model_training  import MultiModelLearningEngine
from src.engine.hyperparameter  import HyperparameterOptimizationEngine
from src.engine.evaluation      import ModelEvaluationEngine
from src.engine.explainability  import ExplainableAIEngine
from src.engine.knowledge_memory import KnowledgeMemorySystem


class A2MLPipeline:
    """
    Master Orchestrator for the Adaptive Autonomous Machine Learning System.

    Chains all engines in sequence:
        DataInput → MetaLearning → Preprocessing → FeatureEng →
        ModelTraining → Hyperopt → Evaluation → XAI → Memory
    """

    def __init__(self, dataset_path_or_buffer):
        # ── Load Configuration ──
        self.config = self._load_config()
        
        self.data_input      = DataInputEngine()
        self.meta_learning   = MetaLearningEngine()
        self.preprocessing   = AdaptivePreprocessingEngine()
        self.feature_opt     = FeatureOptimizationEngine()
        self.multi_model     = MultiModelLearningEngine()
        self.hyperopt        = HyperparameterOptimizationEngine()
        self.evaluation      = ModelEvaluationEngine()
        self.explainability  = ExplainableAIEngine()
        self.knowledge_memory = KnowledgeMemorySystem(
            memory_file_path=self.config.get("memory", {}).get("file_path", "knowledge_memory.json")
        )

        self.results = {}
        self.df = self.data_input.load_data(dataset_path_or_buffer)

    # ─────────────────────────────────────────────────────────
    def run_pipeline(
        self,
        target_column: str = None,
        feature_opt_strategy: str = "auto",
        apply_hyperopt: bool = True,
        dataset_name: str = "User Upload",
    ) -> dict:
        """
        Executes the full ML lifecycle.

        Steps:
            1.  Data Input & Target Detection
            2.  Problem Type Detection
            3.  Dataset Analysis & Complexity Scoring
            4.  Memory Recommendation (Euclidean similarity)
            5.  Adaptive Preprocessing
            6.  Feature Engineering
            7.  Train / Test Split
            8.  Multi-Model Training + Hyperparameter Optimisation
            9.  Model Evaluation & Auto Selection
            10. Explainable AI (SHAP + PDP)
            11. Memory Update
        """

        # ── 1. Data Input & Target & Fail Safes ────────────────
        if len(self.df) < 15:
            # Fallback for very small datasets: prevent cross-validation crashes
            print("[A2ML] Warning: Very small dataset (<15 rows). Forcing hyperparameter optimization off.")
            apply_hyperopt = False
            
        self.data_input.detect_target_column(target_column)
        problem_type = self.data_input.detect_problem_type()

        # ── 2. Meta-Learning (dataset analysis + complexity) ───
        stats_report = self.meta_learning.analyze_dataset(
            self.df, self.data_input.target_col, problem_type
        )

        # ── 3. Memory Recommendation ───────────────────────────
        # Pass stats_report enriched with problem_type for similarity search
        stats_for_memory = {**stats_report, "problem_type": problem_type}
        recommended_model = self.knowledge_memory.suggest_model_based_on_history(
            stats_for_memory
        )

        # ── 4. Adaptive Preprocessing ──────────────────────────
        processed_df = self.preprocessing.fit_transform(
            self.df, self.data_input.target_col, problem_type
        )

        # ── 5. Feature / Target Split ──────────────────────────
        if problem_type == "clustering":
            X, y = processed_df, None
        else:
            target = self.data_input.target_col
            X = processed_df.drop(columns=[target])
            y = processed_df[target]

        # ── 6. Feature Engineering ─────────────────────────────
        X_opt = self.feature_opt.optimize_features(
            X, y, problem_type, strategy=feature_opt_strategy
        )

        # ── 7. Train / Test Split ──────────────────────────────
        if problem_type == "clustering":
            X_train, X_test = X_opt, X_opt
            y_train, y_test = None, None
        else:
            p_cfg = self.config.get("pipeline", {})
            X_train, X_test, y_train, y_test = train_test_split(
                X_opt, y, 
                test_size=p_cfg.get("test_size", 0.2), 
                random_state=p_cfg.get("random_state", 42)
            )

        # ── 8. Model Training + HyperOpt ──────────────────────
        raw_models = self.multi_model.get_models(problem_type)
        trained_models = {}

        for name, model in raw_models.items():
            if problem_type == "clustering":
                trained_models[name] = model
                continue

            if apply_hyperopt:
                trained_models[name] = self.hyperopt.optimize(
                    model, X_train, y_train, name, cv=3
                )
            else:
                model.fit(X_train, y_train)
                trained_models[name] = model

        # ── 9. Evaluation & Auto Selection ────────────────────
        benchmark_df   = self.evaluation.evaluate_models(
            trained_models, X_train, y_train, X_test, y_test, problem_type
        )
        best_model_name = self.evaluation.auto_select_best_model(problem_type)
        best_model      = trained_models[best_model_name]
        
        # ── 9b. Model Persistence (Save trained model) ────────
        os.makedirs("logs", exist_ok=True)
        joblib.dump(best_model, "logs/best_model.pkl")

        # ── 10. Explainable AI ────────────────────────────────
        feature_importance = {}
        pdp_data = None
        if problem_type in ("classification", "regression"):
            try:
                exp_res = self.explainability.explain_model(best_model, X_test)
                feature_importance = exp_res.get("importance", {})
                pdp_data = exp_res.get("pdp", None)
            except Exception as e:
                print(f"[XAI] Explainability failed: {e}")

        # ── 11. Memory Update ─────────────────────────────────
        metrics = (
            benchmark_df.loc[benchmark_df["Model"] == best_model_name]
            .to_dict(orient="records")[0]
        )
        self.knowledge_memory.store_run(
            dataset_name=dataset_name,
            problem_type=problem_type,
            stats_report=stats_report,
            best_model=best_model_name,
            metrics=metrics,
        )

        # ── Compile Results ───────────────────────────────────
        self.results = {
            "problem_type":           problem_type,
            "target":                 self.data_input.target_col,
            "dataset_report":         stats_report,
            "recommended_from_memory": recommended_model,
            "optimized_features":     list(X_opt.columns),
            "benchmark_results":      benchmark_df,
            "best_model_name":        best_model_name,
            "best_model_instance":    best_model,
            "metrics":                metrics,
            "explanation":            feature_importance,
            "pdp":                    pdp_data,
        }

        return self.results

    def _load_config(self) -> dict:
        config_path = "config/config.yaml"
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    return yaml.safe_load(f)
            except Exception:
                pass
        return {}
