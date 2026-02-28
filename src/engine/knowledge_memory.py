import json
import os
import numpy as np
from datetime import datetime


class KnowledgeMemorySystem:
    """
    Stores dataset characteristics and best model decisions from every run.
    Uses Euclidean distance on dataset metadata vectors to recommend
    Models for similar future datasets.

    Memory vector: [num_rows, num_features, missing_ratio, raw_complexity, correlation, entropy]
    """

    def __init__(self, memory_file_path: str = "knowledge_memory.json"):
        self.memory_file_path = memory_file_path
        self._ensure_file_exists()

    # ─────────────────────────────────────────────
    def _ensure_file_exists(self):
        if not os.path.exists(self.memory_file_path):
            with open(self.memory_file_path, "w") as f:
                json.dump({"runs": []}, f, indent=4)

    def get_memory(self) -> dict:
        try:
            with open(self.memory_file_path, "r") as f:
                return json.load(f)
        except Exception:
            return {"runs": []}

    # ─────────────────────────────────────────────
    def store_run(
        self,
        dataset_name: str,
        problem_type: str,
        stats_report: dict,
        best_model: str,
        metrics: dict,
    ):
        """
        Persists a pipeline run record in memory.
        Includes the metadata vector used for Euclidean similarity search.
        """
        memory = self.get_memory()

        run_record = {
            "timestamp":           datetime.now().isoformat(),
            "dataset":             dataset_name,
            "problem_type":        problem_type,
            "data_complexity":     stats_report.get("data_complexity_score", "Unknown"),
            "best_model_selected": best_model,
            "metrics":             metrics,
            # ── Metadata vector for similarity search ──
            "meta_vector": {
                "num_rows":        stats_report.get("num_rows", 0),
                "num_features":    stats_report.get("num_features", 0),
                "missing_ratio":   stats_report.get("missing_ratio", 0.0),
                "raw_complexity":  stats_report.get("raw_complexity", 0.0),
                "correlation":     stats_report.get("mean_target_correlation", 0.0),
                "entropy":         stats_report.get("mean_feature_entropy", 0.0),
            },
        }

        memory["runs"].append(run_record)

        with open(self.memory_file_path, "w") as f:
            json.dump(memory, f, indent=4)

    # ─────────────────────────────────────────────
    def suggest_model_based_on_history(self, current_stats: dict) -> str:
        """
        Recommends a model using Euclidean distance on metadata vectors.

        Distance formula:
            d = sqrt( Σ (v_current_i - v_past_i)² )

        The closest past run (same problem_type) recommends its best model.
        """
        memory = self.get_memory()
        if not memory["runs"]:
            return None

        current_problem_type = current_stats.get("problem_type")
        current_vector = np.array([
            current_stats.get("num_rows",       0),
            current_stats.get("num_features",   0),
            current_stats.get("missing_ratio",  0.0),
            current_stats.get("raw_complexity", 0.0),
            current_stats.get("mean_target_correlation", 0.0),
            current_stats.get("mean_feature_entropy", 0.0),
        ], dtype=float)

        # Filter past runs with matching problem type
        compatible_runs = [
            r for r in memory["runs"]
            if r.get("problem_type") == current_problem_type
            and "meta_vector" in r
        ]

        if not compatible_runs:
            return None

        # Normalise vectors for fair distance comparison
        all_vectors = np.array([
            [
                r["meta_vector"].get("num_rows", 0),
                r["meta_vector"].get("num_features", 0),
                r["meta_vector"].get("missing_ratio", 0.0),
                r["meta_vector"].get("raw_complexity", 0.0),
                r["meta_vector"].get("correlation", 0.0),
                r["meta_vector"].get("entropy", 0.0),
            ]
            for r in compatible_runs
        ], dtype=float)

        # Avoid division by zero; normalise each dimension
        col_max = all_vectors.max(axis=0)
        col_max[col_max == 0] = 1.0
        normed_past    = all_vectors / col_max
        normed_current = current_vector / col_max

        distances = np.linalg.norm(normed_past - normed_current, axis=1)
        closest_idx = int(np.argmin(distances))
        closest_run = compatible_runs[closest_idx]

        # Only recommend if reasonably similar (distance < 0.5 on normed scale)
        if distances[closest_idx] < 0.5:
            return closest_run["best_model_selected"]

        return None
