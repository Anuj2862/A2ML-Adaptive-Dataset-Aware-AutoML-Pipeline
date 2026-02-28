# A²ML: Adaptive Autonomous Machine Learning System

A dataset-aware AutoML research prototype that automatically constructs machine learning pipelines for structured datasets using statistical dataset analysis, adaptive preprocessing policies, and memory-driven model selection.

---
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Overview

A²ML is a research-grade, fully autonomous machine learning pipeline. Given any structured tabular CSV dataset, it automatically:

1. Detects the column types and target variable
2. Identifies the problem type (regression / classification / clustering)
3. Computes a dataset complexity score
4. Applies adaptive preprocessing
5. Performs intelligent feature engineering
6. Trains multiple ML models with hyperparameter tuning
7. Evaluates and auto-selects the best model
8. Explains predictions using SHAP + Partial Dependence Plots
9. Stores run history to recommend models for similar future datasets

---

## 🎯 Problem Statement

Traditional ML workflows require manual preprocessing, model selection, and evaluation — demanding deep expertise and significant time. A²ML eliminates this burden by constructing optimal pipelines automatically.

---

## ✅ Scope Boundaries

**Included:**
- Tabular structured datasets (CSV format)
- Regression, Classification, Clustering problems
- Dataset-aware adaptive preprocessing
- Automated feature engineering
- Model comparison and benchmarking
- Explainable AI (SHAP + PDP)
- Knowledge-memory-based model recommendation

**Not Included:**
- Image processing / Computer Vision
- NLP / Text classification
- Deep learning pipelines
- Large-scale distributed training
- Production cloud deployment

---

## 📁 Project Structure

```
A2ML/
├── main.py                          # CLI entry point
├── app.py                           # Streamlit dashboard entry
├── requirements.txt
├── knowledge_memory.json            # Adaptive memory (auto-created)
├── A2ML_Project_Report.html         # Full project report
├── README.md
│
├── config/
│   └── config.yaml                  # System configuration
│
├── data/
│   ├── iris.csv                     # Classification (150 rows)
│   ├── housing.csv                  # Regression (20k rows)
│   ├── air_quality.csv              # Regression (400 rows)
│   ├── heart_disease.csv            # Classification (303 rows)
│   └── customer_segmentation.csv    # Clustering (500 rows)
│
├── src/
│   ├── engine/
│   │   ├── data_input.py            # Data loading + target/type detection
│   │   ├── meta_learning.py         # Dataset analysis + complexity score
│   │   ├── preprocessing.py         # Adaptive preprocessing
│   │   ├── feature_opt.py           # Feature engineering
│   │   ├── model_training.py        # Model instantiation
│   │   ├── hyperparameter.py        # GridSearchCV tuning
│   │   ├── evaluation.py            # Model benchmarking
│   │   ├── explainability.py        # SHAP + PDP
│   │   ├── knowledge_memory.py      # Persistent memory
│   │   └── pipeline.py              # Master orchestrator
│   │
│   ├── ui/
│   │   ├── dashboard.py             # Streamlit dashboard
│   │   ├── ml_knowledge.py          # Educational KB
│   │   └── style.css                # UI styling
│   │
│   └── utils/
│       └── logger.py                # Logging utility
│
├── experiments/
│   ├── exp1_pipeline_vs_baseline.py
│   ├── exp2_feature_engineering.py
│   ├── exp3_knowledge_memory.py
│   └── exp4_ablation_study.py
│
└── logs/                            # Runtime logs (auto-created)
```

---

## 🚀 Quick Start

### 1. Clone / Navigate to project
```bash
cd "/Users/anuj/Desktop/Machine Learning/A2ML"
```

### 2. Activate virtual environment
```bash
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4a. Run CLI
```bash
# Classification
python main.py --dataset data/iris.csv --target target

# Regression
python main.py --dataset data/housing.csv --target target

# Clustering (no target)
python main.py --dataset data/customer_segmentation.csv

# With specific feature strategy
python main.py --dataset data/air_quality.csv --target pm25 --strategy mutual_info

# Skip hyperparameter tuning (faster)
python main.py --dataset data/iris.csv --target target --no-hyperopt
```

### 4b. Run Web Dashboard
```bash
streamlit run app.py
# Open http://localhost:8501
```

---

## 🔬 Complexity Score Formula

```
complexity = 0.4 × (num_features / num_samples)
           + 0.3 × (missing_ratio)
           + 0.3 × (skewness_score)
```

- **Low**: score < 0.15
- **Medium**: 0.15 ≤ score < 0.35
- **High**: score ≥ 0.35

---

## 🧠 Knowledge Memory (Euclidean Similarity)

After each run, the system stores a metadata vector:
```
[rows, features, missing_ratio, complexity, correlation, entropy]
```

For new datasets, it computes Euclidean distance against all stored vectors (normalised), and recommends the best model from the closest matching run.

---

## 🧪 Running Experiments

```bash
python experiments/exp1_pipeline_vs_baseline.py        # Pipeline vs baseline
python experiments/exp2_feature_engineering.py         # Feature strategy comparison
python experiments/exp3_knowledge_memory.py            # Memory recommendation test
python experiments/exp4_ablation_study.py              # Ablation: preprocessing vs features vs full
python experiments/exp5_statistical_significance.py    # T-test for academic validity
```

---

## 📊 Supported Models

| Problem | Models |
|---------|--------|
| **Regression** | Linear Regression, Ridge, Lasso, SVR, Decision Tree, Random Forest, XGBoost |
| **Classification** | SVM, KNN, Decision Tree, Random Forest, Naive Bayes, XGBoost |
| **Clustering** | K-Means, DBSCAN |

---

## 🏗️ Research Contributions

1. Dataset-aware preprocessing framework
2. Automated model selection strategy
3. Adaptive feature engineering pipeline
4. Knowledge-memory-driven model recommendation
5. Explainable ML pipeline (SHAP + PDP)

---

## ⚠️ Known Limitations

- Supports tabular (CSV) data only
- Limited meta-learning capability
- High computational cost for very large datasets
- Quality of results depends on dataset quality
- XGBoost requires `libomp` on macOS (`brew install libomp`)

---

## 🔮 Future Extensions

- Deep learning integration (PyTorch/TensorFlow)
- NLP pipeline support
- Cloud deployment (Docker + AWS/GCP)
- AutoML ensembling
- Data drift monitoring

---

*Built for academic research — February 2026*
