"""
TuneMatch Airflow DAG — orchestrates the full ML pipeline.

DAG: tunematch_ml_pipeline
Schedule: Weekly (retrains on fresh data)

Tasks:
  1. ingest_data          → download/refresh Kaggle datasets
  2. preprocess_data      → clean, merge, write Parquet
  3. split_data           → temporal train/val/test split
  4. build_features       → audio features + embeddings
  5. train_content_based  → K-NN model
  6. train_collaborative  → ALS model
  7. evaluate_models      → compute metrics, log to MLflow
  8. export_artifacts     → copy best models to /models/prod/
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

default_args = {
    "owner":            "tunematch",
    "depends_on_past":  False,
    "start_date":       datetime(2024, 1, 1),
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,
}

dag = DAG(
    dag_id="tunematch_ml_pipeline",
    default_args=default_args,
    description="Weekly TuneMatch model retraining pipeline",
    schedule_interval="@weekly",
    catchup=False,
    tags=["tunematch", "ml", "recommendation"],
)


def _ingest():
    from src.data.ingest import load_config, ingest_all
    from pathlib import Path
    config = load_config()
    ingest_all(config, Path(config["paths"]["raw_data"]))


def _preprocess():
    from src.data.preprocess import run
    run()


def _split():
    from src.data.split import split
    split()


def _build_features():
    from src.features.audio_features import run as run_audio
    from src.features.embeddings import run as run_embeddings
    run_audio()
    run_embeddings()


def _train_content_based():
    from src.models.content_based import train
    train()


def _train_collaborative():
    from src.models.collaborative import train
    train()


def _evaluate():
    """Evaluate both models against the genre baseline and log to MLflow."""
    import mlflow
    import pandas as pd
    from pathlib import Path
    from src.models.content_based import ContentBasedRecommender
    from src.evaluation.metrics import evaluate_model, compare_models
    from src.evaluation.baseline import GenreBaseline
    from src.data.ingest import load_config

    config = load_config()
    proc_dir  = Path(config["paths"]["processed_data"])
    model_dir = Path(config["paths"]["models"])

    test_df  = pd.read_parquet(proc_dir / "tracks_test.parquet")
    train_df = pd.read_parquet(proc_dir / "tracks_train.parquet")

    cb_path = model_dir / "content_based.pkl"
    if not cb_path.exists():
        print("Content-based model not found — skipping evaluation")
        return

    cb_model  = ContentBasedRecommender.load(cb_path)
    baseline  = GenreBaseline().fit(train_df)

    k_values  = config["evaluation"]["k_values"]
    n_eval    = min(100, len(test_df))
    sample    = test_df.sample(n_eval, random_state=42)

    cb_recs   = []
    base_recs = []
    relevant  = []

    for _, seed_row in sample.iterrows():
        # Ground truth: same genre tracks in test set
        if "genre" in test_df.columns:
            rel = set(
                test_df[
                    (test_df["genre"] == seed_row.get("genre")) &
                    (test_df["track_id"] != seed_row.get("track_id"))
                ]["track_id"].tolist()
            )
        else:
            rel = set()
        relevant.append(rel)

        # Find seed in train set
        seed_in_train = train_df[train_df["track_id"] == seed_row.get("track_id")]
        if seed_in_train.empty:
            cb_recs.append([])
            base_recs.append([])
            continue

        seed_idx = seed_in_train.index[0]
        cb_rec  = cb_model.recommend(seed_idx, n=max(k_values))
        base_rec = baseline.recommend(seed_idx, n=max(k_values))

        cb_recs.append(cb_rec["track_id"].tolist() if "track_id" in cb_rec.columns else [])
        base_recs.append(base_rec["track_id"].tolist() if "track_id" in base_rec.columns else [])

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="evaluation"):
        for k in k_values:
            from src.evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k
            import numpy as np

            for prefix, recs in [("cb", cb_recs), ("baseline", base_recs)]:
                p = np.mean([precision_at_k(r, rel, k) for r, rel in zip(recs, relevant)])
                r = np.mean([recall_at_k(r, rel, k) for r, rel in zip(recs, relevant)])
                n = np.mean([ndcg_at_k(r, rel, k) for r, rel in zip(recs, relevant)])
                mlflow.log_metric(f"{prefix}_precision@{k}", p)
                mlflow.log_metric(f"{prefix}_recall@{k}", r)
                mlflow.log_metric(f"{prefix}_ndcg@{k}", n)
                print(f"{prefix} @{k}: P={p:.4f} R={r:.4f} NDCG={n:.4f}")


def _export_artifacts():
    """Copy best models to a /models/prod/ directory."""
    import shutil
    from pathlib import Path
    from src.data.ingest import load_config

    config    = load_config()
    model_dir = Path(config["paths"]["models"])
    prod_dir  = model_dir / "prod"
    prod_dir.mkdir(parents=True, exist_ok=True)

    for fname in ["content_based.pkl", "als_model.pkl", "hybrid_ranker.pkl"]:
        src = model_dir / fname
        dst = prod_dir / fname
        if src.exists():
            shutil.copy2(src, dst)
            print(f"Exported {fname} → {dst}")


# ── Task definitions ──────────────────────────────────────────────────────────

t1_ingest = PythonOperator(
    task_id="ingest_data",
    python_callable=_ingest,
    dag=dag,
)

t2_preprocess = PythonOperator(
    task_id="preprocess_data",
    python_callable=_preprocess,
    dag=dag,
)

t3_split = PythonOperator(
    task_id="split_data",
    python_callable=_split,
    dag=dag,
)

t4_features = PythonOperator(
    task_id="build_features",
    python_callable=_build_features,
    dag=dag,
)

t5_content = PythonOperator(
    task_id="train_content_based",
    python_callable=_train_content_based,
    dag=dag,
)

t6_collab = PythonOperator(
    task_id="train_collaborative",
    python_callable=_train_collaborative,
    dag=dag,
)

t7_eval = PythonOperator(
    task_id="evaluate_models",
    python_callable=_evaluate,
    dag=dag,
)

t8_export = PythonOperator(
    task_id="export_artifacts",
    python_callable=_export_artifacts,
    dag=dag,
)

# ── Dependencies ──────────────────────────────────────────────────────────────

t1_ingest >> t2_preprocess >> t3_split >> t4_features >> [t5_content, t6_collab] >> t7_eval >> t8_export
