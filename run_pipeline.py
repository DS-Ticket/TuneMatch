"""
TuneMatch — One-shot pipeline runner (no Airflow required for local dev).

Usage:
    python run_pipeline.py                  # full pipeline
    python run_pipeline.py --steps ingest preprocess split
    python run_pipeline.py --steps features train_cb
"""

import argparse
import sys
from loguru import logger


STEPS = {
    "ingest": {
        "fn": lambda: __import__("src.data.ingest", fromlist=["main"]).main(),
        "desc": "Download Kaggle datasets",
    },
    "preprocess": {
        "fn": lambda: __import__("src.data.preprocess", fromlist=["run"]).run(),
        "desc": "Clean + merge raw data → Parquet",
    },
    "split": {
        "fn": lambda: __import__("src.data.split", fromlist=["split"]).split(),
        "desc": "Temporal train/val/test split",
    },
    "features": {
        "fn": lambda: (
            __import__("src.features.audio_features", fromlist=["run"]).run(),
            __import__("src.features.embeddings",     fromlist=["run"]).run(),
        ),
        "desc": "Build audio feature + embedding matrices",
    },
    "train_cb": {
        "fn": lambda: __import__("src.models.content_based", fromlist=["train"]).train(),
        "desc": "Train content-based K-NN model",
    },
    "train_als": {
        "fn": lambda: __import__("src.models.collaborative", fromlist=["train"]).train(),
        "desc": "Train ALS collaborative filtering model",
    },
}

FULL_ORDER = ["ingest", "preprocess", "split", "features", "train_cb", "train_als"]


def main():
    parser = argparse.ArgumentParser(description="TuneMatch pipeline runner")
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=list(STEPS.keys()),
        default=FULL_ORDER,
        help="Pipeline steps to run (default: all)",
    )
    args = parser.parse_args()

    logger.info("TuneMatch Pipeline")
    logger.info("=" * 50)

    for step in args.steps:
        logger.info(f"\n▶ Step: {step} — {STEPS[step]['desc']}")
        try:
            STEPS[step]["fn"]()
            logger.success(f"  ✓ {step} complete")
        except Exception as e:
            logger.error(f"  ✗ {step} failed: {e}")
            sys.exit(1)

    logger.success("\nPipeline complete. Next steps:")
    logger.info("  Launch API:       python -m src.api.app")
    logger.info("  Launch UI:        streamlit run app/streamlit_app.py")
    logger.info("  Launch MLflow:    mlflow ui --port 5000")
    logger.info("  Run tests:        pytest tests/")


if __name__ == "__main__":
    main()
