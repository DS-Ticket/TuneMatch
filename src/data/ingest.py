"""
Data ingestion — downloads all Kaggle datasets defined in config.yaml.

Usage:
    python -m src.data.ingest                    # download all
    python -m src.data.ingest --dataset spotify_tracks
"""

import argparse
import os
import zipfile
from pathlib import Path

import kagglehub
import yaml
from loguru import logger


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def download_dataset(slug: str, dest_dir: Path) -> Path:
    """Download a Kaggle dataset by slug, unzip into dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading: {slug}")

    try:
        # kagglehub caches downloads and returns local path
        path = kagglehub.dataset_download(slug)
        cached = Path(path)

        # Copy/move files into our raw data directory
        dataset_name = slug.split("/")[-1].replace("-", "_")
        out_dir = dest_dir / dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for src_file in cached.rglob("*"):
            if src_file.is_file():
                dst = out_dir / src_file.name
                if not dst.exists():
                    import shutil
                    shutil.copy2(src_file, dst)
                    logger.info(f"  Copied: {src_file.name} → {dst}")
                else:
                    logger.info(f"  Already exists: {src_file.name}")

        return out_dir

    except Exception as e:
        logger.error(f"Failed to download {slug}: {e}")
        raise


def ingest_all(config: dict, raw_dir: Path) -> None:
    datasets = config["datasets"]
    results = {}

    for name, ds_cfg in datasets.items():
        slug = ds_cfg["slug"]
        logger.info(f"\n{'='*60}")
        logger.info(f"Dataset: {name}")
        logger.info(f"Slug:    {slug}")
        logger.info(f"Desc:    {ds_cfg.get('description', '')}")
        try:
            out = download_dataset(slug, raw_dir)
            results[name] = {"status": "ok", "path": str(out)}
        except Exception as e:
            results[name] = {"status": "error", "error": str(e)}

    logger.info("\n\nIngestion Summary:")
    for name, r in results.items():
        status = "✓" if r["status"] == "ok" else "✗"
        logger.info(f"  {status} {name}: {r.get('path', r.get('error'))}")


def main():
    parser = argparse.ArgumentParser(description="TuneMatch data ingestion")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Specific dataset key from config (default: all)")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    raw_dir = Path(config["paths"]["raw_data"])

    if args.dataset:
        if args.dataset not in config["datasets"]:
            logger.error(f"Unknown dataset: {args.dataset}")
            logger.info(f"Available: {list(config['datasets'].keys())}")
            return
        ds_cfg = config["datasets"][args.dataset]
        download_dataset(ds_cfg["slug"], raw_dir)
    else:
        ingest_all(config, raw_dir)


if __name__ == "__main__":
    main()
