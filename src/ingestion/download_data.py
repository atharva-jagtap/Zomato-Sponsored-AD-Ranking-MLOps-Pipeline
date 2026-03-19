"""
Download raw datasets from Kaggle.

Datasets:
  1. Zomato Bangalore Restaurants - real restaurant catalog
     kaggle datasets download -d himanshupoddar/zomato-bangalore-restaurants

  2. Outbrain Click Prediction - click interaction schema
     kaggle competitions download -c outbrain-click-prediction
     (Only clicks_train.csv and documents_meta.csv needed - ~2GB total)

Usage:
    python src/ingestion/download_data.py
    python src/ingestion/download_data.py --dataset zomato
    python src/ingestion/download_data.py --dataset outbrain_clicks

Requires:
    pip install kaggle
    Either:
      - ~/.kaggle/kaggle.json
      - or environment variables KAGGLE_USERNAME and KAGGLE_KEY
"""

import argparse
import os
import subprocess
import zipfile
from pathlib import Path

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "zomato": {
        "type": "dataset",
        "slug": "himanshupoddar/zomato-bangalore-restaurants",
        "files": ["zomato.csv"],
        "description": "51,717 Bangalore restaurants with cuisine, cost, rating, location",
    },
    "outbrain_clicks": {
        "type": "competition",
        "slug": "outbrain-click-prediction",
        "files": ["clicks_train.csv"],
        "description": "87M display ad click events - provides interaction schema",
    },
    "outbrain_docs": {
        "type": "competition",
        "slug": "outbrain-click-prediction",
        "files": ["documents_meta.csv"],
        "description": "Document metadata - maps to restaurant metadata pattern",
    },
}


def format_process_error(result: subprocess.CompletedProcess[str]) -> str:
    return (result.stderr or result.stdout or "Unknown Kaggle error").strip()


def download_kaggle_dataset(slug: str, dest_dir: Path) -> None:
    cmd = ["kaggle", "datasets", "download", "-d", slug, "-p", str(dest_dir), "--unzip"]
    print(f"  Downloading dataset: {slug}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Kaggle download failed:\n{format_process_error(result)}")


def download_kaggle_competition(slug: str, files: list[str], dest_dir: Path) -> None:
    for file_name in files:
        requested_names = [file_name]
        if not file_name.endswith(".zip"):
            requested_names.append(f"{file_name}.zip")

        result = None
        successful_name = None
        for requested_name in requested_names:
            cmd = [
                "kaggle",
                "competitions",
                "download",
                "-c",
                slug,
                "-f",
                requested_name,
                "-p",
                str(dest_dir),
            ]
            print(f"  Downloading competition file: {requested_name}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                successful_name = requested_name
                break

        if result.returncode != 0:
            print(f"  Warning: {file_name} failed - {format_process_error(result)}")
            continue

        zip_path = dest_dir / (successful_name if successful_name and successful_name.endswith(".zip") else f"{file_name}.zip")
        if zip_path.exists() and zipfile.is_zipfile(zip_path):
            with zipfile.ZipFile(zip_path, "r") as archive:
                archive.extractall(dest_dir)
            zip_path.unlink()


def check_kaggle_credentials() -> None:
    creds = Path.home() / ".kaggle" / "kaggle.json"
    has_env_credentials = bool(os.getenv("KAGGLE_USERNAME")) and bool(os.getenv("KAGGLE_KEY"))
    if not creds.exists() and not has_env_credentials:
        raise FileNotFoundError(
            "Kaggle credentials not found.\n"
            "Provide either ~/.kaggle/kaggle.json or environment variables KAGGLE_USERNAME and KAGGLE_KEY.\n"
            "Get your API token from: https://www.kaggle.com/settings/account\n"
            "File method: mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json"
        )


def main(dataset: str = "all") -> None:
    check_kaggle_credentials()

    targets = DATASETS if dataset == "all" else {dataset: DATASETS[dataset]}

    for name, config in targets.items():
        dest = RAW_DIR / name
        dest.mkdir(parents=True, exist_ok=True)

        expected = dest / config["files"][0]
        if expected.exists():
            print(f"[skip] {name} - already downloaded at {expected}")
            continue

        print(f"\n[download] {name}: {config['description']}")
        if config["type"] == "dataset":
            download_kaggle_dataset(config["slug"], dest)
        elif config["type"] == "competition":
            download_kaggle_competition(config["slug"], config["files"], dest)

    print("\nAll datasets ready in data/raw/")
    print("Next step: python src/validation/expectations.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="all",
        choices=["all", "zomato", "outbrain_clicks", "outbrain_docs"],
        help="Which dataset to download",
    )
    args = parser.parse_args()
    main(args.dataset)
