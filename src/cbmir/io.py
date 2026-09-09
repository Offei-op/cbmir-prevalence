"""Paths, manifests and fingerprints shared by every stage."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

QUERIES = {
    "Tuberculosis": "tb_queries.csv",
    "Pneumothorax": "pneumo_queries.csv",
    "Cardiomegaly": "cardio_queries.csv",
    "Emphysema": "emphy_queries.csv",
}
MODELS = ["CNN-PT", "CNN-Common", "CNN-NIH", "SwAV-PT", "SwAV-CLS", "SwAV-SupCon"]
TRAIN_FILES = [
    "cnn_common_train.csv",
    "cnn_common_val.csv",
    "nih_permitted_train.csv",
    "nih_permitted_val.csv",
]


def slug(model):
    if model not in MODELS:
        raise ValueError(f"Unknown model: {model}")
    return model.lower().replace("-", "_")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_manifest(path):
    frame = pd.read_csv(path, dtype={"group_id": str, "image_path": str})
    required = {"image_path", "source", "labels", "group_id"}
    if not required.issubset(frame.columns) or frame[list(required)].isna().any().any():
        raise ValueError(f"Missing manifest fields in {path}")
    if frame.empty or frame.image_path.duplicated().any():
        raise ValueError(f"Empty or repeated image paths in {path}")
    return frame


def labels_contain(series, target):
    return series.map(lambda s: target in str(s).split("|"))


def resolve_manifest(root, stored_path):
    """Support relative new manifests and relocated historical Kaggle manifests."""
    root = Path(root)
    stored = Path(stored_path)
    candidate = (
        root / "retrieval_manifests" / stored.name if stored.is_absolute() else root / stored
    )
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def master_index(root):
    root = Path(root)
    summary = pd.read_csv(root / "retrieval_manifest_index.csv")
    files = [resolve_manifest(root, f) for f in summary.file] + [root / f for f in QUERIES.values()]
    paths = sorted(set(p for f in files for p in read_manifest(f).image_path))
    return pd.DataFrame({"image_path": paths, "embedding_id": np.arange(len(paths))})


def index_hash(index):
    if index.image_path.duplicated().any() or not np.array_equal(
        index.embedding_id, np.arange(len(index))
    ):
        raise ValueError("Index must have unique paths and consecutive ordered IDs")
    return fingerprint(index.image_path.tolist())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)
