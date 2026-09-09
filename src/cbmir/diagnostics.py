"""Optional query/gallery cohort separability; not causal evidence of source bias."""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .io import QUERIES, resolve_manifest, read_manifest, index_hash, slug


def run(config):
    import torch

    root = Path(config["artifacts_dir"])
    index = pd.read_csv(root / "embedding_cache/embedding_index.csv")
    mapping = index.set_index("image_path").embedding_id
    summary = pd.read_csv(root / "retrieval_manifest_index.csv")
    records = []
    for model in config["models"]:
        payload = torch.load(
            root / "embedding_cache" / f"{slug(model)}_embeddings.pt",
            map_location="cpu",
            weights_only=True,
        )
        if payload["index_sha256"] != index_hash(index):
            raise ValueError("Cache index mismatch")
        for target in ["Tuberculosis", "Pneumothorax"]:
            q = read_manifest(root / QUERIES[target])
            positive = []
            for name in summary.loc[summary.target == target, "file"]:
                db = read_manifest(resolve_manifest(root, name))
                positive.append(db[db.target_present == 1])
            positive = pd.concat(positive).drop_duplicates("image_path")
            n = min(len(q), len(positive))
            frame = pd.concat(
                [
                    q.sample(n, random_state=42).assign(cohort=0),
                    positive.sample(n, random_state=42).assign(cohort=1),
                ],
                ignore_index=True,
            )
            ids = frame.image_path.map(mapping)
            if ids.isna().any():
                raise ValueError("Missing diagnostic image embeddings")
            x = payload["embeddings"][torch.tensor(ids.to_numpy(), dtype=torch.long)].numpy()
            y = frame.cohort.to_numpy()
            # Group NIH repeat images by patient. Elsewhere IDs remain image-level.
            groups = frame.source + ":" + frame.group_id.astype(str)
            cv = (
                StratifiedGroupKFold(5, shuffle=True, random_state=42)
                if groups.duplicated().any()
                else StratifiedKFold(5, shuffle=True, random_state=42)
            )
            splits = list(cv.split(x, y, groups) if groups.duplicated().any() else cv.split(x, y))
            if any(len(np.unique(y[test])) != 2 for _, test in splits):
                raise ValueError("A diagnostic fold lacks both cohorts")
            pipeline = make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=3000, solver="liblinear")
            )
            scores = cross_validate(
                pipeline, x, y, cv=splits, scoring={"accuracy": "accuracy", "auc": "roc_auc"}
            )
            records.append(
                {
                    "model": model,
                    "pathology": target,
                    "n_images": 2 * n,
                    "accuracy_mean": scores["test_accuracy"].mean(),
                    "auc_mean": scores["test_auc"].mean(),
                    "auc_sd": scores["test_auc"].std(ddof=1),
                }
            )
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    pd.DataFrame(records).to_csv(out / "exploratory_cohort_separability.csv", index=False)
    return str(out / "exploratory_cohort_separability.csv")
