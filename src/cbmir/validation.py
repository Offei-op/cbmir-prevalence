"""Manifest checks. Content audit reports potential duplicates without relabelling."""

from itertools import product
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

from .io import QUERIES, TRAIN_FILES, labels_contain, read_manifest, resolve_manifest, write_json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def nih_patients(frame):
    return set(frame.loc[frame.source.str.startswith("NIH"), "group_id"])


def validate_artifacts(root, config, check_images=False):
    root = Path(root)
    summary = pd.read_csv(root / "retrieval_manifest_index.csv")
    keys = ["target", "prevalence", "seed"]
    require(not summary.duplicated(keys).any(), "Duplicate retrieval conditions")
    expected = set(product(QUERIES, config["prevalences"], config["database_seeds"]))
    require(
        set(summary[keys].itertuples(index=False, name=None)) == expected,
        "Incomplete/unexpected experiment grid",
    )
    train = {f: read_manifest(root / f) for f in TRAIN_FILES}
    queries = {p: read_manifest(root / f) for p, f in QUERIES.items()}
    all_training = pd.concat(list(train.values()), ignore_index=True)
    used_paths = set(all_training.image_path)
    used_patients = nih_patients(all_training)
    require(
        set(train[TRAIN_FILES[0]].image_path).isdisjoint(train[TRAIN_FILES[1]].image_path),
        "Common train/validation overlap",
    )
    require(
        nih_patients(train[TRAIN_FILES[2]]).isdisjoint(nih_patients(train[TRAIN_FILES[3]])),
        "NIH train/validation patient overlap",
    )
    for name, frame in train.items():
        for rare in QUERIES:
            require(not labels_contain(frame.labels, rare).any(), f"Target label in {name}: {rare}")
    query_paths = set()
    query_patients = set()
    for target, q in queries.items():
        require(len(q) == config["queries_per_pathology"], f"Wrong query count: {target}")
        require(labels_contain(q.labels, target).all(), f"Query missing target: {target}")
        require(set(q.image_path).isdisjoint(used_paths), "Query/training image overlap")
        require(nih_patients(q).isdisjoint(used_patients), "Query/training NIH patient overlap")
        if q.source.str.startswith("NIH").all():
            require(q.group_id.nunique() == len(q), "Repeated NIH query patient within pathology")
        query_paths.update(q.image_path)
        query_patients.update(nih_patients(q))
    source_rows = []
    all_paths = used_paths | query_paths
    for row in summary.itertuples():
        db = read_manifest(resolve_manifest(root, row.file))
        n = config["database_size"]
        require(len(db) == n, "Wrong gallery size")
        require(
            set(db.image_path).isdisjoint(used_paths | query_paths),
            "Gallery overlaps training/queries",
        )
        require(
            nih_patients(db).isdisjoint(used_patients | query_patients),
            "Gallery NIH patient leakage",
        )
        for rare in QUERIES:
            count = round(
                n * (row.prevalence if rare == row.target else config["non_target_prevalence"])
            )
            require(
                int(labels_contain(db.labels, rare).sum()) == count,
                f"Wrong realized prevalence: {rare}",
            )
        require(
            np.array_equal(
                db.target_present.to_numpy(), labels_contain(db.labels, row.target).astype(int)
            ),
            "Incorrect relevance annotation",
        )
        all_paths.update(db.image_path)
        for source, count in db.source.value_counts().items():
            source_rows.append(
                dict(
                    target=row.target,
                    prevalence=row.prevalence,
                    seed=row.seed,
                    source=source,
                    n=count,
                    fraction=count / n,
                )
            )
    if check_images:
        from PIL import Image

        for path in sorted(all_paths):
            with Image.open(path) as im:
                im.verify()
    pd.DataFrame(source_rows).to_csv(root / "gallery_source_composition.csv", index=False)
    result = {
        "conditions": len(summary),
        "unique_images": len(all_paths),
        "checks_passed": True,
        "image_readability_checked": check_images,
        "limitations": [
            "Patient IDs unavailable for some external sources",
            "Path/patient checks do not establish absence of content duplicates or unannotated diseases",
        ],
    }
    write_json(root / "manifest_validation.json", result)
    return result


def content_audit(root):
    """Hash the full image universe; flag cross-role and conflicting-label groups.

    Equal pHash is a review candidate, not proof that two images are identical.
    No raw image pixels are written to reports.
    """
    import imagehash
    from PIL import Image

    root = Path(root)
    frames = []
    for name in TRAIN_FILES:
        frames.append(read_manifest(root / name).assign(role=name))
    for target, name in QUERIES.items():
        frames.append(read_manifest(root / name).assign(role="query:" + target))
    index = pd.read_csv(root / "retrieval_manifest_index.csv")
    for name in index.file:
        frames.append(read_manifest(resolve_manifest(root, name)).assign(role="gallery"))
    data = pd.concat(frames, ignore_index=True).drop_duplicates(["image_path", "role"])
    hashes = []
    for path in sorted(data.image_path.unique()):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        with Image.open(path) as im:
            perceptual = str(imagehash.phash(im.convert("L")))
        hashes.append({"image_path": path, "sha256": h.hexdigest(), "phash": perceptual})
    out = data.merge(pd.DataFrame(hashes), on="image_path", validate="many_to_one")
    out.to_csv(root / "content_hash_inventory.csv", index=False)
    for key in ["sha256", "phash"]:
        candidates = out.groupby(key).filter(
            lambda g: g.image_path.nunique() > 1 or g.role.nunique() > 1
        )
        candidates.to_csv(root / f"{key}_review_candidates.csv", index=False)
    return {"images_hashed": len(hashes), "review_required": True}
