"""Exact cosine ranking and query-level Precision/nDCG. No hit-rate endpoint."""

from pathlib import Path
import numpy as np
import pandas as pd
from .io import QUERIES, read_manifest, resolve_manifest, index_hash, slug, sha256_file, write_json


def query_metrics(indices, relevance, k):
    indices, relevance = np.asarray(indices), np.asarray(relevance)
    if indices.ndim != 2 or not 0 < k <= indices.shape[1] or k > len(relevance):
        raise ValueError("Invalid retrieval depth")
    if not np.isin(relevance, [0, 1]).all():
        raise ValueError("Relevance must be binary")
    if (
        not np.issubdtype(indices.dtype, np.integer)
        or (indices < 0).any()
        or (indices >= len(relevance)).any()
    ):
        raise ValueError("Invalid gallery indices")
    rel = relevance[indices[:, :k]]
    discounts = 1 / np.log2(np.arange(2, k + 2))
    ideal = discounts[: min(k, int(relevance.sum()))].sum()
    # Explicit nDCG=0 convention for an empty relevant set; main protocol always R>=100.
    ndcg = (rel * discounts).sum(1) / ideal if ideal > 0 else np.zeros(len(indices))
    return rel.mean(1), ndcg


def retrieve_topk(queries, gallery, k, chunk_size=64):
    """Exact ranking; equal similarities broken by ascending manifest row index."""
    import torch

    if not 0 < k <= len(gallery) or chunk_size <= 0:
        raise ValueError("Invalid k/chunk size")
    if queries.ndim != 2 or gallery.ndim != 2 or queries.shape[1] != gallery.shape[1]:
        raise ValueError("Incompatible embedding dimensions")
    rows = []
    with torch.inference_mode():
        for q in queries.split(chunk_size):
            similarity = q @ gallery.T
            rows.append(torch.argsort(similarity, descending=True, stable=True)[:, :k].cpu())
    return torch.cat(rows).numpy()


def run(config):
    import torch
    from .embeddings import check_tensor
    from .validation import validate_artifacts

    root = Path(config["artifacts_dir"])
    validate_artifacts(root, config)
    cache_dir = root / "embedding_cache"
    index = pd.read_csv(cache_dir / "embedding_index.csv")
    mapping = index.set_index("image_path").embedding_id
    fingerprint = index_hash(index)
    summary = pd.read_csv(root / "retrieval_manifest_index.csv")
    queries = {t: read_manifest(root / f) for t, f in QUERIES.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    provenance = {}
    for model in config["models"]:
        path = cache_dir / f"{slug(model)}_embeddings.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (
            not isinstance(payload, dict)
            or payload.get("index_sha256") != fingerprint
            or payload.get("model") != model
        ):
            raise ValueError(
                f"Unverified or mismatched cache: {path}; use import-legacy with its ORIGINAL index"
            )
        cache = payload["embeddings"]
        check_tensor(cache, len(index))
        provenance[model] = {
            "cache_sha256": sha256_file(path),
            "provenance": payload.get("provenance"),
        }

        def lookup(frame, cache=cache):
            ids = frame.image_path.map(mapping)
            if ids.isna().any():
                raise ValueError("Image absent from embedding index")
            return cache[torch.tensor(ids.to_numpy(), dtype=torch.long)].to(device)

        for row in summary.itertuples():
            db = read_manifest(resolve_manifest(root, row.file))
            q = queries[row.target]
            ranks = retrieve_topk(lookup(q), lookup(db), max(config["ks"]))
            for k in config["ks"]:
                precision, ndcg = query_metrics(ranks, db.target_present.to_numpy(), k)
                for i, query in enumerate(q.itertuples()):
                    results.append(
                        {
                            "model": model,
                            "pathology": row.target,
                            "prevalence": float(row.prevalence),
                            "database_seed": int(row.seed),
                            "query_id": i,
                            "query_path": query.image_path,
                            "query_group_id": query.group_id,
                            "query_source": query.source,
                            "K": k,
                            "precision": float(precision[i]),
                            "ndcg": float(ndcg[i]),
                        }
                    )
        print(f"Evaluated {model}", flush=True)
        del lookup, cache, payload
    result = pd.DataFrame(results)
    expected = (
        len(config["models"]) * len(config["ks"]) * sum(len(queries[t]) for t in summary.target)
    )
    if len(result) != expected:
        raise ValueError("Unexpected result count")
    destination = root / "results"
    destination.mkdir(exist_ok=True)
    result.to_csv(destination / "query_level_rare_retrieval_results.csv", index=False)
    write_json(
        destination / "evaluation_provenance.json",
        {
            "models": provenance,
            "config": config,
            "index_sha256": fingerprint,
            "manifest_index_sha256": sha256_file(root / "retrieval_manifest_index.csv"),
            "tie_policy": "stable descending similarity, then gallery manifest row",
        },
    )
    return {"rows": len(result), "path": str(destination)}
