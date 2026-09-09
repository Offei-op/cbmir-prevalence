"""Frozen image index, one validated cache per model, legacy cache import."""

from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from .io import master_index, index_hash, sha256_file, slug, write_json
from .learning import backbone, XRayDataset, loader, TRANSFORM_SPEC, seed_everything


def check_tensor(tensor, n):
    if not isinstance(tensor, torch.Tensor) or tensor.shape != (n, 2048):
        raise ValueError("Expected N x 2048 embedding tensor")
    if not torch.isfinite(tensor).all() or not torch.allclose(
        tensor.float().norm(dim=1), torch.ones(n), atol=1e-3
    ):
        raise ValueError("Embeddings must be finite and L2 normalized")


def run(config, model_name, checkpoint=None):
    root = Path(config["artifacts_dir"])
    out = root / "embedding_cache"
    out.mkdir(parents=True, exist_ok=True)
    index = master_index(root)
    fingerprint = index_hash(index)
    index_path = out / "embedding_index.csv"
    if index_path.exists():
        if index_hash(pd.read_csv(index_path)) != fingerprint:
            raise ValueError(
                "Existing index differs: use a fresh artifact directory; never mix caches"
            )
    else:
        index.to_csv(index_path, index=False)
    dest = out / f"{slug(model_name)}_embeddings.pt"
    if dest.exists():
        raise FileExistsError(
            f"Cache exists: {dest}. Validate/use it or deliberately move it before recomputing"
        )
    seed_everything(config["training_seed"])
    initialization = "cnn" if model_name.startswith("CNN") else "swav"
    baseline = model_name in ["CNN-PT", "SwAV-PT"]
    net = backbone(initialization, config, pretrained=baseline)
    checkpoint_sha = None
    checkpoint_metadata_verified = False
    if not baseline:
        path = (
            Path(checkpoint) if checkpoint else root / "checkpoints" / f"{slug(model_name)}_best.pt"
        )
        state = torch.load(path, map_location="cpu", weights_only=True)
        checkpoint_sha = sha256_file(path)
        metadata = state.get("metadata")
        if metadata:
            if metadata.get("model") != model_name:
                raise ValueError("Checkpoint model identity mismatch")
            prefix = "cnn_common" if model_name == "CNN-Common" else "nih_permitted"
            for split in ["train", "val"]:
                if metadata[f"{split}_manifest_sha256"] != sha256_file(
                    root / f"{prefix}_{split}.csv"
                ):
                    raise ValueError(
                        "Checkpoint training/validation manifests differ from current artifacts"
                    )
            checkpoint_metadata_verified = True
        if model_name == "SwAV-SupCon":
            net.fc = nn.Identity()
            net.load_state_dict(state["backbone"])
        else:
            weights = state.get("state_dict", state)  # historical plain state_dict accepted
            net.fc = nn.Linear(2048, weights["fc.weight"].shape[0])
            net.load_state_dict(weights)
    net.fc = nn.Identity()
    digest = hashlib.sha256()
    for name, tensor in sorted(net.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    backbone_state_sha256 = digest.hexdigest()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device).eval()
    data = loader(
        XRayDataset(index),
        config["embedding_batch_size"],
        False,
        config["num_workers"],
        config["training_seed"],
    )
    chunks = []
    with torch.inference_mode():
        for images, ids in data:
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                z = net(images.to(device))
            chunks.append(F.normalize(z.float(), dim=1).cpu())
    features = torch.cat(chunks)
    check_tensor(features, len(index))
    payload = {
        "embeddings": features,
        "index_sha256": fingerprint,
        "model": model_name,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_metadata_verified": checkpoint_metadata_verified,
        "backbone_state_sha256": backbone_state_sha256,
        "transform": TRANSFORM_SPEC,
        "config": config,
        "torch": str(torch.__version__),
        "provenance": "fresh_extraction",
    }
    temp = dest.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(dest)
    return str(dest)


def import_legacy(config, model_name, cache_path, index_path):
    """Require the index from the SAME original run as the supplied tensor.

    Shape cannot establish this provenance. The caller must retain that pairing.
    Imported artifacts record that original provenance is not independently verified.
    """
    root = Path(config["artifacts_dir"])
    out = root / "embedding_cache"
    out.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(index_path)
    index_hash(old)
    current = master_index(root)
    if set(old.image_path) != set(current.image_path):
        raise ValueError("Legacy and current image universes differ; re-extract embeddings")
    tensor = torch.load(cache_path, map_location="cpu", weights_only=True)
    check_tensor(tensor, len(old))
    order = old.set_index("image_path").loc[current.image_path, "embedding_id"].to_numpy()
    tensor = tensor[torch.from_numpy(np.asarray(order, dtype=np.int64))]
    dest = out / f"{slug(model_name)}_embeddings.pt"
    if dest.exists():
        raise FileExistsError(dest)
    shared_index = out / "embedding_index.csv"
    if shared_index.exists() and index_hash(pd.read_csv(shared_index)) != index_hash(current):
        raise ValueError("Conflicting existing index")
    torch.save(
        {
            "embeddings": tensor,
            "index_sha256": index_hash(current),
            "model": model_name,
            "provenance": "legacy_import_unverified_original_pairing",
            "legacy_cache_sha256": sha256_file(cache_path),
            "legacy_index_sha256": sha256_file(index_path),
        },
        dest,
    )
    current.to_csv(shared_index, index=False)
    write_json(
        dest.with_suffix(".json"),
        {
            "note": "Original tensor/index provenance remains the user's responsibility; no new training or extraction occurred"
        },
    )
    return str(dest)
