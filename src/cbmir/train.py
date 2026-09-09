"""One model per invocation; all trainable layers updated, validation-loss selection."""

from pathlib import Path
import platform
import pandas as pd
import torch
from torch import nn
from .io import read_manifest, sha256_file, slug, write_json
from .learning import (
    seed_everything,
    XRayDataset,
    loader,
    backbone,
    SupConResNet,
    weighted_supcon,
    unwrap,
    TRANSFORM_SPEC,
)


def run(config, model_name):
    if model_name not in ["CNN-Common", "CNN-NIH", "SwAV-CLS", "SwAV-SupCon"]:
        raise ValueError("Pretrained baselines need embed, not train")
    root = Path(config["artifacts_dir"])
    out = root / "checkpoints"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{slug(model_name)}_best.pt"
    if path.exists():
        raise FileExistsError(f"{path}: use a new artifacts directory for another training run")
    seed = config["training_seed"]
    seed_everything(seed)
    common = model_name == "CNN-Common"
    contrastive = model_name == "SwAV-SupCon"
    prefix = "cnn_common" if common else "nih_permitted"
    train_file, val_file = root / f"{prefix}_train.csv", root / f"{prefix}_val.csv"
    train_frame, val_frame = read_manifest(train_file), read_manifest(val_file)
    vocabulary = [] if common else sorted({x for s in train_frame.labels for x in s.split("|")})
    for frame in [train_frame, val_frame]:
        from .io import QUERIES, labels_contain

        if any(labels_contain(frame.labels, p).any() for p in QUERIES):
            raise ValueError("Rare target found in training/validation labels")
    task = "common" if common else "nih"
    train_ds = XRayDataset(train_frame, task, vocabulary, True, contrastive, seed)
    val_ds = XRayDataset(val_frame, task, vocabulary, False, contrastive, seed + 100000)
    batch = config["supcon_batch_size"] if contrastive else config["batch_size"]
    train_loader = loader(train_ds, batch, True, config["num_workers"], seed)
    val_loader = loader(val_ds, batch, False, config["num_workers"], seed)
    base = backbone("cnn" if model_name.startswith("CNN") else "swav", config)
    torch.manual_seed(seed)
    if contrastive:
        net = SupConResNet(base)
    else:
        base.fc = nn.Linear(2048, 3 if common else len(vocabulary))
        net = base
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    amp = device.type == "cuda"
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["max_epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    criterion = nn.CrossEntropyLoss() if common else nn.BCEWithLogitsLoss()
    metadata = {
        "model": model_name,
        "config": config,
        "labels": vocabulary,
        "train_manifest_sha256": sha256_file(train_file),
        "val_manifest_sha256": sha256_file(val_file),
        "transform": TRANSFORM_SPEC,
        "torch": str(torch.__version__),
        "python": platform.python_version(),
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
    write_json(out / f"{slug(model_name)}_run.json", metadata)

    def epoch(data, training):
        net.train(training)
        total, count = 0.0, 0
        with torch.set_grad_enabled(training):
            for batch_data in data:
                if contrastive:
                    v1, v2, y = batch_data
                    x = torch.cat([v1, v2]).to(device)
                else:
                    x, y = batch_data
                    x = x.to(device)
                y = y.to(device)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp):
                    features = net(x)
                    loss = None if contrastive else criterion(features, y)
                if contrastive:
                    loss = weighted_supcon(features, y, config["temperature"])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training/validation loss")
                if training:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                total += loss.item() * len(y)
                count += len(y)
        return total / count

    best, waiting, history = float("inf"), 0, []
    for ep in range(1, config["max_epochs"] + 1):
        tr, va = epoch(train_loader, True), epoch(val_loader, False)
        history.append(
            {"epoch": ep, "train_loss": tr, "val_loss": va, "lr": optimizer.param_groups[0]["lr"]}
        )
        pd.DataFrame(history).to_csv(out / f"{slug(model_name)}_history.csv", index=False)
        print(f"{model_name}: epoch {ep}, train={tr:.6f}, val={va:.6f}", flush=True)
        if va < best:
            best, waiting = va, 0
            raw = unwrap(net)
            state = (
                {"backbone": raw.backbone.state_dict(), "projector": raw.projector.state_dict()}
                if contrastive
                else {"state_dict": raw.state_dict()}
            )
            state.update(metadata=metadata, epoch=ep, val_loss=va)
            temp = path.with_suffix(".tmp")
            torch.save(state, temp)
            temp.replace(path)
        else:
            waiting += 1
        scheduler.step()
        if ep >= config["min_epochs"] and waiting >= config["patience"]:
            break
    return str(path)
