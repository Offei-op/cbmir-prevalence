"""Shared ResNet models, image transformations, and training objectives."""

import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image

CLASS_TO_IDX = {"Normal": 0, "Pneumonia": 1, "COVID-19": 2}
TRANSFORM_SPEC = {
    "size": 224,
    "grayscale_replication": True,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "flip": 0.5,
    "rotation": 7,
    "crop_scale": [0.9, 1.0],
    "crop_ratio": [0.95, 1.05],
    "brightness": 0.1,
    "contrast": 0.1,
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def image_transform(training=False):
    steps = [transforms.Resize((224, 224))]
    if training:
        steps += [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(7),
            transforms.RandomResizedCrop(224, scale=(0.90, 1.0), ratio=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.10, contrast=0.10),
        ]
    return transforms.Compose(
        steps
        + [
            transforms.ToTensor(),
            transforms.Normalize(TRANSFORM_SPEC["mean"], TRANSFORM_SPEC["std"]),
        ]
    )


class XRayDataset(Dataset):
    def __init__(
        self, frame, task="embedding", vocabulary=None, training=False, two_view=False, seed=42
    ):
        self.frame = frame.reset_index(drop=True)
        self.task, self.training, self.two_view, self.seed = task, training, two_view, seed
        self.transform = image_transform(training or two_view)
        self.vocabulary = vocabulary or []
        self.targets = []
        if task == "common":
            self.targets = [CLASS_TO_IDX[s] for s in frame.labels]
        elif task == "nih":
            for s in frame.labels:
                labels = set(s.split("|"))
                unknown = labels - set(self.vocabulary)
                if unknown:
                    raise ValueError(f"Labels absent from training vocabulary: {unknown}")
                self.targets.append(torch.tensor([float(x in labels) for x in self.vocabulary]))

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        with Image.open(self.frame.iloc[idx].image_path) as original:
            im = original.convert("L")
            im = Image.merge("RGB", (im, im, im))
        if self.two_view:
            views = []
            for offset in [0, 1]:
                if self.training:
                    views.append(self.transform(im))
                else:
                    # Torchvision transforms use the CPU Torch RNG; leave CUDA RNG untouched.
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(self.seed + 2 * idx + offset)
                        views.append(self.transform(im))
            return *views, self.targets[idx]
        return self.transform(im), idx if self.task == "embedding" else self.targets[idx]


def loader(dataset, batch_size, shuffle, workers, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def backbone(initialization, config, pretrained=True):
    if initialization == "cnn":
        return resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    if not pretrained:
        return resnet50(weights=None)
    if config.get("swav_local_weights"):
        net = resnet50(weights=None)
        net.load_state_dict(
            torch.load(config["swav_local_weights"], map_location="cpu", weights_only=True)
        )
        return net
    # Matches original checkpoint source. Pin a commit/local weight file for a final release.
    return torch.hub.load(config["swav_repo"], "resnet50", trust_repo=True)


class SupConResNet(nn.Module):
    def __init__(self, base):
        super().__init__()
        base.fc = nn.Identity()
        self.backbone = base
        self.projector = nn.Sequential(
            nn.Linear(2048, 2048), nn.ReLU(inplace=True), nn.Linear(2048, 128)
        )

    def forward(self, x):
        return F.normalize(self.projector(self.backbone(x)), dim=1)


def weighted_supcon(features, labels, temperature=0.07):
    """Jaccard-weighted positive pairs; features ordered [view1 batch, view2 batch]."""
    if temperature <= 0 or features.shape[0] != 2 * len(labels) or len(labels) == 0:
        raise ValueError("Invalid contrastive batch/temperature")
    # All similarity arithmetic is explicitly outside autocast.
    with torch.autocast(device_type=features.device.type, enabled=False):
        z = F.normalize(features.float(), dim=1)
        y = torch.cat([labels.float(), labels.float()])
        intersection = y @ y.T
        size = y.sum(1, keepdim=True)
        union = size + size.T - intersection
        weights = intersection / union.clamp_min(1e-12)
        ids = torch.arange(len(labels), device=z.device).repeat(2)
        weights = torch.maximum(weights, (ids[:, None] == ids[None, :]).float())
        eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
        weights = weights.masked_fill(eye, 0)
        logits = (z @ z.T / temperature).masked_fill(eye, -torch.inf)
        logp = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        logp = logp.masked_fill(eye, 0)  # Avoid 0 * -inf without concealing other NaNs.
        return (-(weights * logp).sum(1) / weights.sum(1).clamp_min(1e-12)).mean()


def unwrap(model):
    return model.module if isinstance(model, nn.DataParallel) else model
