"""Synthetic integration: real dataset adapters, manifests, cache checks and retrieval."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from cbmir.prepare import run as prepare
from cbmir.validation import validate_artifacts


def test_prepare_evaluate_and_leakage_guard(tmp_path, monkeypatch):
    rng = np.random.default_rng(12)
    data = tmp_path / "data"

    def image(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.integers(0, 256, (40, 40), dtype=np.uint8)).save(path)

    roots = [
        (
            "tawsifurrahman/covid19-radiography-database/COVID-19_Radiography_Dataset",
            ["Normal/images", "Viral Pneumonia/images", "COVID/images"],
        ),
        (
            "jtiptj/chest-xray-pneumoniacovid19tuberculosis/train",
            ["NORMAL", "PNEUMONIA", "COVID19"],
        ),
        (
            "sachinkumar413/covid-pneumonia-normal-chest-xray-images",
            ["NORMAL", "PNEUMONIA", "COVID"],
        ),
    ]
    for root, labels in roots:
        for label in labels:
            for i in range(12):
                image(data / root / label / f"{i}.png")
    tb = data / "offeibekoe/dataset-of-tuberculosis-chest-x-rays-images"
    for i in range(12):
        image(tb / "TB Chest X-rays" / f"{i}.png")
        image(tb / "Normal Chest X-rays" / f"{i}.png")
        image(
            data / "jtiptj/chest-xray-pneumoniacovid19tuberculosis/train/TURBERCULOSIS" / f"{i}.png"
        )
    nih = data / "khanfashee/nih-chest-x-ray-14-224x224-resized"
    rows = []
    for i in range(140):
        label = (
            "No Finding"
            if i < 80
            else ["Pneumothorax", "Cardiomegaly", "Emphysema"][(i - 80) // 20]
        )
        name = f"{i}.png"
        image(nih / "images-224/images-224" / name)
        rows.append({"Image Index": name, "Patient ID": i, "Finding Labels": label})
    pd.DataFrame(rows).to_csv(nih / "Data_Entry_2017.csv", index=False)
    siim = data / "vbookshelf/pneumothorax-chest-xray-images-and-masks/siim-acr-pneumothorax"
    rows = []
    for i in range(8):
        image(siim / "images" / f"{i}.png")
        rows.append({"new_filename": f"{i}.png", "ImageId": str(i), "has_pneumo": 1})
    pd.DataFrame(rows[:4]).to_csv(siim / "stage_1_train_images.csv", index=False)
    pd.DataFrame(rows[4:]).to_csv(siim / "stage_1_test_images.csv", index=False)
    config = json.loads((Path(__file__).parents[1] / "configs/experiment.json").read_text())
    config.update(
        data_root=str(data),
        artifacts_dir=str(tmp_path / "artifacts"),
        database_size=40,
        queries_per_pathology=2,
        prevalences=[0.1, 0.05, 0.025],
        non_target_prevalence=0.025,
        database_seeds=[0, 1],
        common_source_quotas={"JTIPTJ": 5, "SachinKumar": 5, "Tawsifurrahman": 5},
    )
    prepare(config)
    root = Path(config["artifacts_dir"])
    assert validate_artifacts(root, config, check_images=True)["conditions"] == 24
    # Run real training/extraction orchestration with a tiny substitute backbone.
    # This checks all loss branches and checkpoint formats without pretrained downloads.
    torch = pytest.importorskip("torch")
    from cbmir import train, embeddings
    from cbmir.retrieval import run as evaluate

    class TinyBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.project = torch.nn.Linear(3, 2048)
            self.fc = torch.nn.Linear(2048, 1000)

        def forward(self, x):
            return self.fc(self.project(x.mean(dim=(2, 3))))

    def tiny(*args, **kwargs):
        return TinyBackbone()

    monkeypatch.setattr(train, "backbone", tiny)
    monkeypatch.setattr(embeddings, "backbone", tiny)
    config.update(
        max_epochs=1,
        min_epochs=1,
        patience=1,
        batch_size=16,
        supcon_batch_size=16,
        embedding_batch_size=32,
    )
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for model in config["models"]:
            if model not in ["CNN-PT", "SwAV-PT"]:
                train.run(config, model)
            embeddings.run(config, model)
    finally:
        torch.set_num_threads(old_threads)
    assert evaluate(config)["rows"] == 6 * 24 * 2 * 3
    # This must reject leakage even if all gallery counts still match.
    queries = pd.read_csv(root / "cardio_queries.csv")
    train = pd.read_csv(root / "nih_permitted_train.csv")
    train.loc[0, "group_id"] = queries.loc[0, "group_id"]
    train.to_csv(root / "nih_permitted_train.csv", index=False)
    with pytest.raises(ValueError, match="patient overlap"):
        validate_artifacts(root, config)
