"""Dataset adapters and split construction, consolidated from data-audit notebook.

Defaults preserve the study grid; conflict handling and sorted enumeration are
explicit changes requiring regenerated manifests. No work runs on import.
"""


def run(config):
    def display(value):
        print(value.to_string() if hasattr(value, "to_string") else value)

    from pathlib import Path
    from collections import Counter
    import hashlib

    import numpy as np
    import pandas as pd
    from PIL import Image
    from sklearn.model_selection import train_test_split
    from tqdm.auto import tqdm

    import imagehash

    ROOT = Path(config["data_root"])
    WORKING = Path(config["artifacts_dir"])
    WORKING.mkdir(parents=True, exist_ok=True)
    if (
        (WORKING / "retrieval_manifest_index.csv").exists()
        or (WORKING / "embedding_cache").exists()
        or (WORKING / "checkpoints").exists()
    ):
        raise FileExistsError(
            "Use a fresh artifact directory for regenerated manifests; existing experiment artifacts must not be overwritten"
        )

    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    EXCLUDE_DIR_NAMES = {"mask", "masks", "png_mask", "png_masks"}

    RARE_CLASSES = ["Tuberculosis", "Pneumothorax", "Cardiomegaly", "Emphysema"]
    RARE_NIH = ["Pneumothorax", "Cardiomegaly", "Emphysema"]
    COMMON_CLASSES = ["Normal", "Pneumonia", "COVID-19"]

    N_DATABASE = config["database_size"]
    N_QUERIES = config["queries_per_pathology"]
    NON_TARGET_RARE_PREVALENCE = config["non_target_prevalence"]
    PREVALENCES = config["prevalences"]
    SEEDS = config["database_seeds"]
    RANDOM_SEED = config["split_seed"]

    tqdm.pandas()

    def is_xray(path: Path) -> bool:
        # Reject masks and unsupported image types.
        if path.suffix.lower() not in IMAGE_EXTS:
            return False
        parent_names = {part.lower() for part in path.parts}
        return not bool(parent_names & EXCLUDE_DIR_NAMES)

    def folder_manifest(folder: Path, dataset: str, source: str, label: str) -> pd.DataFrame:
        # Canonical manifest for a folder-labelled dataset.
        rows = []
        for path in sorted(folder.rglob("*")):
            if path.is_file() and is_xray(path):
                rows.append(
                    {
                        "image_path": str(path),
                        "dataset": dataset,
                        "source": source,
                        "group_id": f"{source}:{path.stem}",
                        "labels": label,
                    }
                )
        if not rows:
            raise ValueError(f"No radiographs found in {folder}")
        return pd.DataFrame(rows)

    def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
        # Byte-level hash: exact duplicate detection.
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(chunk_size):
                h.update(chunk)
        return h.hexdigest()

    def compute_phash(path: str):
        # Perceptual hash: robust to resizing/re-encoding.
        try:
            with Image.open(path) as img:
                return str(imagehash.phash(img.convert("L")))
        except Exception:
            return None

    def multilabel_counts(df: pd.DataFrame) -> pd.DataFrame:
        counts = Counter()
        for label_string in df["labels"]:
            counts.update(label_string.split("|"))
        return (
            pd.DataFrame(counts.items(), columns=["label", "count"])
            .sort_values("count", ascending=False)
            .reset_index(drop=True)
        )

    def contains_label(series: pd.Series, label: str) -> pd.Series:
        # Exact membership test for pipe-separated NIH labels.
        return series.fillna("").apply(lambda x: label in x.split("|"))

    nih_root = ROOT / "khanfashee/nih-chest-x-ray-14-224x224-resized"
    nih_csv = pd.read_csv(nih_root / "Data_Entry_2017.csv")
    nih_image_dir = nih_root / "images-224" / "images-224"

    nih = pd.DataFrame(
        {
            "image_path": nih_csv["Image Index"].apply(lambda x: str(nih_image_dir / x)),
            "dataset": "NIH_ChestXray14",
            "source": "NIH",
            "group_id": nih_csv["Patient ID"].astype(str),
            "labels": nih_csv["Finding Labels"],
        }
    )

    nih["exists"] = nih["image_path"].map(lambda p: Path(p).exists())
    print("NIH images:", len(nih))
    print("Missing files:", (~nih["exists"]).sum())
    if not nih["exists"].all():
        raise FileNotFoundError("NIH manifest references missing images")
    display(multilabel_counts(nih))

    print()
    print("Rare target counts:")
    for target in RARE_CLASSES:
        print(target, int(contains_label(nih["labels"], target).sum()))

    siim_root = (
        ROOT / "vbookshelf" / "pneumothorax-chest-xray-images-and-masks" / "siim-acr-pneumothorax"
    )
    siim_train_meta = pd.read_csv(siim_root / "stage_1_train_images.csv")
    siim_test_meta = pd.read_csv(siim_root / "stage_1_test_images.csv")

    all_siim_images = [p for p in siim_root.rglob("*") if p.is_file() and is_xray(p)]
    if len({p.name for p in all_siim_images}) != len(all_siim_images):
        raise ValueError("SIIM has ambiguous duplicate filenames")
    file_lookup = {p.name: str(p) for p in all_siim_images}

    siim_meta = pd.concat(
        [
            siim_train_meta.assign(original_split="train"),
            siim_test_meta.assign(original_split="test"),
        ],
        ignore_index=True,
    )

    siim = pd.DataFrame(
        {
            "image_path": siim_meta["new_filename"].map(file_lookup),
            "dataset": "SIIM_Pneumothorax",
            "source": "SIIM",
            "group_id": siim_meta["ImageId"].astype(str),
            "labels": siim_meta["has_pneumo"].map({1: "Pneumothorax", 0: "No Finding"}),
            "original_split": siim_meta["original_split"],
        }
    )

    print(siim["labels"].value_counts())
    if siim[["image_path", "labels"]].isna().any().any():
        raise ValueError("SIIM contains missing paths or unmapped labels")

    common_records = []

    def add_common_folder(folder: Path, dataset: str, source: str, label: str):
        for path in sorted(folder.iterdir()):
            if path.is_file() and is_xray(path):
                common_records.append(
                    {
                        "image_path": str(path),
                        "dataset": dataset,
                        "source": source,
                        "group_id": f"{source}:{path.stem}",
                        "labels": label,
                    }
                )

    # Tawsifurrahman
    root = ROOT / "tawsifurrahman" / "covid19-radiography-database" / "COVID-19_Radiography_Dataset"
    add_common_folder(root / "Normal" / "images", "COVID19_Radiography", "Tawsifurrahman", "Normal")
    add_common_folder(
        root / "Viral Pneumonia" / "images", "COVID19_Radiography", "Tawsifurrahman", "Pneumonia"
    )
    add_common_folder(
        root / "COVID" / "images", "COVID19_Radiography", "Tawsifurrahman", "COVID-19"
    )

    # JTIPTJ
    root = ROOT / "jtiptj" / "chest-xray-pneumoniacovid19tuberculosis"
    jtiptj_map = {"NORMAL": "Normal", "PNEUMONIA": "Pneumonia", "COVID19": "COVID-19"}
    for split in ["train", "val", "test"]:
        for folder_name, label in jtiptj_map.items():
            folder = root / split / folder_name
            if folder.exists():
                add_common_folder(folder, "ChestXray_Pneumonia_COVID_TB", "JTIPTJ", label)

    # SachinKumar
    root = ROOT / "sachinkumar413" / "covid-pneumonia-normal-chest-xray-images"
    sachin_map = {"NORMAL": "Normal", "PNEUMONIA": "Pneumonia", "COVID": "COVID-19"}
    for folder_name, label in sachin_map.items():
        add_common_folder(root / folder_name, "COVID_Pneumonia_Normal", "SachinKumar", label)

    common_df = pd.DataFrame(common_records)
    print("Common-case images:", len(common_df))
    print(common_df["labels"].value_counts())
    display(pd.crosstab(common_df["source"], common_df["labels"]))

    common_df["sha256"] = common_df["image_path"].progress_apply(sha256_file)
    exact_dupes = common_df[common_df.duplicated("sha256", keep=False)].sort_values("sha256")
    cross_source_exact = exact_dupes.groupby("sha256").filter(lambda g: g["source"].nunique() > 1)

    print("Exact duplicate groups:", exact_dupes["sha256"].nunique())
    print("Cross-source exact duplicate groups:", cross_source_exact["sha256"].nunique())

    common_df["phash"] = common_df["image_path"].progress_apply(compute_phash)
    print("Failed pHashes:", common_df["phash"].isna().sum())

    phash_dupes = common_df[
        common_df.duplicated("phash", keep=False) & common_df["phash"].notna()
    ].copy()
    cross_source_phash = phash_dupes.groupby("phash").filter(lambda g: g["source"].nunique() > 1)

    print("Identical-pHash groups:", phash_dupes["phash"].nunique())
    print("Cross-source identical-pHash groups:", cross_source_phash["phash"].nunique())

    if common_df["phash"].isna().any():
        raise ValueError("Unreadable common images: resolve before splitting")
    common_df["dup_group"] = common_df["phash"]
    failed = common_df["dup_group"].isna()
    common_df.loc[failed, "dup_group"] = "sha256:" + common_df.loc[failed, "sha256"].astype(str)

    label_conflicts = common_df.groupby("dup_group").filter(lambda g: g["labels"].nunique() > 1)
    print("Duplicate groups with label conflict:", label_conflicts["dup_group"].nunique())
    label_conflicts.to_csv(WORKING / "duplicate_label_conflicts.csv", index=False)
    if len(label_conflicts):
        if config.get("duplicate_conflict_policy", "error") == "exclude":
            common_df = common_df[~common_df["dup_group"].isin(label_conflicts["dup_group"])].copy()
        else:
            raise ValueError(
                "Conflicting duplicate labels: inspect duplicate_label_conflicts.csv; resolve or explicitly set duplicate_conflict_policy=exclude and regenerate all downstream artifacts"
            )

    common_unique = (
        common_df.sort_values(["source", "image_path"])
        .drop_duplicates("dup_group", keep="first")
        .reset_index(drop=True)
    )

    print()
    print("Original common pool:", len(common_df))
    print("Unique radiographs:", len(common_unique))
    print(common_unique["labels"].value_counts())
    display(pd.crosstab(common_unique["source"], common_unique["labels"]))

    SOURCE_QUOTAS = config["common_source_quotas"]
    common_parts = []

    for label in COMMON_CLASSES:
        for source, n in SOURCE_QUOTAS.items():
            subset = common_unique[
                (common_unique["labels"] == label) & (common_unique["source"] == source)
            ]
            if len(subset) < n:
                raise ValueError(f"Insufficient {source}/{label}: need {n}, found {len(subset)}")
            common_parts.append(subset.sample(n=n, random_state=RANDOM_SEED))

    cnn_common_pool = pd.concat(common_parts, ignore_index=True)
    cnn_common_pool["stratify_key"] = cnn_common_pool["source"] + "__" + cnn_common_pool["labels"]

    cnn_common_train, cnn_common_val = train_test_split(
        cnn_common_pool,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=cnn_common_pool["stratify_key"],
    )

    cnn_common_train = cnn_common_train.drop(columns="stratify_key").reset_index(drop=True)
    cnn_common_val = cnn_common_val.drop(columns="stratify_key").reset_index(drop=True)

    assert set(cnn_common_train["dup_group"]).isdisjoint(set(cnn_common_val["dup_group"]))
    assert set(cnn_common_train["image_path"]).isdisjoint(set(cnn_common_val["image_path"]))

    print("TRAIN")
    display(pd.crosstab(cnn_common_train["source"], cnn_common_train["labels"]))
    print("VALIDATION")
    display(pd.crosstab(cnn_common_val["source"], cnn_common_val["labels"]))

    cnn_common_train.to_csv(WORKING / "cnn_common_train.csv", index=False)
    cnn_common_val.to_csv(WORKING / "cnn_common_val.csv", index=False)

    # Tuberculosis
    tb_query_pool = folder_manifest(
        ROOT / "jtiptj" / "chest-xray-pneumoniacovid19tuberculosis",
        "ChestXray_Pneumonia_COVID_TB",
        "JTIPTJ_TB",
        "Tuberculosis",
    )
    tb_query_pool = tb_query_pool[
        tb_query_pool["image_path"].str.contains("TURBERCULOSIS", case=False)
    ].copy()

    tb_retrieval_pool = folder_manifest(
        ROOT / "offeibekoe" / "dataset-of-tuberculosis-chest-x-rays-images",
        "TB_Chest_Xrays",
        "TB_Chest_Xrays",
        "Tuberculosis",
    )
    tb_retrieval_pool = tb_retrieval_pool[
        tb_retrieval_pool["image_path"].str.contains("TB Chest X-rays", case=False)
    ].copy()

    tb_queries = tb_query_pool.sample(n=N_QUERIES, random_state=RANDOM_SEED).reset_index(drop=True)

    # Pneumothorax
    pneumo_queries = (
        siim[siim["labels"] == "Pneumothorax"]
        .sample(n=N_QUERIES, random_state=RANDOM_SEED)
        .reset_index(drop=True)
    )

    def make_fixed_nih_queries(df, target, n_queries=N_QUERIES, seed=RANDOM_SEED):
        rng = np.random.default_rng(seed)
        target_df = df[contains_label(df["labels"], target)].copy()
        patient_ids = target_df["group_id"].drop_duplicates().to_numpy()
        rng.shuffle(patient_ids)

        if len(patient_ids) < n_queries:
            raise ValueError(f"{target}: only {len(patient_ids)} distinct positive patients")

        rows = []
        for pid in patient_ids[:n_queries]:
            patient_imgs = target_df[target_df["group_id"] == pid]
            rows.append(
                patient_imgs.sample(
                    n=1,
                    random_state=int(rng.integers(1_000_000_000)),
                )
            )
        return pd.concat(rows, ignore_index=True)

    cardio_queries = make_fixed_nih_queries(nih, "Cardiomegaly")
    emphy_queries = make_fixed_nih_queries(nih, "Emphysema")

    nih_query_patients = set(cardio_queries["group_id"]) | set(emphy_queries["group_id"])

    print("TB queries:", len(tb_queries))
    print("Pneumothorax queries:", len(pneumo_queries))
    print("Cardiomegaly queries:", len(cardio_queries))
    print("Emphysema queries:", len(emphy_queries))
    print("Reserved NIH query patients:", len(nih_query_patients))

    rare_patient_mask = nih["labels"].apply(
        lambda x: any(target in x.split("|") for target in RARE_NIH)
    )
    rare_patient_ids = set(nih.loc[rare_patient_mask, "group_id"])

    nih_clean = nih[~nih["group_id"].isin(rare_patient_ids)].copy()
    clean_patients = nih_clean["group_id"].drop_duplicates()

    dev_patients, retrieval_bg_patients = train_test_split(
        clean_patients,
        test_size=0.15,
        random_state=RANDOM_SEED,
    )
    nih_train_patients, nih_val_patients = train_test_split(
        dev_patients,
        test_size=0.20,
        random_state=RANDOM_SEED,
    )

    nih_train = nih_clean[nih_clean["group_id"].isin(nih_train_patients)].copy()
    nih_val = nih_clean[nih_clean["group_id"].isin(nih_val_patients)].copy()
    nih_background_pool = nih_clean[nih_clean["group_id"].isin(retrieval_bg_patients)].copy()

    assert set(nih_train["group_id"]).isdisjoint(set(nih_val["group_id"]))
    assert set(nih_train["group_id"]).isdisjoint(set(nih_background_pool["group_id"]))
    assert set(nih_val["group_id"]).isdisjoint(set(nih_background_pool["group_id"]))
    assert set(nih_train["group_id"]).isdisjoint(rare_patient_ids)
    assert set(nih_val["group_id"]).isdisjoint(rare_patient_ids)

    print("NIH train:", len(nih_train), "images /", nih_train["group_id"].nunique(), "patients")
    print("NIH val:", len(nih_val), "images /", nih_val["group_id"].nunique(), "patients")
    print(
        "NIH retrieval background:",
        len(nih_background_pool),
        "images /",
        nih_background_pool["group_id"].nunique(),
        "patients",
    )

    display(multilabel_counts(nih_train))
    display(multilabel_counts(nih_val))

    nih_train.to_csv(WORKING / "nih_permitted_train.csv", index=False)
    nih_val.to_csv(WORKING / "nih_permitted_val.csv", index=False)
    nih_background_pool.to_csv(WORKING / "nih_retrieval_background.csv", index=False)

    nih_retrieval_safe = nih[~nih["group_id"].isin(nih_query_patients)].copy()

    def exclusive_rare_pool(df: pd.DataFrame, target: str) -> pd.DataFrame:
        mask = contains_label(df["labels"], target)
        for other in RARE_CLASSES:
            if other != target:
                mask &= ~contains_label(df["labels"], other)
        return df[mask].copy()

    TARGET_POOLS = {
        "Tuberculosis": tb_retrieval_pool.copy(),
        "Pneumothorax": exclusive_rare_pool(nih_retrieval_safe, "Pneumothorax"),
        "Cardiomegaly": exclusive_rare_pool(nih_retrieval_safe, "Cardiomegaly"),
        "Emphysema": exclusive_rare_pool(nih_retrieval_safe, "Emphysema"),
    }

    for target, pool in TARGET_POOLS.items():
        print(target, len(pool))
        assert len(pool) >= int(N_DATABASE * max(PREVALENCES))

    used_common_paths = set(cnn_common_train["image_path"]) | set(cnn_common_val["image_path"])
    common_retrieval_pool = common_unique[
        ~common_unique["image_path"].isin(used_common_paths)
    ].copy()

    tb_dataset_root = ROOT / "offeibekoe" / "dataset-of-tuberculosis-chest-x-rays-images"
    tb_normal_pool = folder_manifest(
        tb_dataset_root, "TB_Chest_Xrays", "TB_Chest_Xrays_Normal", "Normal"
    )
    tb_normal_pool = tb_normal_pool[
        tb_normal_pool["image_path"].str.contains("Normal Chest X-rays", case=False)
    ].copy()
    tb_normal_pool["phash"] = tb_normal_pool["image_path"].progress_apply(compute_phash)

    if tb_normal_pool["phash"].isna().any():
        raise ValueError("Unreadable TB normal image")
    tb_normal_pool = tb_normal_pool.drop_duplicates("phash").copy()
    existing_common_hashes = set(common_unique["phash"].dropna())
    tb_normal_pool = tb_normal_pool[~tb_normal_pool["phash"].isin(existing_common_hashes)].copy()

    common_retrieval_pool = pd.concat([common_retrieval_pool, tb_normal_pool], ignore_index=True)

    nih_bg = nih_background_pool.copy()
    nih_bg["source"] = "NIH_Background"
    background_pool = pd.concat([common_retrieval_pool, nih_bg], ignore_index=True)

    print("External common retrieval pool:", len(common_retrieval_pool))
    print("NIH retrieval background:", len(nih_bg))
    print("Combined background pool:", len(background_pool))
    display(background_pool["source"].value_counts())

    def balanced_source_sample(df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
        groups = {source: group.copy() for source, group in df.groupby("source")}
        remaining = n
        active = set(groups)
        quotas = {source: 0 for source in groups}

        while remaining > 0 and active:
            fair_share = int(np.ceil(remaining / len(active)))
            exhausted = []

            for source in sorted(active):
                available = len(groups[source]) - quotas[source]
                take = min(fair_share, available, remaining)
                quotas[source] += take
                remaining -= take

                if quotas[source] >= len(groups[source]):
                    exhausted.append(source)
                if remaining == 0:
                    break

            for source in exhausted:
                active.remove(source)

        if remaining > 0:
            raise ValueError(f"Not enough background images; short by {remaining}")

        samples = []
        for source, quota in quotas.items():
            if quota:
                samples.append(
                    groups[source].sample(
                        n=quota,
                        random_state=int(rng.integers(1_000_000_000)),
                    )
                )

        return pd.concat(samples, ignore_index=True)

    def build_retrieval_database(
        target: str,
        prevalence: float,
        seed: int,
        N: int = N_DATABASE,
        non_target_prevalence: float = NON_TARGET_RARE_PREVALENCE,
    ) -> pd.DataFrame:
        if target not in RARE_CLASSES:
            raise ValueError(f"Unknown target: {target}")

        rng = np.random.default_rng(seed)
        n_target = int(round(N * prevalence))
        n_other = int(round(N * non_target_prevalence))

        selected = []
        used_paths = set()

        # 1) Target pathology.
        target_pool = TARGET_POOLS[target]
        if len(target_pool) < n_target:
            raise ValueError(f"{target}: need {n_target}, only {len(target_pool)} available")

        target_sample = target_pool.sample(
            n=n_target,
            random_state=int(rng.integers(1_000_000_000)),
        ).copy()
        target_sample["design_role"] = "target"
        selected.append(target_sample)
        used_paths.update(target_sample["image_path"])

        # 2) Other rare pathologies, each fixed at 2%.
        for rare_class in RARE_CLASSES:
            if rare_class == target:
                continue

            pool = TARGET_POOLS[rare_class]
            pool = pool[~pool["image_path"].isin(used_paths)]

            if len(pool) < n_other:
                raise ValueError(f"{rare_class}: need {n_other}, only {len(pool)} available")

            sample = pool.sample(
                n=n_other,
                random_state=int(rng.integers(1_000_000_000)),
            ).copy()
            sample["design_role"] = f"distractor_{rare_class}"
            selected.append(sample)
            used_paths.update(sample["image_path"])

        # 3) Fill the remainder with source-balanced common/background images.
        n_background = N - sum(len(part) for part in selected)
        background = background_pool[~background_pool["image_path"].isin(used_paths)].copy()
        background_sample = balanced_source_sample(background, n_background, rng)
        background_sample["design_role"] = "common_background"
        selected.append(background_sample)

        # 4) Assemble and annotate.
        db = pd.concat(selected, ignore_index=True)
        db["target_class"] = target
        db["target_present"] = contains_label(db["labels"], target).astype(int)
        db["prevalence_condition"] = prevalence
        db["seed"] = seed

        db = db.sample(
            frac=1,
            random_state=int(rng.integers(1_000_000_000)),
        ).reset_index(drop=True)

        assert len(db) == N
        assert db["image_path"].nunique() == N
        assert db["target_present"].sum() == n_target
        return db

    query_output = {
        "tb_queries": tb_queries,
        "pneumo_queries": pneumo_queries,
        "cardio_queries": cardio_queries,
        "emphy_queries": emphy_queries,
    }
    for name, df in query_output.items():
        df.to_csv(WORKING / f"{name}.csv", index=False)

    OUTPUT_DIR = WORKING / "retrieval_manifests"
    OUTPUT_DIR.mkdir(exist_ok=True)
    manifest_summary = []

    for target in RARE_CLASSES:
        for prevalence in PREVALENCES:
            for seed in SEEDS:
                db = build_retrieval_database(target, prevalence, seed)
                prevalence_tag = str(prevalence).replace(".", "p")
                filename = f"{target.lower()}_prev_{prevalence_tag}_seed_{seed}.csv"
                path = OUTPUT_DIR / filename
                db.to_csv(path, index=False)

                manifest_summary.append(
                    {
                        "target": target,
                        "prevalence": prevalence,
                        "seed": seed,
                        "n_images": len(db),
                        "n_target": int(db["target_present"].sum()),
                        "file": str(path.relative_to(WORKING)),
                    }
                )

    summary_df = pd.DataFrame(manifest_summary)
    summary_df.to_csv(WORKING / "retrieval_manifest_index.csv", index=False)
    print("Total manifests:", len(summary_df))
    display(summary_df.head(10))

    for _, row in summary_df.iterrows():
        db = pd.read_csv(WORKING / row["file"])

        assert len(db) == N_DATABASE
        assert db["image_path"].nunique() == N_DATABASE

        expected_target = int(round(N_DATABASE * row["prevalence"]))
        assert int(db["target_present"].sum()) == expected_target

        for rare in RARE_CLASSES:
            count = int(contains_label(db["labels"], rare).sum())
            if rare == row["target"]:
                assert count == expected_target
            else:
                assert count == int(N_DATABASE * NON_TARGET_RARE_PREVALENCE)

    print(f"All {len(summary_df)} retrieval manifests validated successfully.")
    from .validation import validate_artifacts

    validate_artifacts(WORKING, config)
