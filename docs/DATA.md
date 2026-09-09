# Data and artifact contracts

## Required input collections

Paths are relative to `data_root`. These are the locations encoded in the uploaded notebooks, not independently verified dataset provenance. Obtain the data from the named sources and check their licenses and original publications. Kaggle reuploads are not necessarily independent clinical cohorts.

| Folder | Files used | Role |
|---|---|---|
| `khanfashee/nih-chest-x-ray-14-224x224-resized` | `Data_Entry_2017.csv`; `images-224/images-224/` | NIH fine-tuning, backgrounds, rare retrieval; Cardiomegaly/Emphysema queries |
| `vbookshelf/pneumothorax-chest-xray-images-and-masks/siim-acr-pneumothorax` | `stage_1_train_images.csv`, `stage_1_test_images.csv`, converted images; mask directories excluded | Pneumothorax queries |
| `jtiptj/chest-xray-pneumoniacovid19tuberculosis` | `train`, `val`, `test` folders; labels NORMAL, PNEUMONIA, COVID19, TURBERCULOSIS | Common pool; TB queries |
| `sachinkumar413/covid-pneumonia-normal-chest-xray-images` | NORMAL, PNEUMONIA, COVID folders | Common pool |
| `tawsifurrahman/covid19-radiography-database/COVID-19_Radiography_Dataset` | Normal/images, Viral Pneumonia/images, COVID/images | Common pool |
| `offeibekoe/dataset-of-tuberculosis-chest-x-rays-images` | TB Chest X-rays and Normal Chest X-rays folders, discovered recursively | TB gallery positives and extra normal background |

The TB input folder is the user's uploaded collection; its original clinical provenance is not established by the filename. Cite the actual originating collection in the paper. SIIM CSV fields expected: `new_filename`, `ImageId`, `has_pneumo`. NIH fields expected: `Image Index`, `Patient ID`, `Finding Labels`.

The standalone exploratory notebook 00 produces a different metadata schema. It is optional and not an input to preparation.

## Canonical manifests

Every active manifest contains `image_path`, `dataset`, `source`, `group_id`, and pipe-separated `labels`. `group_id` is a patient ID for NIH; it is an image identifier where patient metadata is unavailable. Read NIH IDs as strings consistently.

Required top-level CSVs in `artifacts_dir`:

- `cnn_common_train.csv`, `cnn_common_val.csv`;
- `nih_permitted_train.csv`, `nih_permitted_val.csv`;
- `tb_queries.csv`, `pneumo_queries.csv`, `cardio_queries.csv`, `emphy_queries.csv`;
- `retrieval_manifest_index.csv`.

The preparation stage additionally saves `nih_retrieval_background.csv`. Gallery manifests live under `retrieval_manifests/`. They add `design_role`, `target_class`, `target_present`, `prevalence_condition`, and `seed`. The gallery index uses `target`, `prevalence`, `seed`, `n_images`, `n_target`, and `file`.

New gallery-index file references are relative to `artifacts_dir`. Historical absolute gallery paths can be relocated by basename under `retrieval_manifests/`. Image paths inside CSVs must still resolve to actual source images for hashing, training or extraction. Evaluation from cached features needs image identities but does not reopen image pixels.

## Checkpoints and caches

New checkpoints contain model weights plus the model name, label vocabulary, training/validation manifest checksums, configuration, transform specification, runtime information, selected epoch and validation loss. Supervised checkpoint key is `state_dict`; SupCon keys are `backbone` and `projector`.

Historical checkpoints accepted by the extraction stage:

- plain ResNet classifier `state_dict` with `fc.weight`;
- SupCon dictionaries with `backbone` and `projector`.

New caches contain `embeddings`, `index_sha256`, `model` and provenance metadata. `embedding_index.csv` maps each unique path to a consecutive tensor row. The path ordering is part of the fingerprint. The importer accepts historical plain tensors only together with the original paired index.

Changing actual file bytes at an unchanged image path is not detected by the path-index fingerprint alone. Keep the content-hash inventory and freeze source files. Do not treat a cache hash as a substitute for recording the images and checkpoint that produced it.

## Result contract

Query-level result keys: model, pathology, prevalence, database_seed, query_id, K. Each row also contains query_path, precision and ndcg. New evaluation retains query_group_id and query_source. Historical nine-column result CSVs are accepted by analysis, which checks complete query/model/condition pairing.

With default settings:

- 4 targets × 4 prevalences × 5 database seeds = 80 galleries;
- 6 models × 80 galleries × 200 queries × 3 depths = 288,000 query-level rows;
- aggregation within gallery/query sets gives 1,440 model/gallery/depth rows.

A declared `training_seed` is a single run setting. Keep separate directories for additional fitted models. Do not combine them as extra database-seed rows.
