# CBMIR under decreasing relevant-image prevalence

**A reproducible workflow for studying how chest X-ray retrieval changes when relevant cases become less prevalent in a fixed-size database.**

The central question is: **How does decreasing relevant-image prevalence affect CBMIR performance, and how do selected pretrained and fine-tuned representations compare under those conditions?**

This repository consolidates seven research notebooks into reusable Python modules and ordered notebook entry points. It includes the experiment configuration, validation checks, statistical analysis, a code audit, and a draft methodology for the conference paper.

> **Status:** code cleanup and synthetic verification. The medical-image experiment has not been rerun as part of this cleanup. The uploaded notebooks report historical results, but the raw data, frozen manifests, checkpoints and full results CSV are not included. See [the audit](docs/AUDIT.md) before interpreting or reusing those results.

## Experiment at a glance

| Component | Default protocol |
|---|---|
| Targets | Tuberculosis, Pneumothorax, Cardiomegaly, Emphysema |
| Query sets | 200 fixed images per target |
| Database | 20,000 images |
| Target prevalence | 5%, 2%, 1%, 0.5%: 1,000, 400, 200, 100 relevant images |
| Other target classes | 2% each |
| Database sampling | Five seeds per condition; 80 manifests |
| Representations | Six ResNet-50 conditions, 2,048-dimensional embeddings |
| Retrieval | L2-normalized embeddings, exact cosine ranking |
| Metrics | Precision@k and binary nDCG@k, k=5,10,20 |
| Main analysis | Precision@10; model and prevalence comparisons |

Prevalence and relevance are defined by available annotations. “Held out” refers to fine-tuning labels, not proof that no training image contains an unannotated target finding. The lowest condition has 100 relevant images; this is not a one-shot gallery experiment.

## Model conditions

| Name | Initialization | Fine-tuning |
|---|---|---|
| CNN-PT | ImageNet supervised ResNet-50 V2 weights | None |
| CNN-Common | CNN-PT | Normal/Pneumonia/COVID-19 cross-entropy |
| CNN-NIH | CNN-PT | Permitted NIH multi-label BCE |
| SwAV-PT | Official SwAV ResNet-50 checkpoint | None |
| SwAV-CLS | SwAV-PT | Same NIH task as CNN-NIH |
| SwAV-SupCon | SwAV-PT | Weighted multi-label supervised contrastive loss |

Classification and projection heads are discarded for retrieval. SwAV-SupCon is retained from the uploaded implementation; it is distinct from SwAV's original pretraining objective. The comparisons evaluate selected checkpoints and adaptation conditions, not every supervised or self-supervised method.

## Repository guide

| Location | Purpose |
|---|---|
| `configs/experiment.json` | Paths, sampling grid, training settings and comparisons |
| `notebooks/00_optional_image_inventory.ipynb` | Optional historical image inventory |
| `notebooks/01_prepare_and_audit.ipynb` | Dataset adapters, splits and content audit |
| `notebooks/02_train_representations.ipynb` | Four trainable branches, one model per cell |
| `notebooks/03_extract_embeddings.ipynb` | All six model caches |
| `notebooks/04_evaluate_retrieval.ipynb` | Query-level metrics on shared manifests |
| `notebooks/05_results_and_statistics.ipynb` | Tables, figures and paired analysis |
| `notebooks/06_exploratory_diagnostics.ipynb` | Optional cohort-separability probes |
| `src/cbmir/` | Shared implementation used by notebooks and CLI |
| `tests/` | Metric, pairing, leakage and synthetic integration tests |
| [Methodology draft](docs/METHODOLOGY_DRAFT.md) | Paper-ready starting text with explicit unresolved items |
| [Audit and sufficiency assessment](docs/AUDIT.md) | Findings, fixes, and what remains to verify |
| [Data guide](docs/DATA.md) | Required source layout and artifact contracts |
| [Verification](docs/VERIFICATION.md) | What was actually tested |

## Installation

Use Python 3.10 or newer and run commands from the repository root. Install a compatible PyTorch/Torchvision pair for your GPU environment using [PyTorch's installation instructions](https://pytorch.org/get-started/locally/), then install this package:

```bash
python -m pip install -e '.[training,notebooks,dev]'
python -m cbmir.cli --help
```

Kaggle commonly already provides PyTorch; avoid replacing its GPU build unnecessarily. `python -m pip install -e .` installs only the non-training dependencies. Python modules load training dependencies only in stages that need them.

The dependency ranges are compatibility requirements, not a claim that historical runs used these exact versions. Save `python -m pip freeze` with each actual run. The default SwAV hub reference preserves the original `main` entry point; pin a reviewed commit or a local original-format state dictionary and retain its checksum before a final reproducibility release.

## Run the experiment

Edit `configs/experiment.json`: set `data_root` to the dataset parent directory and `artifacts_dir` to a writable run directory. Paths are relative to your current working directory unless absolute. On Kaggle, `/kaggle/working/cbmir_artifacts` is suitable for outputs. Attach the six dataset collections described in [DATA.md](docs/DATA.md).

```bash
python -m cbmir.cli prepare
python -m cbmir.cli validate --check-images
python -m cbmir.cli content-audit
```

**The original data audit found three duplicate groups with conflicting labels.** Preparation now exports `duplicate_label_conflicts.csv` and stops on these groups. Inspect and resolve the source labels, or explicitly set `duplicate_conflict_policy` to `exclude` to exclude whole conflicting groups. This changes the cohort; regenerate downstream artifacts accordingly. If a source quota becomes infeasible, do not silently reduce it—record the design change.

The content audit exports cross-role duplicate candidates for review. It does not silently remove possible duplicates, and a passing path-level validation is not proof that the content audit is complete.

Train and encode one model at a time:

```bash
python -m cbmir.cli train --model CNN-Common
python -m cbmir.cli train --model CNN-NIH
python -m cbmir.cli train --model SwAV-CLS
python -m cbmir.cli train --model SwAV-SupCon

python -m cbmir.cli embed --model CNN-PT
python -m cbmir.cli embed --model CNN-Common
python -m cbmir.cli embed --model CNN-NIH
python -m cbmir.cli embed --model SwAV-PT
python -m cbmir.cli embed --model SwAV-CLS
python -m cbmir.cli embed --model SwAV-SupCon

python -m cbmir.cli evaluate
python -m cbmir.cli analyze
```

For another configuration, place `--config` before the stage, for example `python -m cbmir.cli --config configs/my_run.json evaluate`. For another training seed, use another output directory. Commands reject existing checkpoint/cache destinations instead of overwriting them. They do not resume interrupted optimizer state.

Optional diagnostics:

```bash
python -m cbmir.cli diagnostics
```

These describe query/gallery cohort separability and do not establish causal source bias or improved clinical decisions.

## Continue from existing experiment artifacts

You do not have to retrain simply because the code was reorganized. First recover the original frozen CSV manifests and assess the audit findings. Keep image paths resolvable. Place training/query manifests and `retrieval_manifest_index.csv` directly in `artifacts_dir`, galleries under `retrieval_manifests/`, and checkpoints under `checkpoints/`.

Existing best checkpoints can be encoded directly:

```bash
python -m cbmir.cli embed --model SwAV-SupCon --checkpoint /path/to/swav_supcon_best.pt
```

For an existing tensor cache, supply the **index from the same original cache-producing run**:

```bash
python -m cbmir.cli import-legacy --model CNN-PT \
  --cache /path/to/original/cnn_pt_embeddings.pt \
  --index /path/to/original/embedding_index.csv
```

Repeat separately for each model. The importer matches image universes and reorders rows by path; it cannot establish that an incorrectly paired historical index was genuine. If that provenance is missing, recompute embeddings. Shape checks alone are insufficient.

To analyze the original full query-level CSV without GPUs:

```bash
python -m cbmir.cli analyze --results /path/to/query_level_rare_retrieval_results.csv
```

## Outputs and interpretation

| Output | Meaning |
|---|---|
| `manifest_validation.json` | Structural counts and overlap checks |
| `gallery_source_composition.csv` | Actual source fractions by condition |
| `*_review_candidates.csv` | Duplicate candidates requiring inspection |
| `checkpoints/*_history.csv` | Epoch-level train/validation loss |
| `embedding_cache/*_embeddings.pt` | Features with index/model provenance |
| `results/query_level_rare_retrieval_results.csv` | One row per model/pathology/prevalence/seed/query/k |
| `analysis/headline.csv` | Means, seed SD, random baseline and enrichment |
| `analysis/model_comparisons.csv` | Paired effects, pointwise bootstrap intervals, Holm p-values |
| `analysis/prevalence_changes.csv` | Lower-prevalence minus highest-prevalence precision |
| `analysis/*_at10.pdf` and `.png` | Exportable four-pathology performance panels |

Error bars are the **sample SD across database samples**, not confidence intervals or variation across independently trained networks. Inferential comparisons pair fixed queries after averaging database seeds. Their intervals are conditional on the fitted representations and sampled galleries; see the methodology for dependence limitations. The newly added prevalence contrasts are exploratory additions to the historical analysis, not preregistered tests.

## Tests

```bash
python -m pytest -q
python -m ruff check src tests
```

Synthetic test outputs are temporary and are not research findings. See [VERIFICATION.md](docs/VERIFICATION.md) for actual execution status and limits.

## GitHub publication

Prepared for `Offei-op/cbmir-prevalence`. No remote repository was created during this session: the connected GitHub tools expose repository access but no repository-creation action, and an authenticated GitHub CLI was unavailable.

After reviewing the package, run `bash scripts/publish_github.sh` from the repository root on a machine with authenticated `gh`. It creates a **private** repository and pushes the source, documentation and notebook entry points. If you prefer GitHub's web interface, create an empty repository and upload the package contents. Raw data, checkpoints, caches and notebook outputs are excluded.

## Attribution and data availability

This repository organizes the supplied research notebooks. No study data or pretrained weights are redistributed. Dataset and upstream model terms still apply. A code license has not been selected; choose one before advertising this as licensed open-source software. Do not cite this repository as an accepted paper or claim numerical results from the synthetic verification.

Method references: [SwAV](https://arxiv.org/abs/2006.09882), [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362), and [Torchvision ResNet-50](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html).
