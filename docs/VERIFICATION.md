# Verification record

Verification date: 2026-09-08.

## Executed checks

- All seven original notebook sources inspected; saved execution outputs checked for audit counts and continuation dependencies.
- Python modules compiled; Ruff import/name/error checks passed.
- All seven cleaned notebooks passed nbformat schema validation; all Python code cells compiled and execution outputs were cleared.
- Package installed locally with `python -m pip install -e .`; CLI help loaded successfully.
- **14 tests passed** in the full synthetic suite. A targeted integration rerun after the final checkpoint/provenance guards is recorded in `verification-test-output.txt`.
- Metric tests cover sparse relevance with R<k, the no-relevance convention, invalid ranks, exact ranking ties, Holm adjustment, identical-model comparisons and known paired effects.
- Pairing tests reject missing observations and changing query identities.
- Contrastive-loss backward pass produces finite, nonzero gradients.
- Synthetic analysis writes tables and PDF/PNG plots without depending on notebook state.
- Synthetic integration creates the expected on-disk dataset layouts, reads real image files, constructs train/query/gallery manifests, verifies overlap checks, trains all four adapted branches for one epoch with a small substitute backbone, reloads checkpoints, extracts all six caches, and evaluates retrieval. An injected NIH query/training patient overlap is rejected.

## Limits

The integration test substitutes a tiny network for ResNet-50 to test orchestration without downloading pretrained weights. It uses synthetic random images, small pools and shortened training. The actual pretrained checkpoint download, full ResNet-50 training, multi-GPU DataParallel/AMP behavior, full-size runtime/memory usage, original dataset provenance, original image duplicates, historical tensor/index pairing and reported medical retrieval performance have **not** been independently verified here.

No medical experiment was rerun. Test plots are not included as study figures. The user must run the cleaned code against the actual datasets or import verifiable frozen artifacts before reporting its outputs as research results.

The initial expanded integration check exposed a missing locally installed dependency (`tqdm`); installing the declared package dependencies resolved it. A first schema check found a missing ID on the optional inventory introduction cell; that was corrected and all notebook schemas passed afterward.

## Local verification versions

- torch: 2.14.0+cpu
- torchvision: 0.29.0+cpu
- numpy: 2.3.5
- pandas: 2.2.3
- scipy: 1.17.0
- scikit-learn: 1.8.0
- matplotlib: 3.10.8
- Pillow: 12.3.0
- ImageHash: 4.3.2
- pytest: 9.1.1
- ruff: 0.16.6
- nbformat: 5.11.1
- tqdm: 4.70.0

These are the cleanup verification environment versions, not the historical Kaggle training environment. The dependency manifest intentionally does not claim to recover that historical environment.
