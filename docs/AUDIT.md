# Notebook audit and experiment sufficiency

## Assessment

The notebooks contain the main components needed to study how annotation-defined relevant-image prevalence affects retrieval: fixed queries, fixed database size, controlled prevalence levels, six representations, several pathologies, repeated database samples, and query-level ranking metrics. More model branches are not necessary to make this a focused conference experiment.

The uploads are not a self-contained reproducibility bundle. They contain code and saved notebook outputs, but no underlying images, frozen CSV manifests, tensor caches, trained checkpoints, or full 288,000-row result file. The supplied notebooks' outputs document historical execution; they do not let this review independently reproduce or certify the numerical results.

**Practical conclusion:** enough experimental scope; unfinished artifact validation and reporting. Resolve the items below before treating the historical results as publication-ready. The cleaned code is an implementation to verify on the actual data, not evidence that all issues were already corrected in prior runs.

## What each uploaded notebook contributed

Cell numbers below are zero-based, matching the notebook JSON.

| Original notebook | Role and dependency | New location |
|---|---|---|
| data-preprocessing-for-cbmir.ipynb | Initial source inventory, dimensions, modes and examples; its `path/disease` CSV is not consumed by the audit | Optional notebook 00 |
| data-audit(1).ipynb | Actual dataset adapters, common deduplication, train/query splits, 80 gallery manifests | `prepare.py`, `validation.py`; notebook 01 |
| cbmir-experiment-test(1).ipynb | Main training, extraction and retrieval runner; combines too many stages | `learning.py`, `train.py`, `embeddings.py`, `retrieval.py`; notebooks 02–04 |
| experiment-2.ipynb | Continues SwAV-CLS extraction from a saved checkpoint and trains SwAV-SupCon | Same training/extraction modules; no separate continuation pipeline |
| embedsupcon.ipynb | Extracts SupCon features, copies earlier caches, repeats aggregate retrieval | Embedding stage and explicit legacy import |
| prepare-final-experiment-results.ipynb | Produces query-level results; appends exploratory cohort/similarity diagnostics | `retrieval.py`, optional `diagnostics.py` |
| analysis.ipynb | Summaries, model comparisons, bootstrap and exploratory oracle | `analysis.py`; notebook 05 |

The six model conditions include **SwAV-SupCon**, even though an older discussion had removed that branch. It is preserved because it is in the actual implementation. No hit-rate endpoint was added. One historical exploratory cell calculates any-TB-in-top-10; that output is not carried into the main cleaned analysis.

## Findings that affect trust in results

| Finding | Evidence | Action and effect |
|---|---|---|
| Three conflicting-label duplicate groups | Audit cell 12 saved output reports 3; code then sorts by source/path and keeps the first representative | New default stops and exports conflicts. Review labels or explicitly exclude whole groups. Exclusion changes splits; do not silently reuse old checkpoints/results as though trained on the new split. |
| Cache/index pairing unverified across continuation files | embedsupcon cells 5 and 9 rebuild an index, then copy tensors from two older outputs; only shapes are checked | New caches embed the index hash and model identity. Historical import requires the original index from the same run and reorders by image path. Equal shapes alone never establish correct pairing. |
| Deduplication does not cover the entire image universe | Common sources hashed; TB-normal images checked against common hashes, but rare pools and cross-dataset query/gallery pairs lack a global check | Added full-universe SHA-256/pHash candidate report. Inspect candidates before deciding what to remove. Actual global absence of duplicates remains unverified without image access. |
| Folder labels do not establish absence of other pathologies | Common and external TB datasets carry limited labels; NIH does not offer TB as a target annotation in this pipeline | Describe prevalence and relevance as annotation-defined. Do not assert that every nominal negative is clinically target-free. This is a dataset limitation, not repairable by code alone. |
| Patient metadata incomplete | SIIM uses ImageId; other folder sources use source/stem IDs | NIH separation is checked at patient level. Do not describe all external partitions or bootstrap units as verified independent patients. |
| Total source composition changes with prevalence | Target pools use different sources; only the remaining background is source-balanced | Added gallery source-composition export. Results concern this constructed sampling regime; source matching/sensitivity analyses would be separate design work if pursued. |
| Only one nominal training seed | Runner declares `[42]` but never iterates; continuation training depends on previous RNG draws | New command explicitly accepts one configured seed per artifact directory. Multiple training seeds remain optional additional experiments, not existing evidence. |
| Minimum positive count is 100 | 20,000 × 0.005 = 100 | Study supports decreasing prevalence down to 0.5%; it does not directly address one/few relevant examples. No need to expand scope silently. |

## Code defects and cleanups

- **Hidden notebook state:** `analysis.ipynb` cell 25 uses `wide` left over from an earlier loop, before the later all-condition pivot. New exploratory tied-win analysis builds its own fully indexed table.
- **Incomplete pair handling:** old Wilcoxon paths can propagate missing values or fail for all-zero differences. New analysis rejects incomplete grids and returns p=1 for identical paired scores.
- **Multiplicity:** old code has both per-condition and global Holm values. New output names distinguish them. Bootstrap intervals remain pointwise, not simultaneous intervals. Omnibus and new prevalence contrasts have separately documented correction families.
- **Misleading winner counts:** `idxmax` assigns ties to the first model. New optional win output gives credit to every tied model and warns that totals can exceed 100%.
- **Hard-coded result totals:** replaced constants with configuration-derived counts. Default query-level count remains 288,000 and default seed/model/k aggregate count 1,440.
- **nDCG ideal denominator:** original full-k denominator is correct for the existing R≥100 grid. It is not an error in those results. New metric generalizes to R<k and documents the R=0 convention.
- **Memory release:** old `clear_gpu(*objects)` deletes local argument names, not caller-held models. New stages operate on one model per command, avoiding accumulation across the six branches.
- **Autocast:** casting contrastive inputs to float within an active autocast block does not guarantee all eligible matrix products stay FP32. New contrastive loss explicitly disables autocast, masks the diagonal, and fails on non-finite losses. This may change a retrained SupCon run.
- **Conflicting loader documentation:** original prose promises zero workers while several helpers use four; continuation SupCon uses zero. New worker count is one configuration field, default zero. GPU count and software versions are recorded.
- **Paths and copying:** removed broad copies of whole Kaggle notebook outputs. New manifest paths are relative to one artifact directory, with a documented relocation fallback for historical absolute gallery paths.
- **Image handling:** missing NIH/SIIM paths and unmapped labels fail explicitly; empty required folders fail; image handles are closed in the active training pipeline. Original optional inventory is retained separately.
- **Deterministic source enumeration:** image paths are sorted before fresh split construction. Old filesystem-order splits can differ even with the same numerical seed. Use original manifests to reproduce historical runs.
- **Checkpoint selection:** histories and best-checkpoint metadata are saved. New training starts afresh and does not claim optimizer-state resume. Existing checkpoints are refused as destinations to prevent unintended overwrite.
- **Stable ranking ties:** new evaluator uses similarity descending and gallery row order for ties. Historical `topk` may order ties differently; compare or rerun when exact reproductions matter.
- **Source diagnostics:** optional logistic classifier is named cohort separability, since query/gallery membership conflates source with other cohort differences. Known repeated NIH patient IDs stay within folds. Old similarity-margin probes and any-hit summaries are documented as exploratory rather than added to the paper pipeline.

## Before using historical results

1. Retrieve the frozen training/query/gallery manifests, model checkpoints, histories, and the original embedding index for each cache-producing notebook run. Keep their original files together.
2. Inspect the three conflicting-label duplicate groups and perform the global content-duplicate review. If data membership changes, determine which training models and caches must be regenerated. A clean refactor alone cannot repair old training leakage or mislabeled images.
3. Verify tensor/index provenance. A manifest-only change can permit extraction from an existing checkpoint if its training cohort remains valid, but new extraction must use the appropriate image universe. If original tensor/index pairing cannot be established, recompute embeddings.
4. Run `validate`, `evaluate`, and `analyze` on the actual artifact bundle. Compare imported results to old outputs before replacing paper figures. Do not paste synthetic test figures into the paper.
5. Record final dataset counts, dataset citations and release identifiers, exact checkpoint provenance, chosen metric emphasis, software versions, selected epochs, and the treatment of remaining uncertainty.

A common/seen-class retrieval experiment, additional architectures, extra training seeds, and expanded one-positive tests may be useful extensions. They are not mandatory additions to answer the current focused question. They are required only if the paper makes claims that need them.

## Verification performed during cleanup

See `VERIFICATION.md` for the exact executed checks and their limits. Tests use small synthetic fixtures, not medical study results. Original source files were read without modification; `notebook_inventory.json` records their hashes.
