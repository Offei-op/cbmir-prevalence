# Methodology draft — CBMIR under decreasing database prevalence

**Draft status:** This section describes the protocol encoded in the seven supplied notebooks, with historical counts taken from their saved outputs. It is not certification that all historical artifacts are correct. Resolve the duplicate-label conflicts and cache/index provenance identified in `AUDIT.md` before using final counts or results. Bracketed editorial notes are for the authors and should be removed after verification. New safeguards or analyses are identified after the draft so they are not retrospectively attributed to the original experiments.

## III. Methodology

### A. Study design and image sources

We investigated how decreasing the proportion of images labelled with a target pathology affects content-based chest X-ray retrieval. Database size, query sets, and trained representations were held fixed while the labelled target prevalence was varied. Six ResNet-50 representation conditions were evaluated for Tuberculosis, Pneumothorax, Cardiomegaly, and Emphysema. These targets were excluded from the specified fine-tuning label sets. This exclusion refers to available annotations: it does not establish the absence of unannotated comorbid disease in training images.

Images were drawn from NIH ChestX-ray14, a SIIM pneumothorax image conversion, and four folder-labelled chest X-ray collections. The latter supplied Normal, Pneumonia, COVID-19, and Tuberculosis images. Segmentation masks were excluded. Viral-pneumonia folders were mapped to Pneumonia; the broader Lung Opacity category was not mapped to Pneumonia. Dataset releases, download locations, and source-specific roles are documented in the accompanying repository. [Insert original dataset citations and final cohort table; verify the provenance of the separately uploaded TB collection.]

### B. Data partitioning and duplicate handling

For the common-class branch, byte-level SHA-256 and perceptual hashes were computed across the three common-class collections. Equal perceptual hashes defined the operational duplicate groups; near-hash matches were not automatically merged. The historical audit retained 21,037 records from 26,813 input records after this deduplication. It also identified three duplicate groups with conflicting labels. [Resolve these groups and report the final policy; do not describe their resolution as completed.]

The common-class training pool was balanced to have the same source composition for each class: 569 JTIPTJ, 844 SachinKumar, and 102 Tawsifurrahman images per class, giving 4,545 images in total. An 80/20 training/validation split was stratified by source and class. The remaining common images were eligible for retrieval background. These external collections do not consistently provide patient identifiers, so image deduplication does not establish complete patient-level independence.

NIH partitions were constructed by patient. All images from any patient with a Pneumothorax, Cardiomegaly, or Emphysema annotation were excluded from NIH training and validation. Of the remaining patients, 15% were reserved for retrieval background and the rest were split 80/20 into training and validation. Saved audit outputs report 48,280 training images from 18,611 patients, 12,266 validation images from 4,653 patients, and 10,620 background images from 4,106 patients. The same NIH training and validation manifests were used for CNN-NIH, SwAV-CLS, and SwAV-SupCon.

### C. Query sets and prevalence-controlled databases

Each target had a fixed set of 200 query images. Tuberculosis queries were drawn from JTIPTJ and relevant database images from the separate TB collection. Pneumothorax queries came from SIIM, with relevant database images from NIH. Cardiomegaly and Emphysema queries came from NIH, with one image per selected patient; all images from these query patients were excluded from NIH retrieval pools. Separate pathology query sets were not required to contain mutually exclusive patients.

Every retrieval database contained 20,000 images. Target prevalence was set to 5%, 2%, 1%, or 0.5%, corresponding to 1,000, 400, 200, or 100 labelled relevant images. Each of the other three target pathologies contributed 400 images (2%). The remainder was filled with background images sampled across sources as evenly as availability allowed. NIH rare-positive pools excluded images carrying another manipulated rare label, while other secondary findings were permitted. This controlled the annotation-defined prevalence of the target labels within the constructed manifests.

Five database samples were generated using seeds 0–4 for each target/prevalence combination, producing 80 database manifests. Sampling applied to target, non-target rare, and background pools; it was not limited to background images. The query sets and each model's parameters remained unchanged. All models were evaluated on the same manifests. Total source composition was not held exactly constant as prevalence changed, and database sampling across prevalence levels was not specified as a strictly nested design.

### D. Image representations and fine-tuning

All conditions used a ResNet-50 backbone and its 2,048-dimensional pooled output for retrieval. CNN-PT used Torchvision ImageNet-supervised `IMAGENET1K_V2` weights; SwAV-PT used the ResNet-50 checkpoint supplied through the official SwAV Torch Hub entry point. SwAV learns representations by predicting cluster assignments across augmented image views. Our experiment used an existing checkpoint rather than pretraining SwAV from scratch. [Cite ResNet, SwAV, and the exact checkpoint release.]

| Condition | Initialization | Adaptation objective |
|---|---|---|
| CNN-PT | ImageNet supervised | None |
| CNN-Common | CNN-PT | Three-class cross-entropy |
| CNN-NIH | CNN-PT | NIH multi-label binary cross-entropy |
| SwAV-PT | SwAV | None |
| SwAV-CLS | SwAV-PT | Same NIH objective as CNN-NIH |
| SwAV-SupCon | SwAV-PT | Jaccard-weighted multi-label supervised contrastive loss |

Images were converted to grayscale, replicated into three channels, and resized to 224 × 224 pixels. Inputs were normalized using channel means (0.485, 0.456, 0.406) and standard deviations (0.229, 0.224, 0.225). Training augmentation comprised horizontal flipping with probability 0.5, rotation up to ±7°, random resized cropping with scale 0.90–1.00 and aspect ratio 0.95–1.05, and brightness/contrast jitter of 0.10. Evaluation used resizing and normalization only. This was the study's shared preprocessing, rather than each pretrained checkpoint's own default inference transform.

All backbone layers were fine-tuned using AdamW with learning rate 10⁻⁴, weight decay 10⁻⁴, and cosine scheduling over a maximum of 50 epochs. Classification batch size was 64. Training stopped after at least 10 epochs when validation loss had not improved for seven epochs; the checkpoint with lowest validation loss was selected. NIH labels were derived from the training manifest, including No Finding. No retrieval metric was used for checkpoint selection.

For SwAV-SupCon, each batch contained 64 source images with two augmented views each. A 2,048–2,048–128 projection head with a ReLU hidden activation was used during training. Different-image positive weights equalled the Jaccard overlap of their label sets, and cross-view pairs from the same image received weight one. With temperature τ = 0.07 and self-comparisons excluded, the per-anchor loss was

\[
\ell_i=-\frac{1}{\sum_{j\ne i}w_{ij}}\sum_{j\ne i}w_{ij}
\log\frac{\exp(z_i^\top z_j/\tau)}{\sum_{a\ne i}\exp(z_i^\top z_a/\tau)}.
\]

Validation used fixed augmented views and batch ordering. The projection head was discarded for retrieval. This weighted multi-label objective is an adaptation of supervised contrastive learning; it is not the original SwAV pretraining objective. [Cite the supervised contrastive paper; make no novelty claim for the weighted variant without checking prior work.]

### E. Retrieval and evaluation

Backbone embeddings were L2-normalized. Database images were ranked by descending cosine similarity to each query using exact similarity computation. Relevance was binary: an image was relevant when its recorded labels contained the query's designated target pathology; other findings did not receive partial credit.

Precision and normalized discounted cumulative gain were evaluated at k ∈ {5, 10, 20}, with Precision@10 used for the main analysis:

\[
P@k=\frac{1}{k}\sum_{r=1}^{k}y_r,\qquad
\mathrm{nDCG}@k=\frac{\sum_{r=1}^{k}y_r/\log_2(r+1)}
{\sum_{r=1}^{\min(k,R)}1/\log_2(r+1)},
\]

where R is the number of labelled relevant database images. Here R ≥ 100, so the ideal ranking contains relevant images at every evaluated position. Hit rate was not a study endpoint. Random ranking has expected Precision@k equal to the labelled database prevalence; precision divided by prevalence was also reported descriptively as enrichment over random.

### F. Aggregation and statistical comparisons

For each model, pathology, prevalence, and k, query-level scores were averaged within each database sample and then summarized by the mean and sample standard deviation across the five samples. This standard deviation represents sensitivity to database sampling, not training variability or a confidence interval.

For paired inference at k = 10, each query's precision was first averaged across database samples. Models were compared within each pathology/prevalence condition using a Friedman test, with Kendall's W as an effect-size summary. Six model contrasts were examined using two-sided paired Wilcoxon signed-rank tests: CNN-PT versus SwAV-PT; CNN-PT versus CNN-Common; CNN-NIH versus SwAV-CLS; SwAV-PT versus SwAV-CLS; SwAV-CLS versus SwAV-SupCon; and SwAV-PT versus SwAV-SupCon. Holm-adjusted p-values were calculated across all 96 model contrasts, with within-condition adjustments retained as supplementary output. Paired mean differences were accompanied by pointwise 95% percentile bootstrap intervals from 10,000 query resamples.

These analyses are conditional on the fitted model instances and sampled galleries. Repeated use of shared galleries and incomplete external patient metadata limit independence assumptions. Five database seeds were not treated as five independent training runs. Comparisons concern the selected checkpoint and adaptation conditions, rather than establishing a universal causal advantage of a pretraining paradigm.

## Author notes before submission

- **Do not retrospectively claim new fixes were used.** The cleaned implementation sorts source paths, stops on conflicting duplicate labels, checks all manifest overlaps, binds caches to indices, explicitly disables autocast for contrastive similarity arithmetic, and introduces a stable tie rule. These can change regenerated results.
- **Historical training seeds:** the original runner declares `[42]` but does not loop over it. Continuation notebooks reuse mutable RNG state. Do not claim multiple training seeds or exactly reproducible historical initialization without run records.
- **Added analysis:** cleaned `analysis.py` compares each lower prevalence with 5% within model/pathology using paired query differences and a separate global Holm family (72 contrasts for the default grid). This is a newly added analysis, not a preregistered or historically completed one. If retained after running it, add one sentence to Section F describing it.
- **No zero-shot overclaim:** target labels were excluded during study fine-tuning, but other collections are incompletely annotated for those findings. “Held out from fine-tuning labels” is safer than claiming every model has never encountered the pathology.
- **Hardware:** original notebooks specify Kaggle T4 × 2 with DataParallel and AMP. Insert actual device/runtime versions from the runs used in the final results; the cleanup tests used CPU and synthetic images.
- **Source bias:** background balancing does not balance total gallery sources, which also include source-specific target/distractor pools. Source composition tables are now saved for inspection.
- **Dataset counts:** update all numbers if conflicting duplicates are excluded or source enumeration changes. The old sampled splits cannot be reconstructed solely from the saved count tables.
- **Scope:** this tests 100–1,000 relevant images among 20,000. It does not answer performance with only one or a handful of relevant images.

## Verified method references and implementation sources

- [Torchvision ResNet-50 weights and preprocessing documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html).
- [Official SwAV Torch Hub implementation](https://github.com/facebookresearch/swav/blob/main/hubconf.py).
- Caron et al., [Unsupervised Learning of Visual Features by Contrasting Cluster Assignments](https://arxiv.org/abs/2006.09882).
- Khosla et al., [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362).

Convert these and the verified dataset sources to the conference's numbered reference style in the final manuscript. This draft intentionally does not fabricate dataset provenance or completed audit outcomes.
