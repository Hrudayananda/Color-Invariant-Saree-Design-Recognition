# AIE-CASE: Color-Invariant Saree Design Recognition

A PyTorch system that identifies a saree by its **surface design**, independent of colour palette.
It supports **identification** (Top-K retrieval from a gallery) and **verification** (are two images the same design?).

> **Status:** the code has been syntax-checked and its data-preparation logic unit-tested on synthetic images.
> Training and evaluation have **not** been run on the real datasets. No metrics are claimed here.
> All numbers must come from your own run (`metrics.json`).

## Files

| File | Purpose |
|---|---|
| `saree_design_recognition.ipynb` | Kaggle notebook (upload and run) |
| `saree_design_recognition.py` | Same code as a script, cells marked with `# %%` |

## Quick start (Kaggle)

1. Create a notebook, set **Accelerator = GPU** and **Internet = On**.
2. Add data (recommended):
   - Add the Kaggle dataset `div456/indian-saree-patterns` as an input.
   - Upload the DeepLure corpus as a **private** Kaggle dataset and attach it (its name should contain `deeplure`).
   - Fallback: the notebook tries `kagglehub` for dataset 2 and `gdown` for the DeepLure Google Drive link, which can fail on permissions or quota.
3. Upload `saree_design_recognition.ipynb` and **Run All**.
4. Smoke test first: set the environment variable `QUICK_RUN=1` (2 epochs, about 800 images per source).

The DeepLure corpus is proprietary. Do not redistribute it. Resized copies are cached in `/tmp`, but the retrieval figures contain dataset images, so keep the notebook private.

## Notebook sections

1. **Data preparation:** acquisition, inspection, label inference, validation and cleaning, design IDs, splits
2. **Training:** augmentations, colour-aware sampler, model, losses, loop, ablation runs
3. **Evaluation:** verification pairs, ablation table
4. **Identification:** Top-1/5/10, MRR, retrieval visualisation
5. **Verification:** threshold, metrics, per-category results, similarity histograms
6. **Efficiency:** parameters, FLOPs, latency, gallery time, checkpoint size
7. **Saved artefacts**, then the 500-character approach note and technical defence notes

## Model

`MobileNetV3-Large (ImageNet) -> GeM pooling -> Linear(960->256) + BatchNorm -> L2-norm`, scored with cosine similarity.
One shared-weight (Siamese-style) encoder serves both identification and verification.
Alternative backbones: `efficientnet_b0` and `resnet18`, set via `CFG.backbone`.

## Colour-invariance strategy

- **Augmentations:** hue, saturation, brightness and contrast jitter, random grayscale (20%), channel permutation, channel mixing, random-resized crop, horizontal flip and small rotation. Colour is never removed globally.
- **Sampler:** each batch has P designs x K images. Hard positives are the most differently coloured colourways of a design. Hard negatives are different designs with similar colours, taken from colour-neighbour batches.
- **Losses:** supervised contrastive plus batch-hard triplet.

## How design labels are obtained

Nothing is fabricated. Per source, the notebook tries these in order and prints the method it chose:

1. A metadata CSV or TSV whose rows match image filenames and which has a design-like column
2. Folder-per-design (folders with few images each)
3. Filename colourway grouping (for example `design12_red.jpg` and `design12_blue.jpg`)
4. **Instance mode** if none of the above applies. Each image or near-duplicate cluster is its own identity, and queries are synthetic recolours, reported separately.

If a source gets the wrong method, override it with `CFG.label_mode` (`auto | metadata | folder | filename | instance`) or `CFG.folder_level` (1 = parent folder, 2 = grandparent).

## Splits and leakage control

- Exact duplicates are removed by MD5.
- Near-duplicates are clustered by 64-bit dHash (Hamming distance 4 or less).
- Whole groups, meaning design plus near-duplicate cluster, go to exactly one of train, val or test. Evaluation is therefore open-set.
- Assertions check that group, design, MD5 and uid are disjoint across splits.
- Test is split into **gallery** and **query**. Validation is built the same way and used for model selection and the verification threshold.

## Evaluation protocols

- **`real_colorway`:** queries are genuine different-colourway images of a design. This is the primary challenge metric, available only if the data has multi-colourway designs.
- **`synthetic_recolor`:** the query is a deterministic synthetic recolour of the gallery image. It is a proxy and flatters colour-augmented models, so read `real_colorway` first when present.
- **Colour buckets** (label-free): similar colour is at or below the 20th percentile of random-pair colour distance, and different colour is at or above the 50th.
- **Verification categories:** same design / same colour, same design / different colour, different design / similar colour, different design / different colour. The critical cases are the second (should match) and the third (should not).
- **Ablation:** untrained ImageNet reference, A baseline, B baseline + colour augmentation, C final model.

## Outputs (in `/kaggle/working`)

`best_model.pth`, `config.json`, `metrics.json`, `retrieval_examples.png`, `similarity_distribution.png`, `training_curves.png`, and per-experiment `ckpt_*.pth`.

## Configuration

All settings are in the `Config` dataclass: image size, batch (`P` x `K`), learning rates, epochs, embedding dimension, loss parameters, seed. Device selection is automatic (CUDA if available, otherwise CPU). Seeds are fixed for reproducibility.

## API

```python
verify_pair(model, "a.jpg", "b.jpg", thr)   # -> {"cosine": ..., "same_design": bool}
retrieve(q_emb, g_emb, k=10)                # -> (similarities, gallery indices)
```

## Troubleshooting

- **"RANDOM INIT" warning:** pretrained weights could not download. Turn Internet on.
- **DeepLure download fails:** attach it as a private Kaggle dataset instead.
- **Only synthetic results appear:** no multi-image designs were found. Check the printed label method and override if needed.
- **Out of memory:** lower `CFG.P`, `CFG.K` or `CFG.img_size`.

## Limitations

- Label quality bounds everything. In instance mode, "same design" means same image or near-duplicate.
- Hue-shift augmentation can produce colourways that never occur in real catalogues.
- Designs that differ only by colour placement cannot be separated by a colour-invariant model.
- dHash clustering may merge plain sarees into one split group. This is safe for leakage but reduces diversity.
