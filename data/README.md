# RevuLens Datasets

This directory stores the datasets used by RevuLens. Raw data files and processed CSVs/NPY files are gitignored.

---

## 1. Salminen et al. (2022) — Fake Reviews Dataset (Training & Evaluation)
- **Source**: Kaggle mirror by `mexwell` / original on OSF:
  - Kaggle: https://www.kaggle.com/datasets/salminen/fake-reviews
- **Description**: ~40,432 English Amazon reviews with a 50/50 balance between original (`OR`, presumed genuine) and computer-generated (`CG`, GPT-2 generated).
- **Columns**: `category`, `rating`, `label` (`OR` / `CG`), `text_`
- **License**: CC BY 4.0 (Attribution required: Salminen et al., 2022)
- **Local Path**: Place `fake reviews dataset.csv` into `data/raw/`.

---

## 2. FiReCS (Cosme & De Leon, 2024) — Localized Taglish Reference (Exploratory Check)
- **Source**: Hugging Face `ccosme/FiReCS`
  - URL: https://huggingface.co/datasets/ccosme/FiReCS
- **Description**: 10,487 Filipino-English (Taglish) reviews from Shopee Philippines and Google Maps with sentiment labels only (train 7,340 / test 3,147).
- **Rules (Strictly enforced per GEMINI.md)**:
  - FiReCS has **no authenticity labels**.
  - **FiReCS is NEVER used for training and NEVER labeled Genuine or Deceptive.**
  - It serves as an **external exploratory check only** to measure the share of Taglish reviews flagged Deceptive by the model.
  - **Never report accuracy, precision, recall, or F1 for FiReCS.**
- **Local Path**: Place `FiReCS_train_set.csv` and `FiReCS_test_set.csv` into `data/raw/`.

---

## Processing
Run the preprocessing script to clean, deduplicate, and split the data:
```bash
python notebooks/preprocess.py
```
This generates:
- `data/processed/train.csv` (80% stratified split)
- `data/processed/val.csv` (10% stratified split)
- `data/processed/test.csv` (10% stratified split)
- `data/processed/firecs_exploratory.csv` (deduplicated Taglish reference text)
- `data/processed/metadata.json` (split statistics and class distributions)
