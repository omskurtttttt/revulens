# RevuLens

*An Explainable Browser Extension for Detecting Fake E-Commerce Text Reviews Using DistilBERT-SVM and SHAP*

**RevuLens** is a BSCS thesis project (Bicol University College of Science) designed to assist consumers in identifying potentially deceptive e-commerce text reviews in real time. Rather than relying on opaque black-box predictions or platform-specific web scrapers, RevuLens works on **any shopping page through browser text selection**, pairing a frozen **DistilBERT-SVM** hybrid classification pipeline with **SHAP word-level explanations** rendered directly on the selected review text.

---

## Core Flow & Architecture

1. **Text Selection Trigger**: The shopper selects (highlights) review text on any shopping page (`window.getSelection()`). The extension does **not** parse the page's DOM or scan pages automatically. A small floating **"Analyze"** button appears near the selection.
2. **Service Worker Message Relay**: The content script reads the selection and sends the text to the background service worker (`chrome.runtime.sendMessage`). Content scripts never call the backend directly, preventing HTTPS mixed-content blocks.
3. **Two-Stage FastAPI Backend Pipeline**:
   - The service worker calls `POST /classify` first and immediately displays the classification badge in the result card.
   - The service worker then calls `POST /explain` while a small indicator (*"Loading explanation..."*) displays in the card until word highlights arrive. Classification is never held up by explanation.
4. **Machine Learning Pipeline**:
   - **Text Normalization**: Shared preprocessor (`backend/app/preprocessing.py::normalize_text`) decodes HTML entities, removes URLs, lowercases text, strips punctuation, and collapses whitespace (without stop-word removal).
   - **Contextual Representation**: Frozen multilingual DistilBERT (`distilbert-base-multilingual-cased`, frozen weights, mean pooling over token vectors excluding padding).
   - **Standardization & Classification**: Feature embeddings are standardized with `StandardScaler` (fit strictly on train split). A tuned `LinearSVC` classifies the review based strictly on the **sign of the decision function** ($>0 \rightarrow \text{Deceptive}$, $\le 0 \rightarrow \text{Genuine}$). No probability calibration (`CalibratedClassifierCV`) or confidence percentages are used.
5. **Post-Hoc Explainability (SHAP)**:
   - A SHAP text masker explains the decision-function score for the entire pipeline (`text → normalize → DistilBERT → scaler → SVM decision score`).
   - Sign convention: positive attribution weights push toward Deceptive, negative weights push toward Genuine.
6. **Non-Intrusive Extension UI**:
   - **Result Card**: Displays the display label badge (`Likely Genuine` in green/neutral or `Potentially Deceptive` in amber), a color legend, and a fixed disclaimer: *"Informational only; not proof of fraud."*
   - **Inline Highlights**: Words that influenced the decision are styled directly on the selected text (rose background with solid underline for deceptive-leaning words; green background with dotted underline for genuine-leaning words). Hovering shows a tooltip indicating *"toward Deceptive"* or *"toward Genuine"*.
   - **Zero Confidence Scores**: No percentage, confidence meter, probability, or raw SHAP value is shown anywhere to prevent misleading shoppers into reading model outputs as definitive proof.

---

## Repository Structure

```
revulens/
├── backend/
│   ├── app/
│   │   ├── api/            # Pydantic request/response schemas
│   │   ├── services/       # Feature extractor, LinearSVC classifier, SHAP explainer
│   │   ├── config.py       # Pydantic settings & environment configuration
│   │   ├── constants.py    # Single source of truth for label mappings
│   │   ├── main.py         # FastAPI application entrypoint (thread-safe routes)
│   │   └── preprocessing.py# Shared text normalization function
│   └── models/             # Trained pipeline joblib artifacts (gitignored)
├── extension/
│   ├── background/         # MV3 service worker API relay
│   ├── content/            # Selection listener, floating button, inline highlights, result card
│   ├── popup/              # Minimal on/off toggle, instructions & color guide
│   ├── config.js           # Extension settings & display label parity
│   └── manifest.json       # Chrome Manifest V3
├── notebooks/              # Reproducible pipeline steps (Steps 1 to 9)
├── data/                   # Data documentation & processed results (gitignored)
├── docs/                   # UAT instrument & research evaluation protocol
└── tests/                  # Unit, integration, and concurrency test suite
```

---

## Real Evaluation Results

All figures below are directly produced by real reproducible pipeline runs and saved to [`data/processed/model_evaluation_results.json`](file:///c:/Users/jessica/revulens/data/processed/model_evaluation_results.json) per the project's research integrity guidelines. The hybrid model is evaluated alongside the TF-IDF baseline on the exact same splits (Salminen et al., 2022; 80/10/10 stratified split, seed 42).

### 1. Validation Set: Pooling Strategy Comparison ($N=1,000$)
Mean pooling excluding padding was empirically compared against CLS first-token pooling on the validation split:

| Metric | Mean Pooling (Default) | CLS Token Pooling | Delta (Mean vs. CLS) |
| :--- | :---: | :---: | :---: |
| **Optimal Regularization ($C$)** | `0.01` | `0.01` | — |
| **Validation Accuracy** | **86.70%** | 82.90% | **+3.80%** |
| **Macro F1-Score** | **0.8669** | 0.8287 | **+0.0382** |
| **Deceptive Precision** | 86.30% | 83.80% | +2.50% |
| **Deceptive Recall** | 86.48% | 80.53% | +5.95% |
| **Deceptive F1-Score** | 86.39% | 82.13% | +4.26% |
| **Confusion Matrix** | TN=445, FP=67<br>FN=66, TP=422 | TN=436, FP=76<br>FN=95, TP=393 | — |

*Finding: Mean pooling outperforms CLS token pooling across all classification metrics and was adopted as the primary embedding representation.*

### 2. Held-Out Salminen Test Split Evaluation
Per project guidelines, the hybrid pipeline is not assumed to be superior to the baseline; both are reported side-by-side:

| Metric | TF-IDF Baseline ($N=500$) | DistilBERT Hybrid ($N=500$) | Full Baseline ($N=4,038$) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 96.00% | 85.00% | 95.34% |
| **Macro F1-Score** | 0.9600 | 0.8499 | 0.9534 |
| **Deceptive Precision** | 96.77% | 83.91% | 95.98% |
| **Deceptive Recall** | 95.24% | 86.90% | 94.65% |
| **Deceptive F1-Score** | 0.9600 | 0.8538 | 0.9531 |
| **Confusion Matrix** | TN=240, FP=8<br>FN=12, TP=240 | TN=206, FP=42<br>FN=33, TP=219 | TN=1941, FP=80<br>FN=108, TP=1909 |

### 3. FiReCS Taglish Behavioral Check ($N=500$, Exploratory Only)
FiReCS (Cosme & De Leon, 2024) contains Filipino-English reviews with sentiment labels only. It is **never used for training** and is **never evaluated for accuracy or F1**. Only the exploratory share flagged Deceptive is reported:
- **TF-IDF Baseline**: 24 flagged Potentially Deceptive (**4.80%**), 476 flagged Likely Genuine (**95.20%**).
- **DistilBERT Hybrid**: 2 flagged Potentially Deceptive (**0.40%**), 498 flagged Likely Genuine (**99.60%**).

---

## Terminology & Display Labels

- **Internal Classes**: `Genuine` / `Deceptive`.
- **User-Facing Display Labels**: `Likely Genuine` / `Potentially Deceptive`.
- **Dataset Labels**: Salminen et al. `OR` (Original Review) $\rightarrow$ `Genuine`, `CG` (Computer-Generated Review) $\rightarrow$ `Deceptive`.
- **Authenticity Meaning**: "Genuine" reviews in the dataset are presumed authentic. "Deceptive" refers specifically to GPT-2 synthetic text reviews, not human-written paid reviews.

---

## Scope & Boundaries

- **Selection-Based**: Operates on text selected by the shopper across any e-commerce site. No fragile, platform-specific DOM parsers or web scrapers.
- **Browser Compatibility**: Chromium-based desktop browsers (Chrome, Edge, Brave). Mobile browsers and Safari are out of scope.
- **Modality**: Text reviews only. Images, seller profiles, bot networks, and multi-account signals are not analyzed.
- **User Roles**: Consumer role only. There is no System Administrator role and no user account system.
- **Privacy**: Selected review text is processed in memory and never persisted or logged by the backend. Logs record only endpoint, HTTP status code, response timing, and optionally the decision score for internal diagnostics.
- **Informational Advisory**: RevuLens is an educational and decision-support tool. It does not provide legal or definitive proof of review fraud.

---

## Known Limitations

1. **Synthetic Text Patterns**: The model learns linguistic markers characteristic of GPT-2 generated reviews. Generalization to human-written incentivized fakes is not established.
2. **Presumed Ground Truth**: Dataset "Genuine" reviews are presumed authentic, not independently audited.
3. **Exploratory Taglish Behavior**: Filipino-English (Taglish) reviews are evaluated as an exploratory check; multilingual performance remains constrained by English training data.
4. **Selection Dependency**: Analysis accuracy depends strictly on the completeness of the text highlighted by the consumer.

---

## Local Development & Setup

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.11 / 3.12 / 3.14)
- Google Chrome or any Chromium-based desktop browser

### 2. Backend Setup
```bash
# Clone repository
git clone https://github.com/omskurtttttt/revulens.git
cd revulens

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt

# Start FastAPI development server
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```
The API is available at `http://127.0.0.1:8000` (`/health`, `/classify`, `/explain`, and `/docs`).

### 3. Docker Deployment (Containerized Backend)
Alternatively, deploy the backend using the Dockerfile (Chapter 3):
```bash
# Build backend container image
docker build -t revulens-backend .

# Run container exposing port 8000
docker run -p 8000:8000 revulens-backend
```

### 4. Extension Installation
1. Open Chrome and navigate to `chrome://extensions/`.
2. Enable **Developer mode** (toggle in top right).
3. Click **Load unpacked** and select the `extension/` directory from this repository.
4. Open any shopping site (e.g., Shopee Philippines), highlight any review text, and click the floating **Analyze** button.

### 5. Running Tests & Linters
```bash
# Python unit tests
python -m unittest discover tests

# JavaScript linter
npx eslint extension/
```
All unit, integration, contract, and concurrency tests run locally without requiring remote GPUs. Model-dependent tests skip automatically if local `.joblib` weights are absent.
