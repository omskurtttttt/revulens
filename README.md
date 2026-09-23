# RevuLens

*An Explainable Browser Extension for Detecting Fake E-Commerce Text Reviews Using DistilBERT-SVM and SHAP*

**RevuLens** is a BSCS thesis project (Bicol University College of Science) designed to assist consumers in identifying potentially deceptive text reviews in real time while browsing Shopee Philippines. The system pairs a lightweight **DistilBERT-SVM** hybrid classification model with **SHAP-based explainability**, highlighting individual token contributions directly on the review text.

---

## Architecture Overview

1. **Shopee DOM Parser (Extension Content Script)**: Identifies and extracts visible review text on product pages.
2. **FastAPI Backend Pipeline**:
   - **DistilBERT**: Extracts contextual representations from review text.
   - **SVM Classifier**: Classifies text into binary classes (`Genuine` or `Deceptive`).
   - **SHAP Engine**: Calculates token-level attribution weights for the prediction.
3. **In-Page Overlay**: Highlights tokens according to their contribution score and renders an informational status badge (`Likely Genuine` / `Potentially Deceptive`).

---

## Project Structure

```
revulens/
├── backend/
│   └── app/
│       ├── api/            # API endpoints & routing
│       ├── services/       # Feature extraction, classifier, and SHAP services
│       ├── constants.py    # Centralized label mappings and system constants
│       └── main.py         # FastAPI application entrypoint
├── extension/
│   ├── config.js           # Extension settings and label definitions
│   ├── manifest.json       # Chrome Manifest V3
│   ├── content/            # In-page review DOM parser and overlay script
│   └── popup/              # Browser extension popup UI
├── notebooks/              # Data preprocessing, training, and evaluation scripts
└── tests/                  # Unit and integration test suite
```

---

## Scope & Boundaries

- **Target Platform**: Shopee Philippines (`shopee.ph`).
- **Modality**: Text reviews only.
- **Roles**: Consumer (extension user) and System Administrator (backend monitoring).
- **Disclaimer**: RevuLens is an informational decision-support tool. It identifies learned linguistic and textual patterns and does not assert legal or definitive authenticity.
