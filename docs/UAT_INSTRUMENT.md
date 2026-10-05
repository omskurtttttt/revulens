# RevuLens: ISO/IEC 25010:2023 User Acceptance Testing (UAT) Instrument

**Thesis Title:** *RevuLens: An Explainable Browser Extension for Detecting Fake E-Commerce Text Reviews Using DistilBERT-SVM and SHAP*  
**Institution:** Bicol University College of Science, Department of Computer Science  
**Evaluation Standard:** ISO/IEC 25010:2023 Systems and Software Engineering — Systems and software Quality Requirements and Evaluation (SQuaRE) — Product Quality Model

---

## 1. Overview and Purpose

This evaluation instrument assesses the software product quality of the **RevuLens** explainable browser extension. The evaluation focuses on three primary ISO/IEC 25010:2023 characteristics, plus one supplementary explainability characteristic:
1. **Functional Suitability** (Required: Completeness, Correctness, Appropriateness)
2. **Performance Efficiency** (Required: Time Behaviour, Resource Utilization)
3. **Interaction Capability** (Required: *Usability* in the 2023 edition: Appropriateness Recognizability, Learnability, Operability, User Error Protection, Aesthetics)
4. **Explainability & Trust Calibration** (Supplementary: Word-level SHAP attributions, Color and pattern underlines, Directional tooltips)

Per the thesis evaluation plan:
- **Consumer Cohort ($N = 30$):** Frequent online shoppers who make purchases on platforms such as Shopee Philippines at least 1–2 times per month.
- **Domain Expert Cohort ($N = 1$):** Domain specialist in Artificial Intelligence / Natural Language Processing and E-Commerce Systems (matching the scale of BU CS thesis *Fake-SHA*).

---

## 2. Evaluation Protocol & Tasks

Each participant completes the following guided evaluation workflow:

### Task 1: Extension Activation & Health Verification
1. Open the Chromium browser with the RevuLens extension loaded.
2. Click the RevuLens extension icon in the toolbar.
3. Observe the popup status indicator (`Online` / `Offline`) and review the "How to use" guidance.

### Task 2: Real-World Review Selection (Shopee PH / Shopping Page)
1. Navigate to an online shopping product review section.
2. Highlight a review's text using the mouse cursor (`window.getSelection()`).
3. Note the appearance of the small floating trigger button: `[Analyze]`.

### Task 3: Fast Classification Inspection (<150 ms)
1. Click the `[Analyze]` button.
2. Observe the floating RevuLens inspection card beside the selected text.
3. Note the immediate display of the hedged display label badge (`Likely Genuine` or `Potentially Deceptive`). *No misleading confidence percentage or probability meter is displayed.*

### Task 4: Token-Level SHAP Explanation & Accessible Highlights
1. Observe the token attribution section inside the card.
2. Review the color-coded and patterned word highlights on the selected text:
   - **Rose background with solid underline:** Words that influenced the result toward *Potentially Deceptive*.
   - **Green background with dotted underline:** Words that influenced the result toward *Likely Genuine*.
3. Hover on individual highlighted words to inspect plain-language directional influence tooltips (*"toward Deceptive"* or *"toward Genuine"*). No confusing raw numbers are shown to users.
4. Verify that the fixed disclaimer is visible: *"Informational only; not proof of fraud."*

---

## 3. Rating Scale & Scoring Rubric

Participants evaluate each statement on a 5-point Likert scale with interpretation ranges aligned with Chapter 3:

| Rating | Score Range | Descriptive Equivalent | Interpretation |
|:---:|:---:|:---|:---|
| **5** | 4.21 – 5.00 | **Strongly Agree** | **Excellent Quality** (Exceeds expectations) |
| **4** | 3.41 – 4.20 | **Agree** | **Good Quality** (Meets expectations fully) |
| **3** | 2.61 – 3.40 | **Neutral / Moderate** | **Fair Quality** (Acceptable, minor improvements needed) |
| **2** | 1.81 – 2.60 | **Disagree** | **Poor Quality** (Needs substantial improvement) |
| **1** | 1.00 – 1.80 | **Strongly Disagree** | **Very Poor Quality** (Unacceptable or non-functional) |

---

## 4. Standardized Questionnaire Items

### Dimension 1: Functional Suitability (ISO/IEC 25010:2023 Required)
- **FS-1 (Functional Completeness):** The extension successfully captures highlighted review text and returns both classification and word-level explanations.
- **FS-2 (Functional Correctness):** The classification badge (`Likely Genuine` / `Potentially Deceptive`) accurately reflects model assessment.
- **FS-3 (Functional Appropriateness):** The hedged display wording appropriately assists the consumer in evaluating potential review deception without making misleading guarantees.

### Dimension 2: Performance Efficiency (ISO/IEC 25010:2023 Required)
- **PE-1 (Time Behaviour — Classification):** The initial classification status badge displays promptly (<200 ms) upon highlighting review text.
- **PE-2 (Time Behaviour — Explanation):** The SHAP word-level explanation loads within a reasonable waiting time without freezing the browser page.
- **PE-3 (Resource Utilization):** The extension operates smoothly on the web browser without noticeable memory lag or browser slowdown.

### Dimension 3: Interaction Capability (ISO/IEC 25010:2023 Required)
- **IC-1 (Appropriateness Recognizability):** Users can readily understand the purpose of RevuLens and how it aids online review evaluation.
- **IC-2 (Learnability):** It is intuitive and easy to learn how to highlight a review and view RevuLens inspection results.
- **IC-3 (Operability):** The floating trigger button, modal card, and close buttons are easy to control and navigate.
- **IC-4 (User Error Protection):** The extension prevents user errors by ignoring empty selections and providing clear error states if the backend is unreachable.
- **IC-5 (User Interface Aesthetics):** The floating card design, badge colors, and typographic hierarchy look clean, modern, and visually appealing.

### Dimension 4: Explainability & Trust Calibration (Supplementary Characteristic)
- **EX-1 (Attribution Clarity):** The color-coded word highlights (green dotted for Genuine push, rose solid for Deceptive push) make it clear which words influenced the result.
- **EX-2 (Tooltip Interpretability):** Hovering on individual words provides informative directional tooltips explaining influence without exposing confusing raw math.
- **EX-3 (Trust Calibration):** The explanations and disclaimers help users understand that the system detects learned GPT-2 patterns rather than asserting absolute proof of fraud.

---

## 5. Qualitative Feedback & Expert Assessment

### Consumer Open-Ended Questions:
1. *What did you find most helpful about the words that influenced the result?*
2. *Did the hedged wording ("Likely Genuine" / "Potentially Deceptive") feel appropriate and trustworthy?*
3. *What suggestions do you have for improving the inspection interface?*

### Domain Expert Assessment:
1. *Evaluation of text-selection approach versus platform-specific DOM parsing.*
2. *Assessment of the DistilBERT-SVM hybrid pipeline without probability/confidence scores (decision-function sign determines class).*
3. *Assessment of SHAP text masking and word-level attribution validity for explainability in e-commerce.*
4. *Recommendation for future work.*

---

## 6. Strict Separation of Concerns & Research Integrity (Thesis Rule)

Per `GEMINI.md`:
- **Model Performance Metrics** evaluate algorithmic detection capacity for the hybrid and baseline models on the held-out Salminen test split (reported from saved evaluation files in `data/processed/final_test_evaluation.json`).
- **System Quality & UX Metrics** evaluate software interaction capability and user acceptance (reported from participant survey CSV data in `data/uat_responses.csv`).
- **Research Integrity Rule:** Evaluation figures must never be fabricated, simulated, or typed in by hand. Real evaluation outputs must populate the thesis tables directly.
- These two evaluation domains are **strictly non-interchangeable** and are reported in distinct sections of Thesis Chapter 4.
