/**
 * RevuLens Extension Central Configuration & Constants.
 *
 * Single source of truth for display labels, color mappings, and API endpoints.
 * Strictly adheres to GEMINI.md:
 * - Internal binary classes: "Genuine" / "Deceptive"
 * - User-facing display labels: "Likely Genuine" / "Potentially Deceptive"
 * - One central color mapping for SHAP token attributions
 * - Disclaimers: Informational assessment, no legal claims
 */

const REVULENS_CONFIG = {
  API_BASE_URL: "http://127.0.0.1:8000",
  ENDPOINTS: {
    CLASSIFY: "/classify",
    EXPLAIN: "/explain",
    HEALTH: "/health",
  },
  INTERNAL_CLASSES: {
    GENUINE: "Genuine",
    DECEPTIVE: "Deceptive",
  },
  DISPLAY_LABELS: {
    Genuine: "Likely Genuine",
    Deceptive: "Potentially Deceptive",
  },
  // Single constant mapping for token and badge styling
  COLOR_MAP: {
    deceptive: {
      background: "rgba(239, 68, 68, 0.22)",
      border: "rgba(220, 38, 38, 0.55)",
      text: "#991b1b",
      badgeBg: "#fef2f2",
      badgeText: "#991b1b",
      badgeBorder: "#fecaca",
    },
    genuine: {
      background: "rgba(34, 197, 94, 0.22)",
      border: "rgba(22, 163, 74, 0.55)",
      text: "#166534",
      badgeBg: "#f0fdf4",
      badgeText: "#166534",
      badgeBorder: "#bbf7d0",
    },
    neutral: {
      background: "transparent",
      border: "transparent",
      text: "inherit",
    },
  },
  SETTINGS: {
    DEFAULT_MAX_EVALS: 60,
    MAX_WORDS: 100,
    MIN_SELECTION_CHARS: 5,
  },
  DISCLAIMER_TEXT:
    "RevuLens detects learned textual patterns of GPT-2-generated text. It provides an informational assessment and does not assert legal or ground-truth authenticity.",
};

// Make config universally accessible across globalThis, window, and worker scopes
if (typeof globalThis !== "undefined") {
  globalThis.REVULENS_CONFIG = REVULENS_CONFIG;
}
if (typeof module !== "undefined" && module.exports) {
  module.exports = REVULENS_CONFIG;
}
