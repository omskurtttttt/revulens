/**
 * RevuLens Extension Central Configuration
 *
 * Centralized mapping for display labels and backend connection.
 * Do not hardcode user-facing labels across content scripts or UI components.
 */

export const CONFIG = {
  API_BASE_URL: "http://localhost:8000/api/v1",
  LABEL_MAPPING: {
    Genuine: "Likely Genuine",
    Deceptive: "Potentially Deceptive",
  },
  BADGE_COLORS: {
    "Likely Genuine": {
      background: "#e6f4ea",
      text: "#137333",
      border: "#ceead6",
    },
    "Potentially Deceptive": {
      background: "#fce8e6",
      text: "#c5221f",
      border: "#fad2cf",
    },
  },
  TARGET_DOMAIN: "shopee.ph",
  DISCLAIMER_TEXT:
    "RevuLens detects learned textual patterns and provides an informational assessment. It does not guarantee ground truth authenticity.",
};
