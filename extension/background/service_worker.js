/**
 * RevuLens Background Service Worker (Manifest V3).
 *
 * Adheres strictly to GEMINI.md:
 * - Content scripts NEVER call the backend directly (avoids mixed-content blocking on HTTPS).
 * - All requests are routed through this service worker to the FastAPI backend.
 * - Handles /classify (fast <150ms) and /explain (asynchronous token weights).
 */

try {
  importScripts("../config.js");
} catch (e) {
  console.error("[RevuLens Worker] Failed to load config.js:", e);
}

const CONFIG = (typeof globalThis !== "undefined" && globalThis.REVULENS_CONFIG) || {
  API_BASE_URL: "http://127.0.0.1:8000",
  ENDPOINTS: {
    CLASSIFY: "/classify",
    EXPLAIN: "/explain",
    HEALTH: "/health"
  }
};

console.log("[RevuLens Worker] Service worker initialized. API Base:", CONFIG.API_BASE_URL);

/**
 * Handle incoming messages from content scripts and popup.
 */
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (!message || !message.action) {
    return false;
  }

  const { action, payload } = message;

  if (action === "CLASSIFY_TEXT") {
    handleClassify(payload)
      .then((data) => sendResponse({ success: true, data }))
      .catch((err) => sendResponse({ success: false, error: err.message }));
    return true; // Keep channel open for async response
  }

  if (action === "EXPLAIN_TEXT") {
    handleExplain(payload)
      .then((data) => sendResponse({ success: true, data }))
      .catch((err) => sendResponse({ success: false, error: err.message }));
    return true;
  }

  if (action === "CHECK_HEALTH") {
    handleHealthCheck()
      .then((data) => sendResponse({ success: true, data }))
      .catch((err) => sendResponse({ success: false, error: err.message }));
    return true;
  }

  return false;
});

/**
 * Send classification request to FastAPI backend.
 */
async function handleClassify(payload) {
  const url = `${CONFIG.API_BASE_URL}${CONFIG.ENDPOINTS.CLASSIFY}`;
  const response = await fetchWithTimeout(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      text: payload.text,
    }),
  }, 10000);

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Backend classification error (${response.status}): ${errorText}`);
  }

  const data = await response.json();

  // Update session counters in chrome.storage.local
  updateScanStats(data.label);

  return data;
}

/**
 * Send explanation request to FastAPI backend.
 */
async function handleExplain(payload) {
  const url = `${CONFIG.API_BASE_URL}${CONFIG.ENDPOINTS.EXPLAIN}`;
  const body = {
    text: payload.text,
  };
  if (payload.max_evals) body.max_evals = payload.max_evals;
  if (payload.max_words) body.max_words = payload.max_words;

  const response = await fetchWithTimeout(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  }, 30000); // 30s timeout for SHAP on longer texts

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Backend explain error (${response.status}): ${errorText}`);
  }

  return await response.json();
}

/**
 * Health check ping to backend.
 */
async function handleHealthCheck() {
  const url = `${CONFIG.API_BASE_URL}${CONFIG.ENDPOINTS.HEALTH}`;
  const response = await fetchWithTimeout(url, { method: "GET" }, 3000);
  if (!response.ok) {
    throw new Error(`Health check returned status ${response.status}`);
  }
  return await response.json();
}

/**
 * Fetch wrapper with configurable abort timeout.
 */
async function fetchWithTimeout(resource, options = {}, timeoutMs = 8000) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(resource, {
      ...options,
      signal: controller.signal,
    });
    clearTimeout(id);
    return response;
  } catch (err) {
    clearTimeout(id);
    if (err.name === "AbortError") {
      throw new Error("Request timed out. Please verify that the RevuLens FastAPI backend is running.");
    }
    throw new Error(`Network error connecting to RevuLens backend: ${err.message}. Ensure backend is running at ${CONFIG.API_BASE_URL}.`);
  }
}

/**
 * Increment local stats storage for popup display.
 */
function updateScanStats(label) {
  if (typeof chrome === "undefined" || !chrome.storage || !chrome.storage.local) {
    return;
  }

  chrome.storage.local.get(["totalScanned", "countGenuine", "countDeceptive"], (result) => {
    const total = (result.totalScanned || 0) + 1;
    const isDeceptive = label === "Deceptive";
    const genuine = (result.countGenuine || 0) + (isDeceptive ? 0 : 1);
    const deceptive = (result.countDeceptive || 0) + (isDeceptive ? 1 : 0);

    chrome.storage.local.set({
      totalScanned: total,
      countGenuine: genuine,
      countDeceptive: deceptive,
    });
  });
}
