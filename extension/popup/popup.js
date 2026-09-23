/**
 * RevuLens Popup Controller
 */

const BACKEND_HEALTH_URL = "http://localhost:8000/api/v1/admin/health";

document.addEventListener("DOMContentLoaded", () => {
  const statusContainer = document.getElementById("backend-status");
  const statusLabel = document.getElementById("status-label");
  const toggleHighlights = document.getElementById("toggle-highlights");

  // Check backend health
  checkBackendHealth(statusContainer, statusLabel);

  // Load preferences from storage
  if (typeof chrome !== "undefined" && chrome.storage && chrome.storage.local) {
    chrome.storage.local.get(["enableHighlights"], (result) => {
      if (result.enableHighlights !== undefined) {
        toggleHighlights.checked = result.enableHighlights;
      }
    });

    toggleHighlights.addEventListener("change", (e) => {
      chrome.storage.local.set({ enableHighlights: e.target.checked });
    });
  }
});

async function checkBackendHealth(container, label) {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 2000);

    const res = await fetch(BACKEND_HEALTH_URL, {
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (res.ok) {
      container.className = "status-indicator online";
      label.textContent = "Online";
    } else {
      container.className = "status-indicator offline";
      label.textContent = "Error";
    }
  } catch (err) {
    container.className = "status-indicator offline";
    label.textContent = "Offline";
  }
}
