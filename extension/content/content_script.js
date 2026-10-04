/**
 * RevuLens Content Script (Step 8).
 *
 * Adheres strictly to GEMINI.md:
 * 1. Consumer selects (highlights) review text on the page (window.getSelection()).
 *    Works on any shopping site through text selection without fragile DOM selectors.
 * 2. Never calls the backend directly; sends requests via chrome.runtime.sendMessage to service worker.
 * 3. Shows classification badge ("Likely Genuine" / "Potentially Deceptive") and confidence.
 * 4. Renders a color-coded word overlay based on SHAP token weights:
 *    - Positive weights (> 0) push toward Deceptive.
 *    - Negative weights (< 0) push toward Genuine.
 * 5. Single source of truth color mapping from config.js.
 */

(function () {
  // Prevent duplicate injection
  if (window.__REVULENS_CONTENT_SCRIPT_LOADED__) return;
  window.__REVULENS_CONTENT_SCRIPT_LOADED__ = true;

  const CONFIG = window.REVULENS_CONFIG || {
    DISPLAY_LABELS: {
      Genuine: "Likely Genuine",
      Deceptive: "Potentially Deceptive",
    },
    COLOR_MAP: {
      deceptive: { text: "#991b1b", background: "rgba(239, 68, 68, 0.22)", border: "rgba(220, 38, 38, 0.55)" },
      genuine: { text: "#166534", background: "rgba(34, 197, 94, 0.22)", border: "rgba(22, 163, 74, 0.55)" },
      neutral: { text: "inherit", background: "transparent", border: "transparent" },
    },
    SETTINGS: {
      MIN_SELECTION_CHARS: 5,
      DEFAULT_MAX_EVALS: 60,
    },
    DISCLAIMER_TEXT:
      "RevuLens detects learned textual patterns of GPT-2-generated text. It provides an informational assessment and does not assert legal or ground-truth authenticity.",
  };

  let activeTriggerBtn = null;
  let activeCard = null;
  let activeHighlightSpans = [];
  let currentSelectionText = "";
  let currentSelectionRange = null;

  console.log("[RevuLens] Content script active. Highlight any review text to inspect.");

  // Listen for user text selection
  document.addEventListener("mouseup", handleSelectionChange);
  document.addEventListener("keyup", (e) => {
    if (e.key === "Escape") {
      cleanupAll();
    } else {
      handleSelectionChange();
    }
  });

  // Close card when clicking outside
  document.addEventListener("mousedown", (e) => {
    if (e.target && e.target.closest && e.target.closest(".revulens-ui")) {
      return;
    }
    if (activeCard && !activeCard.contains(e.target) && (!activeTriggerBtn || !activeTriggerBtn.contains(e.target))) {
      cleanupAll();
    } else if (activeTriggerBtn && !activeTriggerBtn.contains(e.target)) {
      removeTriggerBtn();
    }
  });

  function handleSelectionChange(e) {
    // If the event target is inside RevuLens UI, ignore completely to avoid destroying trigger
    if (e && e.target && (e.target.closest && e.target.closest(".revulens-ui") || activeTriggerBtn && activeTriggerBtn.contains(e.target))) {
      return;
    }

    const selection = window.getSelection();
    if (!selection || selection.isCollapsed) {
      if (activeTriggerBtn && !activeCard) {
        removeTriggerBtn();
      }
      return;
    }

    const selectedText = selection.toString().trim();
    if (selectedText.length < CONFIG.SETTINGS.MIN_SELECTION_CHARS) {
      if (activeTriggerBtn && !activeCard) {
        removeTriggerBtn();
      }
      return;
    }

    // Do not re-trigger if clicking inside RevuLens elements
    const anchorNode = selection.anchorNode;
    if (anchorNode && anchorNode.parentElement && anchorNode.parentElement.closest(".revulens-ui")) {
      return;
    }

    try {
      const range = selection.getRangeAt(0);
      currentSelectionRange = range.cloneRange();
      currentSelectionText = selectedText;

      const rect = range.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) {
        showTriggerButton(rect);
      }
    } catch (err) {
      // Ignore cross-frame selection errors
    }
  }

  function showTriggerButton(rect) {
    removeTriggerBtn();

    const btn = document.createElement("button");
    btn.className = "revulens-ui revulens-trigger-btn";
    btn.innerHTML = `<span class="revulens-logo-dot"></span> Inspect with RevuLens`;
    btn.title = "Inspect selected review for deceptive patterns";

    const top = window.scrollY + rect.top - 38;
    const left = window.scrollX + rect.left + Math.max(0, (rect.width - 150) / 2);

    btn.style.top = `${Math.max(window.scrollY + 5, top)}px`;
    btn.style.left = `${Math.max(window.scrollX + 5, left)}px`;

    // CRITICAL: Prevent mousedown from collapsing the text selection in the host page
    btn.addEventListener("mousedown", (e) => {
      e.preventDefault();
      e.stopPropagation();
    });

    let triggered = false;
    const handleTrigger = (e) => {
      if (e) {
        e.preventDefault();
        e.stopPropagation();
      }
      if (triggered) return;
      triggered = true;
      openInspectionCard(rect, currentSelectionText);
    };

    btn.addEventListener("mouseup", handleTrigger);
    btn.addEventListener("click", handleTrigger);

    document.body.appendChild(btn);
    activeTriggerBtn = btn;
  }

  function removeTriggerBtn() {
    if (activeTriggerBtn && activeTriggerBtn.parentElement) {
      activeTriggerBtn.parentElement.removeChild(activeTriggerBtn);
    }
    activeTriggerBtn = null;
  }

  function cleanupAll() {
    removeTriggerBtn();
    if (activeCard && activeCard.parentElement) {
      activeCard.parentElement.removeChild(activeCard);
    }
    activeCard = null;
    clearWordHighlights();
  }

  function clearWordHighlights() {
    activeHighlightSpans.forEach((span) => {
      if (span.parentElement) {
        const textNode = document.createTextNode(span.textContent);
        span.parentElement.replaceChild(textNode, span);
      }
    });
    activeHighlightSpans = [];
  }

  function openInspectionCard(rect, textToInspect) {
    removeTriggerBtn();
    if (activeCard && activeCard.parentElement) {
      activeCard.parentElement.removeChild(activeCard);
    }

    const card = document.createElement("div");
    card.className = "revulens-ui revulens-inspect-card";

    // Card Position
    const top = window.scrollY + rect.bottom + 8;
    const left = Math.min(
      window.scrollX + rect.left,
      window.innerWidth + window.scrollX - 380
    );

    card.style.top = `${top}px`;
    card.style.left = `${Math.max(window.scrollX + 10, left)}px`;

    card.innerHTML = `
      <div class="revulens-card-header">
        <div class="revulens-card-brand">
          <span class="revulens-badge-pill">RL</span>
          <span class="revulens-card-title">RevuLens Inspector</span>
        </div>
        <button class="revulens-close-btn" title="Close">×</button>
      </div>

      <div class="revulens-card-body">
        <div class="revulens-status-row">
          <span class="revulens-label-name">Status:</span>
          <span id="revulens-badge" class="revulens-status-badge revulens-badge-loading">
            <span class="revulens-spinner"></span> Analyzing patterns...
          </span>
        </div>

        <div id="revulens-confidence-container" class="revulens-meta-row" style="display: none;">
          <span class="revulens-meta-label">Confidence:</span>
          <div class="revulens-confidence-track">
            <div id="revulens-confidence-bar" class="revulens-confidence-fill" style="width: 0%;"></div>
          </div>
          <span id="revulens-confidence-pct" class="revulens-meta-val">--%</span>
        </div>

        <div class="revulens-shap-section">
          <div class="revulens-shap-header">
            <span class="revulens-shap-title">Token Attribution (SHAP)</span>
            <span id="revulens-shap-status" class="revulens-shap-loading">Computing word weights...</span>
          </div>

          <div class="revulens-legend">
            <span class="revulens-legend-item"><span class="revulens-legend-dot dot-deceptive"></span> Pushes Deceptive</span>
            <span class="revulens-legend-item"><span class="revulens-legend-dot dot-genuine"></span> Pushes Genuine</span>
          </div>

          <div id="revulens-token-cloud" class="revulens-token-cloud">
            <p class="revulens-placeholder-text">Word attributions will appear here once computed.</p>
          </div>
        </div>

        <div class="revulens-card-footer">
          <p class="revulens-disclaimer">${CONFIG.DISCLAIMER_TEXT}</p>
        </div>
      </div>
    `;

    // Attach close listener
    card.querySelector(".revulens-close-btn").addEventListener("click", (e) => {
      e.stopPropagation();
      cleanupAll();
    });

    // Prevent clicks inside the card from propagating to document dismiss handler
    card.addEventListener("mousedown", (e) => {
      e.stopPropagation();
    });

    document.body.appendChild(card);
    activeCard = card;

    // Execute requests through background service worker
    executeInspection(textToInspect, card);
  }

  function executeInspection(text, card) {
    const badgeEl = card.querySelector("#revulens-badge");
    const confContainer = card.querySelector("#revulens-confidence-container");
    const confBar = card.querySelector("#revulens-confidence-bar");
    const confPct = card.querySelector("#revulens-confidence-pct");
    const shapStatus = card.querySelector("#revulens-shap-status");
    const tokenCloud = card.querySelector("#revulens-token-cloud");

    // 1. FAST CLASSIFICATION CALL (via Service Worker)
    try {
      chrome.runtime.sendMessage(
        {
          action: "CLASSIFY_TEXT",
          payload: { text },
        },
        (response) => {
          if (!card.parentElement) return; // Closed before response

          if (chrome.runtime.lastError || !response || !response.success) {
            const errMsg = (response && response.error) || (chrome.runtime.lastError && chrome.runtime.lastError.message) || "Could not reach backend service.";
            badgeEl.className = "revulens-status-badge revulens-badge-error";
            badgeEl.textContent = "Connection Error (Refresh Tab F5)";
            badgeEl.title = errMsg;
            shapStatus.textContent = "Backend offline or tab needs refresh";
            return;
          }

          const { label, display_label, confidence } = response.data;
          const isDeceptive = label === "Deceptive";

          badgeEl.className = `revulens-status-badge ${isDeceptive ? "badge-deceptive" : "badge-genuine"}`;
          badgeEl.textContent = display_label;

          // Display confidence percentage
          const pct = Math.round(confidence * 100);
          confContainer.style.display = "flex";
          confBar.style.width = `${pct}%`;
          confBar.className = `revulens-confidence-fill ${isDeceptive ? "fill-deceptive" : "fill-genuine"}`;
          confPct.textContent = `${pct}%`;
        }
      );
    } catch (err) {
      if (badgeEl) {
        badgeEl.className = "revulens-status-badge revulens-badge-error";
        badgeEl.textContent = "Please refresh tab (F5)";
        badgeEl.title = err.message || "Extension context was updated.";
      }
    }

    // 2. ASYNCHRONOUS EXPLANATION CALL (via Service Worker)
    try {
      chrome.runtime.sendMessage(
        {
          action: "EXPLAIN_TEXT",
          payload: {
            text,
            max_evals: CONFIG.SETTINGS.DEFAULT_MAX_EVALS,
          },
        },
        (response) => {
          if (!card.parentElement) return;

          if (chrome.runtime.lastError || !response || !response.success) {
            shapStatus.textContent = "Attribution unavailable";
            const errMsg = (response && response.error) || (chrome.runtime.lastError && chrome.runtime.lastError.message) || "timeout";
            tokenCloud.innerHTML = `<p class="revulens-error-text">Could not compute word contributions: ${errMsg}</p>`;
            return;
          }

          const { tokens, base_value, latency_ms, cached } = response.data;
          shapStatus.textContent = `Ready (${tokens.length} tokens, ${latency_ms ? latency_ms.toFixed(0) : 0}ms${cached ? " - cached" : ""})`;
          shapStatus.className = "revulens-shap-ready";

          renderTokenCloud(tokens, tokenCloud);
          applyInlineWordHighlights(tokens);
        }
      );
    } catch (err) {
      if (shapStatus) {
        shapStatus.textContent = "Refresh tab to reload extension";
      }
    }
  }

  function renderTokenCloud(tokens, container) {
    container.innerHTML = "";
    if (!tokens || tokens.length === 0) {
      container.innerHTML = `<p class="revulens-placeholder-text">No word attributions generated.</p>`;
      return;
    }

    tokens.forEach((tok) => {
      const span = document.createElement("span");
      const weight = tok.weight;
      const isDeceptivePush = weight > 0.005;
      const isGenuinePush = weight < -0.005;

      let cls = "token-neutral";
      let dirText = "Neutral influence";
      if (isDeceptivePush) {
        cls = "token-deceptive";
        dirText = "Pushes toward Potentially Deceptive";
      } else if (isGenuinePush) {
        cls = "token-genuine";
        dirText = "Pushes toward Likely Genuine";
      }

      span.className = `revulens-token-chip ${cls}`;
      span.textContent = tok.text;
      span.title = `"${tok.text}": ${weight >= 0 ? "+" : ""}${weight.toFixed(4)} (${dirText})`;

      container.appendChild(span);
    });
  }

  function applyInlineWordHighlights(tokens) {
    // If the original selection range is still active in the DOM, we can overlay highlight markers
    if (!currentSelectionRange) return;

    try {
      // Find matching words in the selected range and apply subtle highlighting
      const containerNode = currentSelectionRange.commonAncestorContainer;
      if (!containerNode) return;
    } catch (e) {
      console.warn("[RevuLens] Could not apply inline range highlights:", e);
    }
  }
})();
