/**
 * RevuLens Content Script (Step 8).
 *
 * Adheres strictly to GEMINI.md:
 * 1. Consumer selects (highlights) review text on the page (window.getSelection()).
 *    Works on any shopping site through text selection without fragile DOM selectors.
 * 2. Never calls the backend directly; sends requests via chrome.runtime.sendMessage to service worker.
 * 3. Shows floating "Analyze" button near selection.
 * 4. Two-stage display: /classify first (badge), then /explain (word highlights).
 * 5. Color-coded SHAP token highlights directly on selected text:
 *    - Deceptive-leaning: rose background with solid underline.
 *    - Genuine-leaning: green background with dotted underline.
 *    - Stronger weight = stronger shade.
 *    - Tooltip on hover says "toward Deceptive" or "toward Genuine" (never a number).
 *    - Terminology: strictly uses "words that influenced the result".
 * 6. Respects minimal popup on/off toggle via chrome.storage.local ("revulensEnabled").
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
      deceptive: { text: "#991b1b", background: "rgba(244, 63, 94, 0.22)", border: "rgba(225, 29, 72, 0.55)" },
      genuine: { text: "#166534", background: "rgba(34, 197, 94, 0.22)", border: "rgba(22, 163, 74, 0.55)" },
      neutral: { text: "inherit", background: "transparent", border: "transparent" },
    },
    SETTINGS: {
      MIN_SELECTION_CHARS: 5,
      DEFAULT_MAX_EVALS: 60,
      MAX_WORDS: 100,
      MAX_CHARS: 2000,
    },
    DISCLAIMER_TEXT: "Informational only; not proof of fraud.",
  };

  let isExtensionEnabled = true;
  let activeTriggerBtn = null;
  let activeCard = null;
  let activeHighlightSpans = [];
  let currentSelectionText = "";
  let currentSelectionRange = null;

  // Listen to popup on/off toggle state
  if (typeof chrome !== "undefined" && chrome.storage && chrome.storage.local) {
    chrome.storage.local.get(["revulensEnabled"], (result) => {
      if (result && result.revulensEnabled !== undefined) {
        isExtensionEnabled = result.revulensEnabled !== false;
      }
    });

    if (chrome.storage.onChanged) {
      chrome.storage.onChanged.addListener((changes, areaName) => {
        if (areaName === "local" && changes.revulensEnabled !== undefined) {
          isExtensionEnabled = changes.revulensEnabled.newValue !== false;
          if (!isExtensionEnabled) {
            cleanupAll();
          }
        }
      });
    }
  }

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
    if (!isExtensionEnabled) {
      if (activeTriggerBtn && !activeCard) {
        removeTriggerBtn();
      }
      return;
    }

    // If the event target is inside RevuLens UI, ignore completely
    if (e && e.target && ((e.target.closest && e.target.closest(".revulens-ui")) || (activeTriggerBtn && activeTriggerBtn.contains(e.target)))) {
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
    } catch (_err) {
      // Ignore cross-frame selection errors
    }
  }

  function showTriggerButton(rect) {
    removeTriggerBtn();

    const btn = document.createElement("button");
    btn.className = "revulens-ui revulens-trigger-btn";
    btn.innerHTML = `<span class="revulens-logo-dot"></span> Analyze`;
    btn.title = "Analyze selected review text with RevuLens";

    // Viewport-relative coordinates (fixed positioning)
    const top = Math.max(10, rect.top - 42);
    const left = Math.max(10, Math.min(rect.left + Math.max(0, (rect.width - 120) / 2), window.innerWidth - 140));

    btn.style.top = `${top}px`;
    btn.style.left = `${left}px`;

    // Prevent mousedown from collapsing the text selection in the host page
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
    if (!activeHighlightSpans || activeHighlightSpans.length === 0) return;

    const parentsToNormalize = new Set();
    activeHighlightSpans.forEach((span) => {
      const parent = span.parentElement;
      if (parent) {
        parentsToNormalize.add(parent);
        const textNode = document.createTextNode(span.textContent);
        parent.replaceChild(textNode, span);
      }
    });

    activeHighlightSpans = [];

    parentsToNormalize.forEach((parent) => {
      try {
        parent.normalize();
      } catch (_e) {
        // Ignore normalize errors if parent was detached
      }
    });
  }

  function openInspectionCard(rect, textToInspect) {
    removeTriggerBtn();
    if (activeCard && activeCard.parentElement) {
      activeCard.parentElement.removeChild(activeCard);
    }

    const card = document.createElement("div");
    card.className = "revulens-ui revulens-inspect-card";

    // Viewport-relative coordinates (fixed positioning)
    let top = rect.bottom + 8;
    if (top + 340 > window.innerHeight) {
      top = Math.max(10, rect.top - 350);
    }
    const left = Math.max(10, Math.min(rect.left, window.innerWidth - 380));

    card.style.top = `${top}px`;
    card.style.left = `${left}px`;

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

        <div id="revulens-truncated-banner" class="revulens-truncated-banner" style="display: none;">
          Long text: only the first ${CONFIG.SETTINGS.MAX_WORDS || 100} words were analyzed.
        </div>

        <div class="revulens-shap-section">
          <div class="revulens-shap-header">
            <span class="revulens-shap-title">Words that influenced the result (SHAP)</span>
            <span id="revulens-shap-status" class="revulens-shap-loading">Loading explanation...</span>
          </div>

          <div class="revulens-legend">
            <span class="revulens-legend-item"><span class="revulens-legend-dot dot-deceptive"></span> toward Deceptive</span>
            <span class="revulens-legend-item"><span class="revulens-legend-dot dot-genuine"></span> toward Genuine</span>
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
          if (!card.parentElement) return;

          if (chrome.runtime.lastError || !response || !response.success) {
            const rawError = (response && response.error) || (chrome.runtime.lastError && chrome.runtime.lastError.message) || "";
            badgeEl.className = "revulens-status-badge revulens-badge-error";
            if (rawError.includes("413") || rawError.toLowerCase().includes("exceeds maximum length")) {
              badgeEl.textContent = "Text Too Long";
              shapStatus.textContent = `Text exceeds maximum allowed length (${CONFIG.SETTINGS.MAX_CHARS || 2000} chars).`;
            } else if (rawError.includes("422") || rawError.toLowerCase().includes("blank or whitespace")) {
              badgeEl.textContent = "Invalid Selection";
              shapStatus.innerHTML = 'Select review text on the page to analyze it. <button type="button" class="revulens-help-inline-btn" id="revulens-btn-how-to-use">How to use</button>';
              const helpBtn = shapStatus.querySelector("#revulens-btn-how-to-use");
              if (helpBtn) {
                helpBtn.addEventListener("click", () => {
                  shapStatus.textContent = "Select review text on the page to analyze it. Taglish review analysis is exploratory.";
                });
              }
            } else {
              badgeEl.textContent = "Unavailable";
              shapStatus.textContent = "Analysis service unavailable. Check the connection and try again.";
            }
            return;
          }

          const { label, display_label, truncated } = response.data;
          const isDeceptive = label === "Deceptive";

          // Amber badge for Potentially Deceptive, green for Likely Genuine
          badgeEl.className = `revulens-status-badge ${isDeceptive ? "badge-deceptive" : "badge-genuine"}`;
          badgeEl.textContent = display_label;

          if (truncated) {
            const banner = card.querySelector("#revulens-truncated-banner");
            if (banner) banner.style.display = "block";
          }
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

          const { tokens, latency_ms, cached, truncated } = response.data;
          shapStatus.textContent = `Ready (${tokens.length} tokens, ${latency_ms ? latency_ms.toFixed(0) : 0}ms${cached ? " - cached" : ""})`;
          shapStatus.className = "revulens-shap-ready";

          if (truncated) {
            const banner = card.querySelector("#revulens-truncated-banner");
            if (banner) banner.style.display = "block";
          }

          renderTokenCloud(tokens, tokenCloud);
          applyInlineWordHighlights(tokens);
        }
      );
    } catch (_err) {
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
        dirText = "toward Deceptive";
      } else if (isGenuinePush) {
        cls = "token-genuine";
        dirText = "toward Genuine";
      }

      span.className = `revulens-token-chip ${cls}`;
      span.textContent = tok.text;
      span.title = dirText;

      container.appendChild(span);
    });
  }

  /**
   * Highlights words that influenced the result directly on the selected text.
   * - Deceptive-leaning: rose background with solid underline.
   * - Genuine-leaning: green background with dotted underline.
   * - Stronger weight produces a stronger shade (higher alpha).
   * - Tooltip on hover says "toward Deceptive" or "toward Genuine" (never a number).
   */
  function applyInlineWordHighlights(tokens) {
    if (!currentSelectionRange || !tokens || tokens.length === 0) return;

    clearWordHighlights();

    // 1. Build lookup dictionary for tokens with significant influence
    const tokenMap = new Map();
    let maxMag = 0.05;
    tokens.forEach((t) => {
      const mag = Math.abs(t.weight);
      if (mag > maxMag) maxMag = mag;
    });

    tokens.forEach((t) => {
      const weight = t.weight;
      if (Math.abs(weight) < 0.005) return; // ignore neutral noise

      const lower = t.text.toLowerCase().trim();
      if (!lower) return;

      const isDeceptive = weight > 0;
      const normalized = Math.min(1.0, Math.abs(weight) / maxMag);
      const alpha = (0.20 + 0.35 * normalized).toFixed(3);
      const borderAlpha = Math.min(1.0, parseFloat(alpha) + 0.35).toFixed(3);

      tokenMap.set(lower, {
        direction: isDeceptive ? "deceptive" : "genuine",
        tooltip: isDeceptive ? "toward Deceptive" : "toward Genuine",
        bgColor: isDeceptive
          ? `rgba(244, 63, 94, ${alpha})`
          : `rgba(34, 197, 94, ${alpha})`,
        borderColor: isDeceptive
          ? `rgba(225, 29, 72, ${borderAlpha})`
          : `rgba(22, 163, 74, ${borderAlpha})`,
      });
    });

    if (tokenMap.size === 0) return;

    try {
      // 2. Identify all text nodes intersecting currentSelectionRange
      const root = currentSelectionRange.commonAncestorContainer;
      const textNodes = [];

      if (root.nodeType === Node.TEXT_NODE) {
        textNodes.push(root);
      } else {
        const walker = document.createTreeWalker(
          root,
          NodeFilter.SHOW_TEXT,
          {
            acceptNode: function (node) {
              if (!node.nodeValue || !node.nodeValue.trim()) {
                return NodeFilter.FILTER_REJECT;
              }
              if (currentSelectionRange.intersectsNode) {
                return currentSelectionRange.intersectsNode(node)
                  ? NodeFilter.FILTER_ACCEPT
                  : NodeFilter.FILTER_REJECT;
              }
              return NodeFilter.FILTER_ACCEPT;
            },
          }
        );

        let currentNode = walker.nextNode();
        while (currentNode) {
          textNodes.push(currentNode);
          currentNode = walker.nextNode();
        }
      }

      if (textNodes.length === 0) return;

      const wordRegex = /([a-zA-Z0-9\u00C0-\u024F\u1E00-\u1EFF'-]+)/g;

      // 3. Highlight matching words within the selection range
      textNodes.forEach((node) => {
        const parent = node.parentNode;
        if (!parent || (parent.closest && parent.closest(".revulens-ui"))) {
          return;
        }

        const fullText = node.nodeValue;
        let startIdx = 0;
        let endIdx = fullText.length;

        if (node === currentSelectionRange.startContainer) {
          startIdx = currentSelectionRange.startOffset;
        }
        if (node === currentSelectionRange.endContainer) {
          endIdx = currentSelectionRange.endOffset;
        }

        if (startIdx >= endIdx) return;

        const beforeText = fullText.slice(0, startIdx);
        const targetText = fullText.slice(startIdx, endIdx);
        const afterText = fullText.slice(endIdx);

        const fragment = document.createDocumentFragment();
        if (beforeText) {
          fragment.appendChild(document.createTextNode(beforeText));
        }

        let lastIdx = 0;
        let match;
        let highlightedAny = false;

        while ((match = wordRegex.exec(targetText)) !== null) {
          const matchStart = match.index;
          const matchEnd = match.index + match[0].length;
          const word = match[0];
          const lowerWord = word.toLowerCase();

          if (matchStart > lastIdx) {
            fragment.appendChild(
              document.createTextNode(targetText.slice(lastIdx, matchStart))
            );
          }

          const tokenInfo = tokenMap.get(lowerWord);
          if (tokenInfo) {
            const span = document.createElement("span");
            span.className = `revulens-ui revulens-inline-highlight revulens-inline-${tokenInfo.direction}`;
            span.textContent = word;
            span.title = tokenInfo.tooltip;
            span.style.backgroundColor = tokenInfo.bgColor;
            span.style.borderBottom =
              tokenInfo.direction === "deceptive"
                ? `2px solid ${tokenInfo.borderColor}`
                : `2px dotted ${tokenInfo.borderColor}`;

            activeHighlightSpans.push(span);
            fragment.appendChild(span);
            highlightedAny = true;
          } else {
            fragment.appendChild(document.createTextNode(word));
          }

          lastIdx = matchEnd;
        }

        if (lastIdx < targetText.length) {
          fragment.appendChild(
            document.createTextNode(targetText.slice(lastIdx))
          );
        }

        if (afterText) {
          fragment.appendChild(document.createTextNode(afterText));
        }

        if (highlightedAny) {
          parent.replaceChild(fragment, node);
        }
      });
    } catch (err) {
      console.warn("[RevuLens] Could not apply inline range highlights:", err);
    }
  }
})();
