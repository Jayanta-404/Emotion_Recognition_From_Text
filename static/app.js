/**
 * Emotion Recognition System - Frontend Application Logic
 * Interactive UI, Multi-Model Inference, Speech Recognition, Batch Analysis
 */

document.addEventListener("DOMContentLoaded", () => {
  // App State
  const state = {
    selectedModel: "keras",
    activeTab: "detector",
    autoAnalyze: false,
    autoAnalyzeTimer: null,
    history: JSON.parse(localStorage.getItem("emotion_history") || "[]"),
    presets: [],
    metadata: null,
    isRecording: false,
    recognition: null
  };

  // DOM Elements
  const navTabBtns = document.querySelectorAll(".nav-tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");
  const modelBtns = document.querySelectorAll(".model-btn");
  const textInput = document.getElementById("emotionTextInput");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const clearBtn = document.getElementById("clearBtn");
  const micBtn = document.getElementById("micBtn");
  const randomSampleBtn = document.getElementById("randomSampleBtn");
  const samplePromptBtn = document.getElementById("samplePromptBtn");
  const presetsBar = document.getElementById("presetsBar");
  const charCountDisplay = document.getElementById("charCountDisplay");
  const liveAutoToggle = document.getElementById("liveAutoAnalyzeToggle");

  // Results Elements
  const emptyState = document.getElementById("emptyState");
  const activeResultBox = document.getElementById("activeResultBox");
  const embeddedComparisonBox = document.getElementById("embeddedComparisonBox");
  const resultHeroBox = document.getElementById("resultHeroBox");
  const resEmoji = document.getElementById("resEmoji");
  const resEmotionName = document.getElementById("resEmotionName");
  const resDesc = document.getElementById("resDesc");
  const resConfidence = document.getElementById("resConfidence");
  const resSentiment = document.getElementById("resSentiment");
  const resValence = document.getElementById("resValence");
  const resModelBadge = document.getElementById("resModelBadge");
  const resLatency = document.getElementById("resLatency");
  const breakdownList = document.getElementById("breakdownList");
  const wordsCloud = document.getElementById("wordsCloud");
  const lastLatencyVal = document.getElementById("lastLatencyVal");

  // Comparison Tab Elements
  const cmpInputText = document.getElementById("cmpInputText");
  const runCompareBtn = document.getElementById("runCompareBtn");
  const comparisonResultsArea = document.getElementById("comparisonResultsArea");

  // Batch Tab Elements
  const dropZone = document.getElementById("dropZone");
  const batchFileInput = document.getElementById("batchFileInput");
  const batchTextarea = document.getElementById("batchTextarea");
  const batchModelSelect = document.getElementById("batchModelSelect");
  const runBatchBtn = document.getElementById("runBatchBtn");
  const batchAnalyticsArea = document.getElementById("batchAnalyticsArea");
  const batchDistGrid = document.getElementById("batchDistGrid");
  const batchTableBody = document.getElementById("batchTableBody");
  const exportBatchCsvBtn = document.getElementById("exportBatchCsvBtn");

  // History Elements
  const openHistoryBtn = document.getElementById("openHistoryBtn");
  const closeHistoryBtn = document.getElementById("closeHistoryBtn");
  const historyDrawer = document.getElementById("historyDrawer");
  const historyBackdrop = document.getElementById("historyBackdrop");
  const historyList = document.getElementById("historyList");
  const historyCount = document.getElementById("historyCount");
  const clearHistoryBtn = document.getElementById("clearHistoryBtn");

  let currentBatchData = null;

  // Initialize
  initApp();

  async function initApp() {
    updateHistoryCount();
    setupNavigation();
    setupModelSelector();
    setupInputListeners();
    setupSpeechRecognition();
    setupBatchUpload();
    setupHistoryDrawer();
    await fetchPresets();
    await fetchMetadata();
  }

  // Navigation Tabs
  function setupNavigation() {
    navTabBtns.forEach(btn => {
      btn.addEventListener("click", () => {
        const tab = btn.getAttribute("data-tab");
        switchTab(tab);
      });
    });
  }

  function switchTab(tabId) {
    state.activeTab = tabId;
    navTabBtns.forEach(b => {
      b.classList.toggle("active", b.getAttribute("data-tab") === tabId);
    });
    tabContents.forEach(c => {
      c.classList.toggle("active", c.id === `tab-${tabId}`);
    });
  }

  // Model Selection
  function setupModelSelector() {
    modelBtns.forEach(btn => {
      btn.addEventListener("click", () => {
        modelBtns.forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        state.selectedModel = btn.getAttribute("data-model");

        if (textInput.value.trim().length > 0) {
          executePrediction();
        }
      });
    });
  }

  // Text Input & Live Debounce
  function setupInputListeners() {
    textInput.addEventListener("input", () => {
      updateCharCount();
      if (state.autoAnalyze) {
        clearTimeout(state.autoAnalyzeTimer);
        state.autoAnalyzeTimer = setTimeout(() => {
          if (textInput.value.trim().length >= 3) {
            executePrediction();
          }
        }, 500);
      }
    });

    textInput.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        executePrediction();
      }
    });

    analyzeBtn.addEventListener("click", () => {
      executePrediction();
    });

    clearBtn.addEventListener("click", () => {
      textInput.value = "";
      updateCharCount();
      resetResults();
      textInput.focus();
    });

    liveAutoToggle.addEventListener("change", (e) => {
      state.autoAnalyze = e.target.checked;
      if (state.autoAnalyze && textInput.value.trim().length >= 3) {
        executePrediction();
      }
    });

    randomSampleBtn.addEventListener("click", pickRandomPreset);
    samplePromptBtn.addEventListener("click", pickRandomPreset);

    // Hero Taxonomy Bar Clicks
    document.querySelectorAll(".hero-emotions-bar .emotion-chip").forEach(chip => {
      chip.addEventListener("click", () => {
        const emotion = chip.getAttribute("data-emotion");
        const match = state.presets.find(p => p.emotion === emotion);
        if (match) {
          textInput.value = match.text;
          updateCharCount();
          switchTab("detector");
          executePrediction();
        }
      });
    });

    // Comparison Tab Run
    runCompareBtn.addEventListener("click", () => {
      const text = cmpInputText.value.trim();
      if (!text) return;
      executeComparison(text, comparisonResultsArea);
    });
  }

  function updateCharCount() {
    const text = textInput.value;
    const chars = text.length;
    const words = text.trim() ? text.trim().split(/\s+/).length : 0;
    charCountDisplay.textContent = `${chars} chars • ${words} words`;
  }

  // Presets
  async function fetchPresets() {
    try {
      const res = await fetch("/api/presets");
      const data = await res.json();
      state.presets = data.presets || [];
      renderPresets(state.presets);
    } catch (e) {
      console.warn("Could not load presets from server, using local fallback", e);
      state.presets = [
        { emotion: "joy", title: "Dream Job", emoji: "😊", text: "I just received the call offering me my dream role! I am overjoyed!" },
        { emotion: "sadness", title: "Saying Goodbye", emoji: "😢", text: "It was so hard waving goodbye today. My heart feels so heavy and empty." },
        { emotion: "anger", title: "Unfair Treatment", emoji: "😡", text: "I am absolutely disgusted and furious with how unfairly they treated us!" },
        { emotion: "fear", title: "Nightmare", emoji: "😨", text: "I was walking in the dark when sudden footsteps behind me made my pulse spike in fear." },
        { emotion: "love", title: "Devotion", emoji: "❤️", text: "You bring so much gentle warmth into my life. I cherish and adore you." },
        { emotion: "surprise", title: "Unexpected Party", emoji: "😲", text: "I opened the door and everyone shouted Surprise! I was completely astonished!" }
      ];
      renderPresets(state.presets);
    }
  }

  function renderPresets(presets) {
    presetsBar.innerHTML = "";
    presets.forEach(preset => {
      const chip = document.createElement("button");
      chip.className = "preset-chip";
      chip.innerHTML = `<span>${preset.emoji}</span><span>${preset.title}</span>`;
      chip.title = preset.text;
      chip.addEventListener("click", () => {
        textInput.value = preset.text;
        updateCharCount();
        executePrediction();
      });
      presetsBar.appendChild(chip);
    });
  }

  function pickRandomPreset() {
    if (!state.presets.length) return;
    const item = state.presets[Math.floor(Math.random() * state.presets.length)];
    textInput.value = item.text;
    updateCharCount();
    executePrediction();
  }

  // Metadata & System Health
  async function fetchMetadata() {
    try {
      const res = await fetch("/api/metadata");
      const data = await res.json();
      state.metadata = data;
      document.getElementById("systemStatusText").textContent = "3 Models Online";
    } catch (e) {
      console.error("Failed to fetch system metadata", e);
      document.getElementById("systemStatusText").textContent = "Backend Offline";
      document.getElementById("systemStatusBadge").style.color = "#ef4444";
    }
  }

  // Core Prediction Execution
  async function executePrediction() {
    const text = textInput.value.trim();
    if (!text) {
      resetResults();
      return;
    }

    analyzeBtn.disabled = true;
    analyzeBtn.style.opacity = "0.7";
    analyzeBtn.textContent = "Analyzing...";

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: text,
          model: state.selectedModel
        })
      });

      if (!res.ok) {
        throw new Error(`API returned status ${res.status}`);
      }

      const data = await res.json();

      if (data.is_comparison) {
        renderComparisonView(data, embeddedComparisonBox);
        activeResultBox.style.display = "none";
        emptyState.style.display = "none";
        embeddedComparisonBox.style.display = "block";
      } else {
        renderSingleResult(data);
        embeddedComparisonBox.style.display = "none";
        emptyState.style.display = "none";
        activeResultBox.style.display = "block";
      }

      // Add to session history
      saveHistory(text, data);

    } catch (err) {
      console.error("Prediction error:", err);
      alert("Failed to get prediction from model backend. Check console for details.");
    } finally {
      analyzeBtn.disabled = false;
      analyzeBtn.style.opacity = "1";
      analyzeBtn.textContent = "Analyze Emotion ✨";
    }
  }

  // Render Single Model Result
  function renderSingleResult(data) {
    resEmoji.textContent = data.emoji;
    resEmotionName.textContent = data.dominant_emotion;
    resDesc.textContent = data.description || "";
    resConfidence.textContent = `${data.confidence_pct}%`;
    resSentiment.textContent = data.sentiment;
    resValence.textContent = (data.valence_score > 0 ? "+" : "") + data.valence_score;
    resModelBadge.textContent = data.model_used.split("(")[0].trim();
    resLatency.textContent = `${data.latency_ms} ms`;
    lastLatencyVal.textContent = `${data.latency_ms} ms`;

    // Dynamic accent color based on emotion
    resultHeroBox.style.setProperty("--result-color", data.color);
    resultHeroBox.style.setProperty("--result-border", data.color);
    resultHeroBox.style.setProperty("--result-glow", `${data.color}33`);
    resultHeroBox.style.setProperty("--result-color-alpha", `${data.color}18`);

    // Render 6 Emotion Breakdown Bars
    breakdownList.innerHTML = "";
    data.breakdown.forEach((item, index) => {
      const row = document.createElement("div");
      row.className = "breakdown-row";
      row.innerHTML = `
        <div class="breakdown-info">
          <div class="breakdown-name-flex">
            <span>${item.emoji}</span>
            <span style="color: ${item.color}">${item.emotion}</span>
            ${index === 0 ? '<span style="font-size: 0.65rem; background: rgba(255,255,255,0.1); padding: 0.1rem 0.4rem; border-radius: 4px;">Top</span>' : ''}
          </div>
          <div class="breakdown-pct" style="color: ${item.color}">${item.percentage}%</div>
        </div>
        <div class="meter-track">
          <div class="meter-fill" style="width: 0%; background: ${item.gradient || item.color}"></div>
        </div>
      `;
      breakdownList.appendChild(row);

      // Trigger width animation
      requestAnimationFrame(() => {
        setTimeout(() => {
          const fill = row.querySelector(".meter-fill");
          if (fill) fill.style.width = `${item.percentage}%`;
        }, index * 40);
      });
    });

    // Word attribution cloud
    wordsCloud.innerHTML = "";
    if (data.word_highlights && data.word_highlights.length > 0) {
      data.word_highlights.forEach(token => {
        const tag = document.createElement("span");
        tag.className = `word-tag ${token.in_vocab ? "in-vocab" : ""}`;
        tag.textContent = token.word;
        if (token.in_vocab) tag.title = "Known in model vocabulary / feature space";
        wordsCloud.appendChild(tag);
      });
    } else {
      wordsCloud.innerHTML = '<span style="font-size: 0.8rem; color: var(--text-muted);">No tokens extracted.</span>';
    }
  }

  // Render Comparison Matrix View
  function renderComparisonView(data, targetContainer) {
    targetContainer.innerHTML = `
      <div class="comparison-view-container">
        <div class="consensus-banner">
          <div class="consensus-left">
            <div class="consensus-emoji">${data.consensus_emoji}</div>
            <div>
              <div class="consensus-title">Consensus Verdict</div>
              <div class="consensus-name" style="color: ${data.consensus_color}">${data.consensus_emotion}</div>
            </div>
          </div>
          <div>
            <span class="consensus-agree-badge">${data.agreement_level}</span>
          </div>
        </div>

        <div class="models-comparison-grid">
          <!-- Keras Card -->
          <div class="model-card-cmp highlight">
            <div class="cmp-card-header">
              <div class="cmp-model-type">Deep Learning</div>
              <div class="cmp-model-name">Keras BiLSTM-CNN</div>
            </div>
            <div class="cmp-prediction-box" style="border-color: ${data.models.keras.color};">
              <div class="cmp-pred-emoji">${data.models.keras.emoji}</div>
              <div class="cmp-pred-name" style="color: ${data.models.keras.color};">${data.models.keras.dominant_emotion}</div>
              <div class="cmp-pred-conf">${data.models.keras.confidence_pct}% Conf</div>
            </div>
            <div>
              <div class="cmp-stats-row"><span>Sentiment:</span> <strong>${data.models.keras.sentiment}</strong></div>
              <div class="cmp-stats-row"><span>Latency:</span> <span style="font-family: var(--font-mono); color: #34d399;">${data.models.keras.latency_ms} ms</span></div>
              <div class="cmp-stats-row"><span>Features:</span> <span>FastText 100d</span></div>
            </div>
          </div>

          <!-- Logistic Regression Card -->
          <div class="model-card-cmp">
            <div class="cmp-card-header">
              <div class="cmp-model-type">Classical ML</div>
              <div class="cmp-model-name">Logistic Regression</div>
            </div>
            <div class="cmp-prediction-box" style="border-color: ${data.models.logistic.color};">
              <div class="cmp-pred-emoji">${data.models.logistic.emoji}</div>
              <div class="cmp-pred-name" style="color: ${data.models.logistic.color};">${data.models.logistic.dominant_emotion}</div>
              <div class="cmp-pred-conf">${data.models.logistic.confidence_pct}% Conf</div>
            </div>
            <div>
              <div class="cmp-stats-row"><span>Sentiment:</span> <strong>${data.models.logistic.sentiment}</strong></div>
              <div class="cmp-stats-row"><span>Latency:</span> <span style="font-family: var(--font-mono); color: #34d399;">${data.models.logistic.latency_ms} ms</span></div>
              <div class="cmp-stats-row"><span>Features:</span> <span>TF-IDF 30K</span></div>
            </div>
          </div>

          <!-- Linear SVM Card -->
          <div class="model-card-cmp">
            <div class="cmp-card-header">
              <div class="cmp-model-type">Classical ML</div>
              <div class="cmp-model-name">Linear SVM (SVC)</div>
            </div>
            <div class="cmp-prediction-box" style="border-color: ${data.models.svm.color};">
              <div class="cmp-pred-emoji">${data.models.svm.emoji}</div>
              <div class="cmp-pred-name" style="color: ${data.models.svm.color};">${data.models.svm.dominant_emotion}</div>
              <div class="cmp-pred-conf">${data.models.svm.confidence_pct}% Conf</div>
            </div>
            <div>
              <div class="cmp-stats-row"><span>Sentiment:</span> <strong>${data.models.svm.sentiment}</strong></div>
              <div class="cmp-stats-row"><span>Latency:</span> <span style="font-family: var(--font-mono); color: #34d399;">${data.models.svm.latency_ms} ms</span></div>
              <div class="cmp-stats-row"><span>Features:</span> <span>TF-IDF 30K</span></div>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  async function executeComparison(text, targetArea) {
    targetArea.innerHTML = `
      <div class="results-empty-state">
        <div class="empty-icon">⏳</div>
        <h3 class="empty-title">Benchmarking All 3 Models...</h3>
      </div>
    `;

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: text,
          model: "compare_all"
        })
      });
      const data = await res.json();
      renderComparisonView(data, targetArea);
    } catch (e) {
      targetArea.innerHTML = `<p style="color: #ef4444; text-align: center; padding: 2rem;">Comparison failed: ${e.message}</p>`;
    }
  }

  function resetResults() {
    emptyState.style.display = "flex";
    activeResultBox.style.display = "none";
    embeddedComparisonBox.style.display = "none";
  }

  // Speech Recognition (Web Speech API)
  function setupSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      micBtn.style.display = "none";
      return;
    }

    state.recognition = new SpeechRecognition();
    state.recognition.continuous = false;
    state.recognition.interimResults = false;
    state.recognition.lang = "en-US";

    state.recognition.onstart = () => {
      state.isRecording = true;
      micBtn.classList.add("recording");
      micBtn.title = "Listening... Speak now";
    };

    state.recognition.onresult = (e) => {
      const transcript = e.results[0][0].transcript;
      textInput.value = (textInput.value + " " + transcript).trim();
      updateCharCount();
      executePrediction();
    };

    state.recognition.onerror = (e) => {
      console.warn("Speech recognition error:", e.error);
      stopRecording();
    };

    state.recognition.onend = () => {
      stopRecording();
    };

    micBtn.addEventListener("click", () => {
      if (state.isRecording) {
        state.recognition.stop();
      } else {
        try {
          state.recognition.start();
        } catch (err) {
          console.warn("Speech start err", err);
        }
      }
    });

    function stopRecording() {
      state.isRecording = false;
      micBtn.classList.remove("recording");
      micBtn.title = "Speak your emotion (Speech-to-Text)";
    }
  }

  // Batch Analysis & File Handling
  function setupBatchUpload() {
    // Drag & drop
    ["dragenter", "dragover"].forEach(name => {
      dropZone.addEventListener(name, (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
      });
    });

    ["dragleave", "drop"].forEach(name => {
      dropZone.addEventListener(name, (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
      });
    });

    dropZone.addEventListener("drop", (e) => {
      const files = e.dataTransfer.files;
      if (files.length) handleBatchFile(files[0]);
    });

    batchFileInput.addEventListener("change", (e) => {
      if (e.target.files.length) handleBatchFile(e.target.files[0]);
    });

    runBatchBtn.addEventListener("click", runBatchAnalysis);
    exportBatchCsvBtn.addEventListener("click", exportBatchCSV);
  }

  function handleBatchFile(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      batchTextarea.value = e.target.result;
    };
    reader.readAsText(file);
  }

  async function runBatchAnalysis() {
    const raw = batchTextarea.value.trim();
    if (!raw) {
      alert("Please enter or upload sentences to analyze.");
      return;
    }

    const lines = raw.split(/\r?\n/).map(l => l.trim()).filter(l => l.length > 0);
    if (!lines.length) return;

    runBatchBtn.disabled = true;
    runBatchBtn.textContent = `Analyzing ${lines.length} lines...`;

    try {
      const res = await fetch("/api/batch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          texts: lines.slice(0, 200), // Max 200 lines per batch
          model: batchModelSelect.value
        })
      });

      if (!res.ok) throw new Error("Batch request failed");
      const data = await res.json();
      currentBatchData = data;
      renderBatchResults(data);
    } catch (err) {
      console.error(err);
      alert("Batch analysis error: " + err.message);
    } finally {
      runBatchBtn.disabled = false;
      runBatchBtn.textContent = "Run Batch Analysis ⚡";
    }
  }

  function renderBatchResults(data) {
    batchAnalyticsArea.style.display = "block";

    // Distribution Grid
    batchDistGrid.innerHTML = "";
    data.distribution.forEach(item => {
      const card = document.createElement("div");
      card.className = "batch-dist-card";
      card.innerHTML = `
        <div style="font-size: 1.5rem;">${item.emoji}</div>
        <div style="font-weight: 700; text-transform: capitalize; color: ${item.color}">${item.emotion}</div>
        <div style="font-size: 1.2rem; font-weight: 800; color: #fff;">${item.count}</div>
        <div style="font-size: 0.75rem; color: var(--text-muted);">${item.percentage}%</div>
      `;
      batchDistGrid.appendChild(card);
    });

    // Table rows
    batchTableBody.innerHTML = "";
    data.results.forEach((item, index) => {
      const tr = document.createElement("tr");
      const pred = item.prediction;
      const dom = pred.is_comparison ? pred.consensus_emotion : pred.dominant_emotion;
      const emoji = pred.is_comparison ? pred.consensus_emoji : pred.emoji;
      const conf = pred.is_comparison ? pred.agreement_level : `${pred.confidence_pct}%`;
      const sentiment = pred.is_comparison ? "Multi" : pred.sentiment;

      tr.innerHTML = `
        <td style="color: var(--text-muted);">${index + 1}</td>
        <td style="max-width: 380px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${item.text}</td>
        <td>
          <span style="font-weight: 600;">${emoji} ${dom}</span>
        </td>
        <td>${conf}</td>
        <td><span style="font-size: 0.8rem; font-weight: 600;">${sentiment}</span></td>
      `;
      batchTableBody.appendChild(tr);
    });
  }

  function exportBatchCSV() {
    if (!currentBatchData || !currentBatchData.results) return;
    let csv = "Index,Text,Predicted Emotion,Confidence,Sentiment\n";
    currentBatchData.results.forEach((item, i) => {
      const pred = item.prediction;
      const dom = pred.is_comparison ? pred.consensus_emotion : pred.dominant_emotion;
      const conf = pred.is_comparison ? pred.agreement_level : `${pred.confidence_pct}%`;
      const sent = pred.is_comparison ? "Multi" : pred.sentiment;
      const textClean = `"${item.text.replace(/"/g, '""')}"`;
      csv += `${i + 1},${textClean},${dom},${conf},${sent}\n`;
    });

    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `emotion_batch_analysis_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  // History Management
  function setupHistoryDrawer() {
    openHistoryBtn.addEventListener("click", () => {
      renderHistoryList();
      historyDrawer.classList.add("active");
      historyBackdrop.classList.add("active");
    });

    const close = () => {
      historyDrawer.classList.remove("active");
      historyBackdrop.classList.remove("active");
    };

    closeHistoryBtn.addEventListener("click", close);
    historyBackdrop.addEventListener("click", close);

    clearHistoryBtn.addEventListener("click", () => {
      state.history = [];
      localStorage.removeItem("emotion_history");
      updateHistoryCount();
      renderHistoryList();
    });
  }

  function saveHistory(text, result) {
    const item = {
      id: Date.now(),
      text: text,
      emotion: result.is_comparison ? result.consensus_emotion : result.dominant_emotion,
      emoji: result.is_comparison ? result.consensus_emoji : result.emoji,
      confidence: result.is_comparison ? result.agreement_level : `${result.confidence_pct}%`,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    state.history.unshift(item);
    if (state.history.length > 50) state.history.pop();
    localStorage.setItem("emotion_history", JSON.stringify(state.history));
    updateHistoryCount();
  }

  function updateHistoryCount() {
    historyCount.textContent = state.history.length;
  }

  function renderHistoryList() {
    historyList.innerHTML = "";
    if (!state.history.length) {
      historyList.innerHTML = `
        <div style="text-align: center; color: var(--text-muted); padding: 2rem 0; font-size: 0.85rem;">
          No analysis history recorded yet.
        </div>
      `;
      return;
    }

    state.history.forEach(item => {
      const el = document.createElement("div");
      el.className = "history-item";
      el.innerHTML = `
        <div class="history-item-top">
          <span style="font-weight: 700; font-size: 0.85rem;">${item.emoji} ${item.emotion} (${item.confidence})</span>
          <span style="font-size: 0.7rem; color: var(--text-muted);">${item.timestamp}</span>
        </div>
        <div class="history-text">${item.text}</div>
      `;
      el.addEventListener("click", () => {
        textInput.value = item.text;
        updateCharCount();
        historyDrawer.classList.remove("active");
        historyBackdrop.classList.remove("active");
        switchTab("detector");
        executePrediction();
      });
      historyList.appendChild(el);
    });
  }

});
