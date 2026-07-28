import { buildUtterances } from "./utterances.mjs";

const elements = {
  statusDot: document.querySelector("#statusDot"),
  serverStatus: document.querySelector("#serverStatus"),
  liveNav: document.querySelector("#liveNav"),
  liveState: document.querySelector("#liveState"),
  historyList: document.querySelector("#historyList"),
  refreshHistory: document.querySelector("#refreshHistory"),
  viewTitle: document.querySelector("#viewTitle"),
  gpuPowerValue: document.querySelector("#gpuPowerValue"),
  gpuMemoryValue: document.querySelector("#gpuMemoryValue"),
  diarizationToggle: document.querySelector("#diarizationToggle"),
  copyButton: document.querySelector("#copyButton"),
  recordButton: document.querySelector("#recordButton"),
  recordLabel: document.querySelector("#recordLabel"),
  errorBanner: document.querySelector("#errorBanner"),
  transcriptScroll: document.querySelector("#transcriptScroll"),
  emptyState: document.querySelector("#emptyState"),
  segmentsList: document.querySelector("#segmentsList"),
  partialCard: document.querySelector("#partialCard"),
  partialText: document.querySelector("#partialText"),
};

const state = {
  server: null,
  socket: null,
  mediaStream: null,
  audioContext: null,
  source: null,
  worklet: null,
  silentGain: null,
  resampler: null,
  serverReady: false,
  running: false,
  stopping: false,
  segments: new Map(),
  partial: null,
  selectedSession: null,
  history: [],
  telemetryTimer: null,
  devEvents: null,
};

class StreamingResampler {
  constructor(inputRate, outputRate) {
    this.ratio = inputRate / outputRate;
    this.position = 0;
    this.pending = new Float32Array(0);
  }

  process(chunk) {
    const input = new Float32Array(this.pending.length + chunk.length);
    input.set(this.pending);
    input.set(chunk, this.pending.length);

    const samples = [];
    while (this.position + 1 < input.length) {
      const left = Math.floor(this.position);
      const fraction = this.position - left;
      samples.push(
        input[left] + (input[left + 1] - input[left]) * fraction,
      );
      this.position += this.ratio;
    }

    if (this.position >= input.length) {
      this.position -= input.length;
      this.pending = new Float32Array(0);
    } else {
      const keepFrom = Math.floor(this.position);
      this.pending = input.slice(keepFrom);
      this.position -= keepFrom;
    }
    return Float32Array.from(samples);
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatClock(seconds) {
  const safeSeconds = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(safeSeconds / 60);
  const remainder = safeSeconds % 60;
  return `${String(minutes).padStart(2, "0")}:${remainder
    .toFixed(1)
    .padStart(4, "0")}`;
}

function formatDuration(seconds) {
  const value = Math.max(0, Math.round(Number(seconds) || 0));
  const minutes = Math.floor(value / 60);
  const remainder = value % 60;
  return minutes ? `${minutes}m ${remainder}s` : `${remainder}s`;
}

function formatDate(value, detailed = false) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Unknown";
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    ...(detailed ? { year: "numeric" } : {}),
    hour: "numeric",
    minute: "2-digit",
  }).format(date);
}

function speakerName(segment) {
  const raw = segment.speaker ?? segment.speaker_id;
  if (raw === undefined || raw === null || raw === "") return "Speaker";
  const text = String(raw).replace(/^speaker[_\s-]*/i, "");
  return /^\d+$/.test(text) ? `Speaker ${Number(text) + 1}` : `Speaker ${text}`;
}

function showError(message) {
  elements.errorBanner.textContent = message;
  elements.errorBanner.hidden = false;
}

function clearError() {
  elements.errorBanner.hidden = true;
  elements.errorBanner.textContent = "";
}

function liveSegments() {
  return [...state.segments.values()].sort(
    (left, right) => Number(left.start) - Number(right.start),
  );
}

function visibleSegments() {
  return state.selectedSession?.segments || liveSegments();
}

function renderTranscript() {
  const completed = buildUtterances(visibleSegments());
  const isLive = state.selectedSession === null;
  const partialText = isLive ? state.partial?.text?.trim() : "";

  elements.viewTitle.textContent = isLive
    ? "Live"
    : formatDate(state.selectedSession.started_at, true);
  elements.liveNav.classList.toggle("selected", isLive);
  for (const item of elements.historyList.querySelectorAll(".history-item")) {
    item.classList.toggle(
      "selected",
      item.dataset.sessionId === state.selectedSession?.id,
    );
  }

  elements.segmentsList.innerHTML = completed
    .map(
      (segment) => `
        <li class="segment">
          <time>${formatClock(segment.start)}</time>
          <div>
            <span class="speaker">${escapeHtml(speakerName(segment))}</span>
            <p>${escapeHtml(segment.text?.trim() || "")}</p>
          </div>
        </li>`,
    )
    .join("");

  elements.partialText.textContent = partialText || "";
  elements.partialCard.hidden = !partialText;
  elements.emptyState.hidden = completed.length > 0 || Boolean(partialText);
  elements.copyButton.disabled = completed.length === 0 && !partialText;

  if (completed.length || partialText) {
    requestAnimationFrame(() => {
      elements.transcriptScroll.scrollTop = elements.transcriptScroll.scrollHeight;
    });
  }
}

function renderHistory() {
  elements.historyList.replaceChildren();
  if (!state.server?.history_enabled) {
    const empty = document.createElement("p");
    empty.className = "history-empty";
    empty.textContent = "Disabled";
    elements.historyList.append(empty);
    return;
  }
  if (!state.history.length) {
    const empty = document.createElement("p");
    empty.className = "history-empty";
    empty.textContent = "No logs";
    elements.historyList.append(empty);
    return;
  }

  for (const session of state.history) {
    const item = document.createElement("button");
    item.type = "button";
    item.className = "history-item";
    item.dataset.sessionId = session.id;
    item.classList.toggle("selected", session.id === state.selectedSession?.id);

    const title = document.createElement("strong");
    title.textContent = formatDate(session.started_at);
    const meta = document.createElement("span");
    const stateLabel = session.status === "active" ? "Live" : formatDuration(session.duration_seconds);
    meta.textContent = session.preview
      ? `${stateLabel} · ${session.preview}`
      : stateLabel;
    item.append(title, meta);
    item.addEventListener("click", () => {
      selectHistory(session.id);
    });
    elements.historyList.append(item);
  }
}

async function loadHistory() {
  if (!state.server?.history_enabled) {
    state.history = [];
    renderHistory();
    return;
  }
  try {
    const response = await fetch("/api/history?limit=100", { cache: "no-store" });
    if (!response.ok) throw new Error(`History returned ${response.status}`);
    const payload = await response.json();
    state.history = payload.sessions || [];
    renderHistory();
  } catch (error) {
    showError(error.message);
  }
}

async function selectHistory(sessionId) {
  clearError();
  try {
    const response = await fetch(`/api/history/${encodeURIComponent(sessionId)}`, {
      cache: "no-store",
    });
    if (!response.ok) throw new Error(`Log returned ${response.status}`);
    state.selectedSession = await response.json();
    renderTranscript();
  } catch (error) {
    showError(error.message);
  }
}

function showLive() {
  state.selectedSession = null;
  renderTranscript();
}

function ingestSegments(segments) {
  let newestPartial = null;
  for (const segment of segments) {
    if (segment.completed) {
      const key = `${segment.start}-${segment.end}`;
      state.segments.set(key, { ...state.segments.get(key), ...segment });
    } else {
      newestPartial = segment;
    }
  }
  state.partial = newestPartial;
  if (state.selectedSession === null) renderTranscript();
}

function renderTelemetry(payload) {
  const gpu = payload.gpu || {};
  if (!gpu.available) {
    elements.gpuPowerValue.textContent = "—";
    elements.gpuMemoryValue.textContent = "—";
    return;
  }
  const used = Number(gpu.memory_used_mib) || 0;
  const total = Number(gpu.memory_total_mib) || 0;
  elements.gpuPowerValue.textContent =
    gpu.power_w == null ? "—" : `${Math.round(gpu.power_w)} W`;
  elements.gpuMemoryValue.textContent =
    `${(used / 1024).toFixed(1)} / ${(total / 1024).toFixed(1)} GB`;
}

async function loadTelemetry() {
  try {
    const response = await fetch("/api/telemetry", { cache: "no-store" });
    if (response.ok) renderTelemetry(await response.json());
  } catch {
    // Server status already communicates connectivity.
  }
}

function enableDevReload() {
  if (!state.server?.dev_mode || !window.EventSource || state.devEvents) return;
  state.devEvents = new EventSource("/api/dev/events");
  state.devEvents.addEventListener("reload", () => window.location.reload());
}

async function loadStatus() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    if (!response.ok) throw new Error(`Status returned ${response.status}`);
    state.server = await response.json();
    elements.statusDot.className = "status-dot ready";
    elements.serverStatus.textContent = state.server.model;
    elements.recordButton.disabled = false;
    elements.diarizationToggle.disabled = !state.server.diarization_available;
    if (!state.server.diarization_available) {
      elements.diarizationToggle.checked = false;
    }
    enableDevReload();
    await Promise.all([loadTelemetry(), loadHistory()]);
    state.telemetryTimer = window.setInterval(loadTelemetry, 1000);
  } catch (error) {
    elements.statusDot.className = "status-dot error";
    elements.serverStatus.textContent = "Offline";
    showError(error.message);
  }
}

function websocketUrl() {
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  const host = location.hostname || "127.0.0.1";
  return `${protocol}//${host}:${state.server.websocket_port}`;
}

function sameOutputThreshold() {
  return String(state.server?.model).toLowerCase().includes("turbo") ? 2 : 10;
}

function handleServerMessage(event) {
  let payload;
  try {
    payload = JSON.parse(event.data);
  } catch {
    return;
  }
  const messageType = payload.status || payload.message;
  if (messageType === "SERVER_READY") {
    state.serverReady = true;
    elements.liveState.textContent = "Listening";
    loadHistory();
  } else if (messageType === "WAIT") {
    elements.liveState.textContent = "Busy";
  } else if (messageType === "WARNING" || messageType === "ERROR") {
    showError(payload.message || messageType);
  } else if (messageType === "DISCONNECT") {
    stopSession();
  }
  if (Array.isArray(payload.segments)) ingestSegments(payload.segments);
}

async function configureAudio() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("Microphone unavailable");
  }
  state.mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  });
  state.audioContext = new AudioContext({ latencyHint: "interactive" });
  await state.audioContext.audioWorklet.addModule("/ui/audio-worklet.js");
  await state.audioContext.resume();

  state.resampler = new StreamingResampler(
    state.audioContext.sampleRate,
    state.server.sample_rate,
  );
  state.source = state.audioContext.createMediaStreamSource(state.mediaStream);
  state.worklet = new AudioWorkletNode(
    state.audioContext,
    "pascalscribe-pcm-capture",
  );
  state.silentGain = state.audioContext.createGain();
  state.silentGain.gain.value = 0;
  state.source.connect(state.worklet);
  state.worklet.connect(state.silentGain);
  state.silentGain.connect(state.audioContext.destination);

  state.worklet.port.onmessage = ({ data }) => {
    if (!state.serverReady || state.socket?.readyState !== WebSocket.OPEN) return;
    const resampled = state.resampler.process(data);
    if (resampled.length) state.socket.send(resampled.buffer);
  };
}

function clearLiveTranscript() {
  state.segments.clear();
  state.partial = null;
}

async function startSession() {
  if (state.running) return;
  clearError();
  clearLiveTranscript();
  state.selectedSession = null;
  state.running = true;
  state.stopping = false;
  state.serverReady = false;
  elements.liveState.textContent = "Connecting";
  elements.recordButton.disabled = true;
  renderTranscript();

  try {
    await configureAudio();
    const socket = new WebSocket(websocketUrl());
    state.socket = socket;
    socket.addEventListener("open", () => {
      if (state.socket !== socket) return;
      socket.send(
        JSON.stringify({
          uid: crypto.randomUUID(),
          language: "en",
          task: "transcribe",
          model: state.server.model,
          use_vad: true,
          enable_diarization: elements.diarizationToggle.checked,
          same_output_threshold: sameOutputThreshold(),
          send_last_n_segments: 100,
        }),
      );
    });
    socket.addEventListener("message", (event) => {
      if (state.socket !== socket) return;
      handleServerMessage(event);
    });
    socket.addEventListener("error", () => {
      if (state.socket !== socket) return;
      showError("WebSocket unavailable");
    });
    socket.addEventListener("close", () => {
      if (state.socket !== socket) return;
      if (state.running && !state.stopping) stopSession();
    });

    elements.recordButton.classList.add("stop");
    elements.recordLabel.textContent = "Stop";
    elements.recordButton.disabled = false;
    elements.diarizationToggle.disabled = true;
  } catch (error) {
    showError(error.message);
    await stopSession();
  }
}

async function stopSession() {
  if (state.stopping) return;
  state.stopping = true;
  state.running = false;
  state.serverReady = false;
  elements.recordButton.disabled = true;

  const socket = state.socket;
  state.socket = null;
  if (socket?.readyState === WebSocket.OPEN) {
    socket.send(new TextEncoder().encode("END_OF_AUDIO"));
    await new Promise((resolve) => window.setTimeout(resolve, 120));
    socket.close(1000, "Session ended");
  } else if (socket?.readyState === WebSocket.CONNECTING) {
    socket.close();
  }

  state.worklet?.disconnect();
  state.source?.disconnect();
  state.silentGain?.disconnect();
  state.mediaStream?.getTracks().forEach((track) => track.stop());
  if (state.audioContext && state.audioContext.state !== "closed") {
    await state.audioContext.close();
  }

  state.worklet = null;
  state.source = null;
  state.silentGain = null;
  state.mediaStream = null;
  state.audioContext = null;
  state.resampler = null;
  elements.recordButton.classList.remove("stop");
  elements.recordLabel.textContent = "Start";
  elements.recordButton.disabled = !state.server;
  elements.diarizationToggle.disabled = !state.server?.diarization_available;
  elements.liveState.textContent = "Idle";
  state.stopping = false;
  window.setTimeout(loadHistory, 400);
}

async function copyTranscript() {
  const text = buildUtterances(visibleSegments())
    .map(
      (segment) =>
        `[${formatClock(segment.start)}] ${speakerName(segment)}: ${segment.text?.trim()}`,
    )
    .join("\n");
  if (!text) return;
  await navigator.clipboard.writeText(text);
  elements.copyButton.textContent = "Copied";
  window.setTimeout(() => {
    elements.copyButton.textContent = "Copy";
  }, 1000);
}

elements.liveNav.addEventListener("click", showLive);
elements.refreshHistory.addEventListener("click", loadHistory);
elements.recordButton.addEventListener("click", () => {
  if (state.running) stopSession();
  else startSession();
});
elements.copyButton.addEventListener("click", () => {
  copyTranscript().catch((error) => showError(error.message));
});
window.addEventListener("beforeunload", () => {
  window.clearInterval(state.telemetryTimer);
  state.devEvents?.close();
  state.mediaStream?.getTracks().forEach((track) => track.stop());
  state.socket?.close();
});

renderTranscript();
loadStatus();
