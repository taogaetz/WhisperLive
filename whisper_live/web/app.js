const elements = {
  serverPill: document.querySelector("#serverPill"),
  serverSummary: document.querySelector("#serverSummary"),
  recordButton: document.querySelector("#recordButton"),
  recordLabel: document.querySelector("#recordLabel"),
  permissionHint: document.querySelector("#permissionHint"),
  micOrbit: document.querySelector("#micOrbit"),
  levelMeter: document.querySelector("#levelMeter"),
  levelValue: document.querySelector("#levelValue"),
  diarizationToggle: document.querySelector("#diarizationToggle"),
  modelValue: document.querySelector("#modelValue"),
  deviceValue: document.querySelector("#deviceValue"),
  computeValue: document.querySelector("#computeValue"),
  copyButton: document.querySelector("#copyButton"),
  clearButton: document.querySelector("#clearButton"),
  streamState: document.querySelector("#streamState"),
  elapsedValue: document.querySelector("#elapsedValue"),
  latencyValue: document.querySelector("#latencyValue"),
  eventValue: document.querySelector("#eventValue"),
  errorBanner: document.querySelector("#errorBanner"),
  transcriptScroll: document.querySelector("#transcriptScroll"),
  emptyState: document.querySelector("#emptyState"),
  segmentsList: document.querySelector("#segmentsList"),
  partialCard: document.querySelector("#partialCard"),
  partialText: document.querySelector("#partialText"),
  eventLog: document.querySelector("#eventLog"),
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
  startedAt: 0,
  sentSamples: 0,
  latestSegmentEnd: 0,
  eventCount: 0,
  segments: new Map(),
  partial: null,
  events: [],
  timer: null,
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

function formatClock(seconds, milliseconds = false) {
  const safeSeconds = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(safeSeconds / 60);
  const remainder = safeSeconds % 60;
  if (milliseconds) {
    return `${String(minutes).padStart(2, "0")}:${remainder
      .toFixed(3)
      .padStart(6, "0")}`;
  }
  return `${String(minutes).padStart(2, "0")}:${String(
    Math.floor(remainder),
  ).padStart(2, "0")}`;
}

function speakerName(segment) {
  const raw = segment.speaker ?? segment.speaker_id;
  if (raw === undefined || raw === null || raw === "") return "Speaker";
  const text = String(raw).replace(/^speaker[_\s-]*/i, "");
  return /^\d+$/.test(text) ? `Speaker ${Number(text) + 1}` : `Speaker ${text}`;
}

function speakerIndex(segment) {
  const name = speakerName(segment);
  let hash = 0;
  for (const character of name) hash = (hash * 31 + character.charCodeAt(0)) | 0;
  return Math.abs(hash) % 6;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function setServerState(kind, summary) {
  elements.serverPill.dataset.state = kind;
  elements.serverSummary.textContent = summary;
}

function setStreamState(label, kind = "idle") {
  elements.streamState.textContent = label;
  elements.streamState.dataset.state = kind;
}

function showError(message) {
  elements.errorBanner.textContent = message;
  elements.errorBanner.hidden = false;
}

function clearError() {
  elements.errorBanner.hidden = true;
  elements.errorBanner.textContent = "";
}

function renderTranscript() {
  const completed = [...state.segments.values()].sort(
    (left, right) => Number(left.start) - Number(right.start),
  );
  elements.emptyState.hidden = completed.length > 0 || Boolean(state.partial);
  elements.segmentsList.innerHTML = completed
    .map((segment) => {
      const speaker = speakerName(segment);
      return `
        <li class="segment speaker-${speakerIndex(segment)}">
          <time>${formatClock(segment.start, true)}</time>
          <div>
            <span class="speaker-label">${escapeHtml(speaker)}</span>
            <p>${escapeHtml(segment.text?.trim() || "")}</p>
          </div>
        </li>`;
    })
    .join("");

  const partialText = state.partial?.text?.trim();
  elements.partialText.textContent =
    partialText || (state.running ? "Listening for speech…" : "Waiting for speech…");
  elements.partialCard.classList.toggle("has-text", Boolean(partialText));

  if (completed.length || partialText) {
    requestAnimationFrame(() => {
      elements.transcriptScroll.scrollTop = elements.transcriptScroll.scrollHeight;
    });
  }
}

function ingestSegments(segments) {
  let newestPartial = null;
  for (const segment of segments) {
    state.latestSegmentEnd = Math.max(
      state.latestSegmentEnd,
      Number(segment.end) || 0,
    );
    if (segment.completed) {
      const key = `${segment.start}-${segment.end}`;
      state.segments.set(key, { ...state.segments.get(key), ...segment });
    } else {
      newestPartial = segment;
    }
  }
  state.partial = newestPartial;
  renderTranscript();
  updateLag();
}

function appendEvent(payload) {
  state.eventCount += 1;
  state.events.unshift({
    at: new Date().toISOString(),
    payload,
  });
  state.events = state.events.slice(0, 80);
  elements.eventValue.textContent = String(state.eventCount);
  elements.eventLog.textContent = state.events
    .map((event) => `${event.at}\n${JSON.stringify(event.payload, null, 2)}`)
    .join("\n\n");
}

function updateLag() {
  if (!state.running || !state.latestSegmentEnd) {
    elements.latencyValue.textContent = "—";
    return;
  }
  const audioSeconds = state.sentSamples / (state.server?.sample_rate || 16000);
  const lag = Math.max(0, audioSeconds - state.latestSegmentEnd);
  elements.latencyValue.textContent =
    lag < 1 ? `${Math.round(lag * 1000)} ms` : `${lag.toFixed(1)} s`;
}

function updateTimer() {
  if (!state.running) return;
  const elapsed = (performance.now() - state.startedAt) / 1000;
  elements.elapsedValue.textContent = formatClock(elapsed);
  updateLag();
}

function setInputLevel(chunk) {
  let squareSum = 0;
  for (const sample of chunk) squareSum += sample * sample;
  const rms = Math.sqrt(squareSum / Math.max(1, chunk.length));
  const db = Math.max(-60, 20 * Math.log10(Math.max(rms, 0.001)));
  const percent = Math.max(0, Math.min(100, ((db + 60) / 60) * 100));
  elements.levelMeter.style.width = `${percent}%`;
  elements.levelValue.textContent = `${Math.round(db)} dB`;
}

async function loadStatus() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    if (!response.ok) throw new Error(`Status request returned ${response.status}`);
    state.server = await response.json();
    elements.modelValue.textContent = state.server.model;
    elements.deviceValue.textContent = state.server.device.toUpperCase();
    elements.computeValue.textContent = state.server.compute_type.toUpperCase();
    elements.diarizationToggle.disabled = !state.server.diarization_available;
    if (!state.server.diarization_available) {
      elements.diarizationToggle.checked = false;
    }
    setServerState("ready", `${state.server.device.toUpperCase()} · READY`);
    elements.recordButton.disabled = false;
  } catch (error) {
    setServerState("error", "Server unavailable");
    showError(`Could not reach WhisperLive: ${error.message}`);
  }
}

function websocketUrl() {
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  const host = location.hostname || "127.0.0.1";
  return `${protocol}//${host}:${state.server.websocket_port}`;
}

function handleServerMessage(event) {
  let payload;
  try {
    payload = JSON.parse(event.data);
  } catch {
    appendEvent({ status: "INVALID_JSON", value: String(event.data) });
    return;
  }
  appendEvent(payload);

  // WhisperLive uses `message` for SERVER_READY and `status` for later
  // lifecycle events. Normalize both shapes before gating microphone audio.
  const messageType = payload.status || payload.message;
  if (messageType === "SERVER_READY") {
    state.serverReady = true;
    setStreamState("LIVE", "live");
    elements.permissionHint.textContent = "Streaming 16 kHz audio to the local GPU.";
  } else if (messageType === "WAIT") {
    setStreamState("QUEUED", "waiting");
    elements.permissionHint.textContent = `GPU busy · estimated wait ${Number(payload.message).toFixed(1)} min`;
  } else if (messageType === "WARNING") {
    showError(payload.message || "The server returned a warning.");
  } else if (messageType === "ERROR") {
    showError(payload.message || "The server returned an error.");
  } else if (messageType === "DISCONNECT") {
    showError(payload.message || "The server ended this session.");
    stopSession();
  }

  if (Array.isArray(payload.segments)) ingestSegments(payload.segments);
}

async function configureAudio() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("Microphone capture is unavailable in this browser or context.");
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
    "whisperlive-pcm-capture",
  );
  state.silentGain = state.audioContext.createGain();
  state.silentGain.gain.value = 0;
  state.source.connect(state.worklet);
  state.worklet.connect(state.silentGain);
  state.silentGain.connect(state.audioContext.destination);

  state.worklet.port.onmessage = ({ data }) => {
    setInputLevel(data);
    if (!state.serverReady || state.socket?.readyState !== WebSocket.OPEN) return;
    const resampled = state.resampler.process(data);
    if (resampled.length) {
      state.socket.send(resampled.buffer);
      state.sentSamples += resampled.length;
    }
  };
}

async function startSession() {
  if (state.running) return;
  clearError();
  state.running = true;
  state.stopping = false;
  state.serverReady = false;
  state.startedAt = performance.now();
  state.sentSamples = 0;
  state.latestSegmentEnd = 0;
  state.eventCount = 0;
  state.events = [];
  elements.eventValue.textContent = "0";
  elements.eventLog.textContent = "Waiting for server events…";
  elements.recordButton.disabled = true;
  setStreamState("CONNECTING", "waiting");

  try {
    await configureAudio();
    state.socket = new WebSocket(websocketUrl());
    state.socket.addEventListener("open", () => {
      state.socket.send(
        JSON.stringify({
          uid: crypto.randomUUID(),
          language: "en",
          task: "transcribe",
          model: "small.en",
          use_vad: true,
          enable_diarization: elements.diarizationToggle.checked,
          send_last_n_segments: 100,
        }),
      );
    });
    state.socket.addEventListener("message", handleServerMessage);
    state.socket.addEventListener("error", () => {
      showError(`Could not open ${websocketUrl()}. Is port ${state.server.websocket_port} reachable?`);
    });
    state.socket.addEventListener("close", () => {
      if (state.running && !state.stopping) {
        showError("The streaming connection closed unexpectedly.");
        stopSession();
      }
    });

    state.timer = window.setInterval(updateTimer, 250);
    elements.recordButton.classList.add("stop");
    elements.recordLabel.textContent = "Stop session";
    elements.recordButton.disabled = false;
    elements.micOrbit.classList.add("active");
    elements.diarizationToggle.disabled = true;
    renderTranscript();
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
  window.clearInterval(state.timer);
  state.timer = null;

  if (state.socket?.readyState === WebSocket.OPEN) {
    state.socket.send(new TextEncoder().encode("END_OF_AUDIO"));
    await new Promise((resolve) => window.setTimeout(resolve, 120));
    state.socket.close(1000, "Session ended");
  }
  state.worklet?.disconnect();
  state.source?.disconnect();
  state.silentGain?.disconnect();
  state.mediaStream?.getTracks().forEach((track) => track.stop());
  if (state.audioContext && state.audioContext.state !== "closed") {
    await state.audioContext.close();
  }

  state.socket = null;
  state.worklet = null;
  state.source = null;
  state.silentGain = null;
  state.mediaStream = null;
  state.audioContext = null;
  state.resampler = null;
  elements.recordButton.classList.remove("stop");
  elements.recordLabel.textContent = "Start microphone";
  elements.recordButton.disabled = !state.server;
  elements.micOrbit.classList.remove("active");
  elements.diarizationToggle.disabled = !state.server?.diarization_available;
  elements.levelMeter.style.width = "0%";
  elements.levelValue.textContent = "— dB";
  elements.permissionHint.textContent = "Your browser will ask for microphone permission.";
  setStreamState("IDLE");
  renderTranscript();
  state.stopping = false;
}

function clearTranscript() {
  state.segments.clear();
  state.partial = null;
  state.latestSegmentEnd = 0;
  renderTranscript();
}

async function copyTranscript() {
  const text = [...state.segments.values()]
    .sort((left, right) => Number(left.start) - Number(right.start))
    .map(
      (segment) =>
        `[${formatClock(segment.start, true)}] ${speakerName(segment)}: ${segment.text?.trim()}`,
    )
    .join("\n");
  if (!text) return;
  await navigator.clipboard.writeText(text);
  elements.copyButton.textContent = "Copied";
  window.setTimeout(() => {
    elements.copyButton.textContent = "Copy";
  }, 1200);
}

elements.recordButton.addEventListener("click", () => {
  if (state.running) stopSession();
  else startSession();
});
elements.clearButton.addEventListener("click", clearTranscript);
elements.copyButton.addEventListener("click", () => {
  copyTranscript().catch((error) => showError(`Copy failed: ${error.message}`));
});
window.addEventListener("beforeunload", () => {
  state.mediaStream?.getTracks().forEach((track) => track.stop());
  state.socket?.close();
});

renderTranscript();
loadStatus();
