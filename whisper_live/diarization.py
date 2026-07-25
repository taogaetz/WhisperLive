"""Lightweight online speaker labeling for completed transcription segments.

The speaker encoder is the official WeSpeaker ResNet34-LM ONNX export. Audio
features are extracted with kaldi-native-fbank, so this path does not require
PyTorch, torchaudio, pyannote, or a second CUDA runtime.
"""

import logging
import os

import numpy as np


DEFAULT_EMBEDDING_MODEL = "/opt/whisperlive/models/wespeaker-voxceleb-resnet34-LM.onnx"


def load_audio(file_path, sample_rate=16000):
    """Load an audio file as mono float32 PCM at the requested sample rate."""
    import av

    container = av.open(file_path)
    resampler = av.AudioResampler(format="flt", layout="mono", rate=sample_rate)
    chunks = []

    try:
        for frame in container.decode(audio=0):
            for resampled_frame in resampler.resample(frame):
                chunks.append(
                    resampled_frame.to_ndarray().reshape(-1).astype(np.float32)
                )
    finally:
        container.close()

    if not chunks:
        return np.array([], dtype=np.float32)
    return np.concatenate(chunks)


class _OnnxSpeakerModel:
    """Raw-audio adapter for the official feature-input WeSpeaker ONNX model."""

    def __init__(self, model_path):
        import kaldi_native_fbank as knf
        import onnxruntime as ort

        options = knf.FbankOptions()
        options.frame_opts.samp_freq = 16000
        options.frame_opts.frame_length_ms = 25
        options.frame_opts.frame_shift_ms = 10
        options.frame_opts.dither = 0.0
        options.frame_opts.snip_edges = True
        options.frame_opts.window_type = "hamming"
        options.mel_opts.num_bins = 80
        # kaldi-native-fbank 1.22 defaults to Slaney-style filters. WeSpeaker
        # was trained with Kaldi filters, so explicitly select that behavior.
        options.mel_opts.use_slaney_mel_scale = False
        options.mel_opts.norm = ""

        session_options = ort.SessionOptions()
        session_options.inter_op_num_threads = 1
        session_options.intra_op_num_threads = 1

        self._knf = knf
        self._fbank_options = options
        self._session = ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )

    def __call__(self, audio_np, sample_rate):
        if sample_rate != 16000:
            raise ValueError(
                f"Speaker embedding expects 16000 Hz audio, got {sample_rate} Hz"
            )

        audio = np.asarray(audio_np, dtype=np.float32).reshape(-1)
        fbank = self._knf.OnlineFbank(self._fbank_options)
        # torchaudio.compliance.kaldi expects 16-bit PCM scale even when its
        # tensor dtype is float. Match WeSpeaker's official infer_onnx.py.
        fbank.accept_waveform(sample_rate, (audio * 32768.0).tolist())
        if fbank.num_frames_ready == 0:
            return None

        features = np.stack(
            [fbank.get_frame(index) for index in range(fbank.num_frames_ready)]
        ).astype(np.float32, copy=False)
        # Cepstral mean normalization, without variance normalization.
        features -= features.mean(axis=0, keepdims=True)
        embedding = self._session.run(
            output_names=["embs"],
            input_feed={"feats": features[None, :, :]},
        )[0]
        return np.asarray(embedding, dtype=np.float32).reshape(-1)


class SpeakerDiarizer:
    """Real-time speaker diarization using speaker embeddings and online clustering.

    Each completed transcription segment's audio is passed through a speaker
    embedding model. The embedding is compared against known speakers using
    cosine similarity. If no match exceeds the threshold, a new speaker is
    created.

    Args:
        similarity_threshold (float): Minimum cosine similarity to match an
            existing speaker. Lower values merge speakers more aggressively.
            Default 0.45 for the WeSpeaker ONNX embedding space.
        max_speakers (int): Maximum number of distinct speakers to track.
            Once reached, new segments are assigned to the closest existing
            speaker. Default 10.
        min_new_speaker_seconds (float): Minimum segment duration allowed to
            create a new speaker cluster. Shorter turns use the closest
            established speaker because their embeddings are less stable.
        embedding_model (str): Path to the WeSpeaker ONNX model.
        hf_token (str or None): Retained for protocol compatibility; unused.
    """

    def __init__(
        self,
        similarity_threshold=0.45,
        max_speakers=10,
        embedding_model=None,
        hf_token=None,
        speaker_names=None,
        min_new_speaker_seconds=1.0,
    ):
        self.similarity_threshold = similarity_threshold
        self.max_speakers = max_speakers
        self.min_new_speaker_seconds = min_new_speaker_seconds
        self.speaker_names = list(speaker_names or [])
        self.speakers = {}  # speaker_id -> embedding (averaged)
        self._speaker_count = 0
        self._model = None
        self._embedding_model_name = (
            embedding_model
            or os.getenv("WHISPERLIVE_SPEAKER_MODEL")
            or DEFAULT_EMBEDDING_MODEL
        )
        self._hf_token = hf_token

    def _next_speaker_id(self):
        if self._speaker_count < len(self.speaker_names):
            return self.speaker_names[self._speaker_count]
        return f"SPEAKER_{self._speaker_count:02d}"

    def _load_model(self):
        """Lazy-load the embedding model on first use."""
        if self._model is not None:
            return
        if not os.path.isfile(self._embedding_model_name):
            raise FileNotFoundError(
                f"Speaker embedding model not found: {self._embedding_model_name}"
            )
        self._model = _OnnxSpeakerModel(self._embedding_model_name)
        logging.info(
            "Speaker embedding model loaded on CPU with ONNX Runtime: %s",
            self._embedding_model_name,
        )

    def _compute_embedding(self, audio_np, sample_rate=16000):
        """Compute a speaker embedding from an audio numpy array.

        Args:
            audio_np (np.ndarray): 1-D float32 audio samples.
            sample_rate (int): Sample rate of the audio.

        Returns:
            np.ndarray: Speaker embedding vector, or None if audio is too short.
        """
        self._load_model()
        if len(audio_np) < sample_rate * 0.3:
            return None
        embedding = self._model(audio_np, sample_rate)
        if embedding is None:
            return None
        norm = np.linalg.norm(embedding)
        if not np.isfinite(norm) or norm == 0:
            return None
        return embedding / norm

    @staticmethod
    def _cosine_similarity(a, b):
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b))

    def identify_speaker(self, audio_np, sample_rate=16000):
        """Identify or create a speaker from an audio segment.

        Args:
            audio_np (np.ndarray): 1-D float32 audio for the segment.
            sample_rate (int): Sample rate. Default 16000.

        Returns:
            str or None: Speaker label (e.g. "SPEAKER_00"), or None if
                the audio is too short to embed.
        """
        embedding = self._compute_embedding(audio_np, sample_rate)
        if embedding is None:
            return None

        best_speaker = None
        best_sim = -1.0

        for speaker_id, stored_emb in self.speakers.items():
            sim = self._cosine_similarity(embedding, stored_emb)
            if sim > best_sim:
                best_sim = sim
                best_speaker = speaker_id

        if (
            best_speaker is not None
            and len(audio_np) < sample_rate * self.min_new_speaker_seconds
        ):
            return best_speaker

        if best_sim >= self.similarity_threshold:
            # Update running average for the matched speaker
            self.speakers[best_speaker] = (
                self.speakers[best_speaker] * 0.9 + embedding * 0.1
            )
            # Re-normalize
            self.speakers[best_speaker] /= np.linalg.norm(self.speakers[best_speaker])
            return best_speaker

        if len(self.speakers) >= self.max_speakers:
            # Assign to closest speaker
            return (
                best_speaker if best_speaker else f"SPEAKER_{self._speaker_count:02d}"
            )

        # Create a new speaker
        speaker_id = self._next_speaker_id()
        self._speaker_count += 1
        self.speakers[speaker_id] = embedding
        return speaker_id

    def enroll_speaker(self, speaker_name, audio_np, sample_rate=16000):
        """Enroll a known speaker from reference audio."""
        embedding = self._compute_embedding(audio_np, sample_rate)
        if embedding is None:
            return False
        self.speakers[speaker_name] = embedding
        return True

    def reset(self):
        """Reset all speaker state."""
        self.speakers.clear()
        self._speaker_count = 0
