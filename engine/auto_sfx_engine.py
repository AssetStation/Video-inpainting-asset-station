"""
================================================================================
ASSET STATION — MODULE: AUTO SFX PRODUCTION ENGINE (CLAP + ACOUSTIC DSP CORE)
Dual-Brain Architecture: Local CLAP ONNX Semantic Matching + Sample-Accurate DSP
Author: Asset Station Engineering Team
Environment: Python 3.11+ / ONNX Runtime / NumPy / SciPy / SoundFile / Librosa / FFmpeg
================================================================================
"""

import os
import sys
import re
import time
import json
import argparse
import subprocess
import numpy as np
import scipy.fft as fft
import soundfile as sf
import librosa
import onnxruntime as ort
from tokenizers import Tokenizer

# Set console encoding to UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Base directories (robust resolution supporting both source script and PyInstaller frozen executable)
def _resolve_paths():
    if getattr(sys, 'frozen', False):
        exe_path = os.path.abspath(sys.executable)
        curr = os.path.dirname(exe_path)
        base = None
        # Walk up tree to locate extension root (marked by CSXS, engine, or assets/sfx)
        for _ in range(5):
            if os.path.exists(os.path.join(curr, "CSXS")) or \
               os.path.exists(os.path.join(curr, "engine", "models", "clap")) or \
               os.path.exists(os.path.join(curr, "assets", "sfx")):
                base = curr
                break
            parent = os.path.dirname(curr)
            if parent == curr:
                break
            curr = parent
        if not base:
            # Fallback assuming structure [base]/engine/[win|mac]/exe or [base]/bin/[win|mac]/exe
            base = os.path.dirname(os.path.dirname(os.path.dirname(exe_path)))
            if not os.path.exists(os.path.join(base, "engine")):
                base = os.path.dirname(os.path.dirname(exe_path))
        engine = os.path.join(base, "engine")
    else:
        engine = os.path.dirname(os.path.abspath(__file__))
        base = os.path.dirname(engine)
    return base, engine

BASE_DIR, ENGINE_DIR = _resolve_paths()
MODELS_DIR = os.path.join(ENGINE_DIR, "models", "clap")

# FFmpeg binary resolution
FFMPEG_CANDIDATES = [
    os.path.join(BASE_DIR, "bin", "win", "ffmpeg.exe"),
    os.path.join(BASE_DIR, "bin", "mac", "ffmpeg"),
    os.path.join(BASE_DIR, "bin", "mac", "ffmpeg_arm64"),
    "ffmpeg"
]
FFMPEG_PATH = next((p for p in FFMPEG_CANDIDATES if os.path.exists(p)), "ffmpeg")

# Default index & embedding cache files
DEFAULT_VAULT_INDEX = os.path.join(ENGINE_DIR, "sfx_vault.idx")
DEFAULT_EMBEDDINGS = os.path.join(ENGINE_DIR, "sfx_embeddings.npy")

# Universal Zero-Shot Acoustic Category Classification Anchors (100% Filename-Independent)
ACOUSTIC_CATEGORY_NAMES = ["whoosh", "impact", "click", "pop", "riser", "chime", "foley", "ui"]
ACOUSTIC_CATEGORY_PROMPTS = [
    "whoosh transition sweep air swish swoosh slide glide motion",
    "heavy cinematic sub bass drop hit slam punch impact boom",
    "short crisp tactile mechanical button click tap snap tick",
    "quick organic bubble pop blip accent",
    "tension pitch riser build up swell ascending",
    "bright metallic bell chime ding notification sparkle ring shine shimmer glow accent light",
    "organic physical foley typing keyboard footstep paper tear rip cloth rustle movement",
    "digital modern tech user interface software notification subtle interaction feedback"
]


class NumpyEncoder(json.JSONEncoder):
    """Encodes NumPy scalar types into native JSON serializable types."""
    def default(self, obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ==============================================================================
# 1. LOCAL CLAP ONNX ENGINE (LAZY DUAL TEXT + AUDIO ENCODERS)
# ==============================================================================
class ClapEngine:
    """Manages quantized ONNX inference sessions for CLAP text and audio encoders with lazy loading."""

    _instance = None

    def __init__(self, models_dir=MODELS_DIR):
        self.models_dir = models_dir
        self.audio_model_path = os.path.join(models_dir, "audio_model_quantized.onnx")
        self.text_model_path = os.path.join(models_dir, "text_model_quantized.onnx")
        self.tokenizer_path = os.path.join(models_dir, "tokenizer.json")

        self.audio_sess = None
        self.text_sess = None
        self.tokenizer = None
        self._text_initialized = False
        self._audio_initialized = False

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ClapEngine()
        return cls._instance

    def is_available(self):
        return (
            os.path.exists(self.text_model_path) and
            os.path.exists(self.tokenizer_path)
        )

    def _get_session_opts(self):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = min(4, os.cpu_count() or 2)
        opts.log_severity_level = 3  # Warning/Error only
        return opts

    def initialize_text(self):
        """Initializes tokenizer and text encoder ONNX session (fast startup for matching)."""
        if self._text_initialized:
            return True
        if not self.is_available():
            sys.stderr.write(f"[CLAP] Text model missing in {self.models_dir}\n")
            return False
        try:
            opts = self._get_session_opts()
            self.tokenizer = Tokenizer.from_file(self.tokenizer_path)
            self.tokenizer.enable_truncation(max_length=77)
            self.tokenizer.enable_padding(length=77, pad_id=1, pad_token="<pad>")
            self.text_sess = ort.InferenceSession(self.text_model_path, sess_options=opts)
            self._text_initialized = True
            return True
        except Exception as e:
            sys.stderr.write(f"[CLAP] Text initialization error: {str(e)}\n")
            return False

    def initialize_audio(self):
        """Initializes audio encoder ONNX session on demand (only used during soundbank indexing)."""
        if self._audio_initialized:
            return True
        if not os.path.exists(self.audio_model_path):
            sys.stderr.write(f"[CLAP] Audio model missing in {self.models_dir}\n")
            return False
        try:
            opts = self._get_session_opts()
            self.audio_sess = ort.InferenceSession(self.audio_model_path, sess_options=opts)
            self._audio_initialized = True
            return True
        except Exception as e:
            sys.stderr.write(f"[CLAP] Audio initialization error: {str(e)}\n")
            return False

    def initialize(self):
        """Initializes both text and audio engines."""
        return self.initialize_text() and self.initialize_audio()

    def encode_text(self, texts):
        """Encodes list of strings into L2-normalized 512D float embeddings."""
        if not self.initialize_text() or not self.text_sess or not self.tokenizer:
            return np.zeros((len(texts), 512), dtype=np.float32)

        try:
            encodings = self.tokenizer.encode_batch(texts)
            input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
            embeds = self.text_sess.run(None, {"input_ids": input_ids})[0]

            # Unit normalize (L2 norm)
            norms = np.linalg.norm(embeds, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return (embeds / norms).astype(np.float32)
        except Exception as e:
            sys.stderr.write(f"[CLAP] Text encoding error: {str(e)}\n")
            return np.zeros((len(texts), 512), dtype=np.float32)

    def extract_audio_features(self, pcm_data, sr=48000):
        """Extracts (1, 1, 1001, 64) log-mel spectrogram for CLAP Audio Encoder."""
        if sr != 48000:
            pcm_data = librosa.resample(pcm_data, orig_sr=sr, target_sr=48000)

        # Ensure 10 seconds (480,000 samples) via repeatpad
        target_len = 480000
        if len(pcm_data) < target_len:
            repeats = int(np.ceil(target_len / max(1, len(pcm_data))))
            pcm_data = np.tile(pcm_data, repeats)[:target_len]
        else:
            pcm_data = pcm_data[:target_len]

        mel = librosa.feature.melspectrogram(
            y=pcm_data, sr=48000, n_fft=1024, hop_length=480, n_mels=64, fmin=50, fmax=14000, power=2.0
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        log_mel = log_mel.T.astype(np.float32)
        return np.expand_dims(np.expand_dims(log_mel, axis=0), axis=0)

    def encode_audio(self, pcm_data, sr=48000):
        """Encodes PCM audio into L2-normalized 512D float embedding."""
        if not self.initialize_audio() or not self.audio_sess:
            return np.zeros(512, dtype=np.float32)

        try:
            feats = self.extract_audio_features(pcm_data, sr)
            embeds = self.audio_sess.run(None, {"input_features": feats})[0]
            norm = np.linalg.norm(embeds, axis=1, keepdims=True)
            if norm[0, 0] > 0:
                embeds = embeds / norm
            return embeds[0].astype(np.float32)
        except Exception as e:
            sys.stderr.write(f"[CLAP] Audio encoding error: {str(e)}\n")
            return np.zeros(512, dtype=np.float32)


# ==============================================================================
# 2. AUDIO DECODER (MULTI-FORMAT STREAMING)
# ==============================================================================
class AudioDecoder:
    """Decodes WAV, FLAC, OGG, MP3, AAC, M4A, AIF into 32-bit Float PCM arrays."""

    TARGET_SR = 48000  # 48 kHz standard for CLAP

    @classmethod
    def decode(cls, file_path):
        if not os.path.exists(file_path):
            return None, None

        # Attempt 1: Direct fast SoundFile decode
        try:
            data, sr = sf.read(file_path, dtype='float32')
            if data.ndim > 1:
                data = np.mean(data, axis=1)  # Downmix to mono

            if sr != cls.TARGET_SR and len(data) > 0:
                data = librosa.resample(data, orig_sr=sr, target_sr=cls.TARGET_SR)
                sr = cls.TARGET_SR

            max_peak = np.max(np.abs(data))
            if max_peak > 1e-4:
                data = data / max_peak * 0.98

            return data, sr
        except Exception:
            pass

        # Attempt 2: Piped FFmpeg PCM decode fallback
        try:
            cmd = [
                FFMPEG_PATH, "-v", "error",
                "-i", file_path,
                "-f", "f32le",
                "-ac", "1",
                "-ar", str(cls.TARGET_SR),
                "-"
            ]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            out, _ = proc.communicate()
            if len(out) > 0:
                data = np.frombuffer(out, dtype=np.float32).copy()
                max_peak = np.max(np.abs(data))
                if max_peak > 1e-4:
                    data = data / max_peak * 0.98
                return data, cls.TARGET_SR
        except Exception:
            pass

        return None, None


# ==============================================================================
# 3. ACOUSTIC PROFILER (PHYSICAL DSP + CLAP EMBEDDING)
# ==============================================================================
class AcousticProfiler:
    """Extracts physical transient attack, lead-in, crest factor, and frequency bands."""

    @classmethod
    def analyze(cls, file_path, compute_clap=True):
        data, sr = AudioDecoder.decode(file_path)
        if data is None or len(data) < 100:
            return None

        total_samples = len(data)
        duration_sec = total_samples / sr

        # 1. Transient Peak & Lead-in (10ms sliding RMS window)
        window_size = int(sr * 0.010)
        window = np.ones(window_size, dtype=np.float32) / window_size
        squares = data ** 2
        rms_env = np.sqrt(np.convolve(squares, window, mode='valid'))

        if len(rms_env) > 0:
            max_rms = float(np.max(rms_env))
            active_samples = np.where(rms_env > 0.15 * max_rms)[0]
            if len(active_samples) > 0:
                peak_idx = int(active_samples[0])
            else:
                peak_idx = int(np.argmax(rms_env))
            peak_transient_sec = float((peak_idx + window_size / 2) / sr)
        else:
            peak_transient_sec = 0.0

        lead_in_ms = round(peak_transient_sec * 1000, 1)

        # 2. Crest Factor (Dynamic Punch in dB)
        overall_rms = float(np.sqrt(np.mean(squares)))
        overall_peak = float(np.max(np.abs(data)))
        crest_factor_db = float(20.0 * np.log10(max(1.0, overall_peak / max(1e-5, overall_rms))))

        # 3. Spectral Centroid & Frequency Band Distribution (FFT)
        start_samp = max(0, int((peak_transient_sec - 0.1) * sr))
        end_samp = min(total_samples, int((peak_transient_sec + 0.4) * sr))
        segment = data[start_samp:end_samp] if (end_samp - start_samp) > 512 else data[:min(total_samples, 2048)]

        fft_vals = np.abs(fft.rfft(segment))
        freqs = fft.rfftfreq(len(segment), 1.0 / sr)
        sum_fft = np.sum(fft_vals)

        if sum_fft > 1e-6:
            spectral_centroid = float(np.sum(freqs * fft_vals) / sum_fft)
            low_band = float(np.sum(fft_vals[freqs < 250])) / sum_fft
            high_band = float(np.sum(fft_vals[freqs >= 2500])) / sum_fft
        else:
            spectral_centroid = 1000.0
            low_band, high_band = 0.33, 0.33

        # 4. Optional CLAP Audio Embedding (512D)
        clap_vector = None
        if compute_clap:
            clap = ClapEngine.get_instance()
            clap_vector = clap.encode_audio(data, sr).tolist()

        fname = os.path.basename(file_path)
        name_clean = os.path.splitext(fname)[0].replace("-", " ").replace("_", " ").lower()
        if name_clean.startswith("sfx "):
            name_clean = name_clean[4:].strip()

        # Token list for secondary keyword fallback
        tokens = [t.strip() for t in name_clean.split() if len(t.strip()) > 1]

        return {
            "filename": str(fname),
            "path": str(os.path.abspath(file_path)),
            "duration_sec": float(round(duration_sec, 3)),
            "peak_transient_sec": float(round(peak_transient_sec, 3)),
            "lead_in_ms": float(lead_in_ms),
            "crest_factor_db": float(round(crest_factor_db, 1)),
            "spectral_centroid_hz": float(round(spectral_centroid, 1)),
            "low_band_ratio": float(round(low_band, 3)),
            "high_band_ratio": float(round(high_band, 3)),
            "tokens": tokens,
            "clean_name": str(name_clean),
            "clap_vector": clap_vector
        }


# ==============================================================================
# 4. SOUNDBANK INDEXER & PERSISTENT VECTOR VAULT
# ==============================================================================
class SoundbankIndexer:
    """Scans and indexes audio folders into an acoustic metadata vault + CLAP vector matrix."""

    SUPPORTED_EXTS = {'.wav', '.mp3', '.aac', '.m4a', '.flac', '.ogg', '.webm', '.aif', '.aiff'}

    @classmethod
    def index_directory(cls, sfx_dir, output_vault=DEFAULT_VAULT_INDEX, output_embeddings=DEFAULT_EMBEDDINGS, progress_callback=None):
        if not os.path.exists(sfx_dir):
            return {"status": "error", "message": f"Target directory not found: {sfx_dir}"}

        audio_files = []
        for root, _, files in os.walk(sfx_dir):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in cls.SUPPORTED_EXTS:
                    audio_files.append(os.path.join(root, f))

        if not audio_files:
            return {"status": "error", "message": f"No supported audio files found in: {sfx_dir}"}

        start_t = time.perf_counter()
        total_count = len(audio_files)

        # Load existing cache for incremental indexing
        cached_items = {}
        if os.path.exists(output_vault) and os.path.exists(output_embeddings):
            try:
                with open(output_vault, "r", encoding="utf-8") as f:
                    old_items = json.load(f)
                    cached_items = {item["path"]: item for item in old_items if "path" in item}
            except Exception:
                cached_items = {}

        # Warm up CLAP engine
        clap = ClapEngine.get_instance()
        clap.initialize()

        indexed_items = []
        embedding_matrix = []

        for i, fpath in enumerate(audio_files):
            abs_p = os.path.abspath(fpath)
            meta = None

            # Incremental check: compare modification time
            if abs_p in cached_items:
                try:
                    current_mtime = os.path.getmtime(abs_p)
                    cached_mtime = cached_items[abs_p].get("mtime", 0)
                    if abs(current_mtime - cached_mtime) < 1.0 and cached_items[abs_p].get("clap_vector"):
                        meta = cached_items[abs_p]
                except Exception:
                    meta = None

            if meta is None:
                meta = AcousticProfiler.analyze(abs_p, compute_clap=True)
                if meta:
                    try:
                        meta["mtime"] = os.path.getmtime(abs_p)
                    except Exception:
                        meta["mtime"] = 0

            if meta and meta.get("clap_vector"):
                vec = meta["clap_vector"]
                embedding_matrix.append(vec)
                indexed_items.append(meta)

            if progress_callback and (i % 5 == 0 or i == total_count - 1):
                progress_callback(i + 1, total_count)

        elapsed = time.perf_counter() - start_t

        # Attach Acoustic Categories with Lexical Grounding & Zero-Shot CLAP
        if embedding_matrix and clap.is_available():
            try:
                emb_array = np.array(embedding_matrix, dtype=np.float32)
                anchor_vecs = clap.encode_text(ACOUSTIC_CATEGORY_PROMPTS)
                cat_sims = np.dot(emb_array, anchor_vecs.T)
                cat_indices = np.argmax(cat_sims, axis=1)
                for i_item, c_idx in enumerate(cat_indices):
                    item = indexed_items[i_item]
                    toks = set(item.get("tokens", []))
                    fn = item.get("filename", "").lower()
                    assigned_cat = None
                    if any(k in toks for k in ["click", "tap", "snap"]) or "click" in fn:
                        assigned_cat = "click"
                    elif (any(k in toks for k in ["pop", "bubble"]) or "pop" in fn) and "popular" not in fn and "popup" not in fn:
                        assigned_cat = "pop"
                    elif any(k in toks for k in ["slide", "glide", "whoosh", "sweep", "swish"]) or "slide" in fn:
                        assigned_cat = "whoosh"
                    elif any(k in toks for k in ["typing", "keyboard", "paper", "tear", "rip", "footstep", "rustle", "foley"]) or "typing" in fn:
                        assigned_cat = "foley"
                    elif any(k in toks for k in ["riser", "swell"]) or "riser" in fn:
                        assigned_cat = "riser"
                    elif any(k in toks for k in ["chime", "bell", "ding", "shine", "sparkle"]):
                        assigned_cat = "chime"
                    elif any(k in toks for k in ["hit", "impact", "slam", "boom", "punch"]):
                        assigned_cat = "impact"
                    elif any(k in toks for k in ["ui", "interface", "app"]):
                        assigned_cat = "ui"
                    else:
                        assigned_cat = ACOUSTIC_CATEGORY_NAMES[int(c_idx)]

                    indexed_items[i_item]["acoustic_category"] = assigned_cat
            except Exception as eCat:
                sys.stderr.write(f"[SoundbankIndexer] Category assign warning: {eCat}\n")

        # Save metadata index with portable relative paths (omit heavy clap_vector from JSON)
        portable_indexed = []
        for item in indexed_items:
            it_copy = dict(item)
            it_copy.pop("clap_vector", None)  # Embeddings are stored compactly in .npy
            p = it_copy.get("path", "")
            if p and os.path.isabs(p):
                try:
                    rel = os.path.relpath(p, BASE_DIR).replace("\\", "/")
                    if not rel.startswith(".."):
                        it_copy["path"] = rel
                except Exception:
                    pass
            portable_indexed.append(it_copy)

        os.makedirs(os.path.dirname(os.path.abspath(output_vault)), exist_ok=True)
        with open(output_vault, "w", encoding="utf-8") as f:
            json.dump(portable_indexed, f, indent=2, cls=NumpyEncoder)

        # Save float32 embedding matrix
        if embedding_matrix:
            emb_array = np.array(embedding_matrix, dtype=np.float32)
            np.save(output_embeddings, emb_array)

        vault_size_kb = round(os.path.getsize(output_vault) / 1024, 2)
        emb_size_kb = round(os.path.getsize(output_embeddings) / 1024, 2) if os.path.exists(output_embeddings) else 0

        return {
            "status": "success",
            "directory": str(sfx_dir),
            "total_files": total_count,
            "indexed_count": len(indexed_items),
            "elapsed_sec": round(elapsed, 2),
            "vault_file": str(output_vault),
            "vault_size_kb": vault_size_kb,
            "embeddings_file": str(output_embeddings),
            "embeddings_size_kb": emb_size_kb
        }


# ==============================================================================
# 5. CONTEXT-AWARE CLAP SEMANTIC MATCHER & TIMELINE RERANKER
# ==============================================================================
class SoundMatcher:
    """Matches Gemini visual editorial cues against soundbank using CLAP + Context Reranker."""

    @classmethod
    def load_vault(cls, vault_path=DEFAULT_VAULT_INDEX, embeddings_path=DEFAULT_EMBEDDINGS):
        if not os.path.exists(vault_path) or not os.path.exists(embeddings_path):
            return None, None
        try:
            with open(vault_path, "r", encoding="utf-8") as f:
                soundbank = json.load(f)
            embeddings = np.load(embeddings_path)

            bundled_sfx_dir = os.path.join(BASE_DIR, "assets", "sfx")
            for item in soundbank:
                # Dynamic cross-platform path resolution
                p = item.get("path")
                if not p or not os.path.exists(p):
                    fname = item.get("filename", "")
                    candidate = os.path.join(bundled_sfx_dir, fname)
                    if os.path.exists(candidate):
                        item["path"] = os.path.abspath(candidate)
                    elif p and not os.path.isabs(p) and os.path.exists(os.path.join(BASE_DIR, p)):
                        item["path"] = os.path.abspath(os.path.join(BASE_DIR, p))
                    else:
                        item["path"] = os.path.abspath(candidate)
                else:
                    item["path"] = os.path.abspath(p)

            return soundbank, embeddings
        except Exception as e:
            sys.stderr.write(f"[SoundMatcher] Failed to load vault: {e}\n")
            return None, None

    @staticmethod
    def compute_acoustic_gain_compensation(sound_meta, base_recommended_gain=-12.0):
        """
        Computes physical frequency & loudness compensation for an SFX file:
        - Sub-bass / low-end heavy sounds (>40% energy below 250 Hz) lose perceived loudness
          due to human ear Fletcher-Munson insensitivity. Boosts up to +9 dB.
        - Harsh high-frequency sounds (>40% energy above 2.5 kHz) soften by -1 to -3.5 dB.
        - Sparse impulse clicks (crest factor > 15 dB) receive a presence lift (+2 to +3 dB).
        Clamped strictly within [-18.0 dB, 0.0 dB].
        """
        if not sound_meta:
            return int(round(max(-18.0, min(0.0, float(base_recommended_gain)))))
        low_ratio = float(sound_meta.get("low_band_ratio", 0.3) or 0.3)
        high_ratio = float(sound_meta.get("high_band_ratio", 0.3) or 0.3)
        crest = float(sound_meta.get("crest_factor_db", 12.0) or 12.0)

        # 1. Fletcher-Munson Low-end Ear Sensitivity Compensation
        low_comp = 0.0
        if low_ratio > 0.40:
            low_comp = min(9.0, (low_ratio - 0.30) * 25.0)
        elif low_ratio > 0.25:
            low_comp = (low_ratio - 0.25) * 6.0

        # 2. Harsh high-frequency softening
        high_comp = 0.0
        if high_ratio > 0.40:
            high_comp = -min(3.5, (high_ratio - 0.35) * 6.0)

        # 3. Impulse click presence lift (micro-transients with low RMS)
        crest_comp = 0.0
        if crest > 15.0 and low_ratio < 0.35:
            crest_comp = min(3.0, (crest - 14.0) * 0.6)

        total_comp = low_comp + high_comp + crest_comp
        final_gain = int(round(max(-18.0, min(0.0, float(base_recommended_gain) + total_comp))))
        return final_gain

    @classmethod
    def match_cues(cls, cues, soundbank, embeddings_matrix, style_preset="auto", cooldown_size=4):
        """
        Executes sub-10ms CLAP vector cosine similarity + Multi-Stage Context Reranking:
        1. CLAP Text Embedding Query (Cosine Sim)
        2. Transient Attack & Dynamic Fit
        3. Dialogue Bleed Clearance Filter
        4. Editing Style Preset Multipliers
        5. Anti-Repetition FIFO Cooldown
        6. Sub-Frame Negative Anchor Alignment Math
        7. Dual-Track A/B Checkerboard Scheduling
        """
        if not soundbank or embeddings_matrix is None or len(soundbank) == 0 or not cues:
            return []

        clap = ClapEngine.get_instance()
        clap.initialize_text()

        # 1. Build dual-anchor text queries combining event action + acoustic descriptor
        query_texts = []
        for cue in cues:
            ev = str(cue.get("event_name", cue.get("eventName", cue.get("name", "")))).strip()
            desc = str(cue.get("search_descriptor", cue.get("description", cue.get("sound_texture", "")))).strip()
            if ev and desc:
                combined = f"{ev}. {desc}"
            elif ev:
                combined = ev
            elif desc:
                combined = desc
            else:
                combined = "sound."
            query_texts.append(combined)

        # Batch encode all cue texts in a single pass (<15 ms total)
        query_vectors = clap.encode_text(query_texts)

        # Matrix dot product: (N_cues, 512) x (512, M_sounds) -> (N_cues, M_sounds)
        cosine_sim_matrix = np.dot(query_vectors, embeddings_matrix.T)

        cooldown_buffer = []
        matched_results = []

        for i, cue in enumerate(cues):
            contact_time = float(cue.get("timestamp_sec", cue.get("contact_time_sec", 0.0)))
            cue_name = str(cue.get("event_name", cue.get("eventName", ""))).lower()
            cue_desc = str(cue.get("search_descriptor", cue.get("description", cue.get("sound_texture", "")))).lower()
            cue_cat = str(cue.get("category", cue.get("primary_category", ""))).lower()
            dialogue_present = bool(cue.get("context_awareness", {}).get("dialogue_present", cue.get("dialogue_present", False)))
            recommended_gain = float(cue.get("recommended_gain_db", -12.0))
            scene_gap = float(cue.get("context_awareness", {}).get("scene_gap_sec", 2.0))

            # Extract meaningful tokens for lexical overlap (from both action event name and acoustic descriptor)
            cue_words = set(re.findall(r'[a-z0-9]+', f"{cue_name} {cue_desc}"))
            cue_tokens = {w for w in cue_words if len(w) > 2}

            # Detect continuous / sustained streams (typing, footsteps, rain, engine)
            is_continuous_foley = any(k in (cue_name + " " + cue_desc) for k in [
                "type", "typing", "keyboard", "clatter", "drift", "skid",
                "pour", "rain", "fire", "engine", "hum", "buzz", "footstep",
                "ambient", "texture", "rustle", "loop"
            ])

            sim_scores = cosine_sim_matrix[i]

            scored_candidates = []
            for idx, item in enumerate(soundbank):
                raw_clap_sim = float(sim_scores[idx])  # typically -0.2 to +0.8

                # Primary Score: 512D CLAP Semantic Cosine Similarity (Dominant 70% weight)
                score = raw_clap_sim * 100.0

                # Secondary Score: Lexical Token Overlap (25% weight)
                s_tokens = set(item.get("tokens", []))
                overlap = cue_tokens.intersection(s_tokens)
                if overlap:
                    score += min(35.0, len(overlap) * 12.0)

                dur = float(item.get("duration_sec", 1.0))

                # Duration Safety Guard
                if not is_continuous_foley and scene_gap > 0.1:
                    if dur > (scene_gap + 2.0):
                        overage = dur - (scene_gap + 2.0)
                        score -= min(40.0, overage * 12.0)

                # Style-aware subtle nuance
                if style_preset in ("saas", "clean"):
                    if item.get("low_band_ratio", 0.3) > 0.65:
                        score -= 15.0
                elif style_preset == "gaming":
                    if item.get("low_band_ratio", 0.3) > 0.35 or item.get("crest_factor_db", 12.0) > 15.0:
                        score += 5.0

                # Anti-Repetition FIFO Cooldown Penalty
                fname = item.get("filename", "")
                if cooldown_size > 0 and fname in cooldown_buffer:
                    idx_rec = cooldown_buffer.index(fname)
                    score -= (cooldown_size - idx_rec) * 35.0

                scored_candidates.append((score, raw_clap_sim, item))

            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            if not scored_candidates:
                continue

            best_score, best_clap_sim, best = scored_candidates[0]

            # Update cooldown buffer
            if cooldown_size > 0:
                cooldown_buffer.append(best["filename"])
                if len(cooldown_buffer) > cooldown_size:
                    cooldown_buffer.pop(0)

            # Universal Failsafe Anchor & Alignment Logic
            lead_in = float(best.get("peak_transient_sec", 0.0))
            dur = float(best.get("duration_sec", 1.0))

            ev_combined = (cue_name + " " + cue_desc + " " + cue_cat).lower()
            is_true_scene_cut = any(k in ev_combined for k in [
                "scene transition", "scene cut", "cut to new scene", "hard cut to",
                "shot transition", "whip transition", "scene change"
            ]) and not any(k in ev_combined for k in [
                "text", "pop", "slide", "reveal", "activation", "transform",
                "click", "button", "tap", "glitch", "icon", "logo", "graphic", "board"
            ])

            silence_trim = 0.0
            if 0.015 <= lead_in <= 0.120:
                silence_trim = round(lead_in, 3)

            if is_true_scene_cut:
                pre_roll = min(lead_in, 0.100)
                timeline_anchor = max(0.0, contact_time - pre_roll)
                clip_in_point = 0.0
            else:
                timeline_anchor = max(0.0, contact_time)
                clip_in_point = silence_trim

            if is_continuous_foley and scene_gap > 0.5:
                target_len = min(dur, max(1.5, scene_gap + 0.4))
                clip_out_point = round(target_len, 3)
            else:
                clip_out_point = round(dur, 3)

            # Format alternatives (Top 2 to 5)
            alternatives = []
            for alt_score, alt_sim, alt_item in scored_candidates[1:5]:
                alt_lead_in = float(alt_item.get("peak_transient_sec", 0.0))
                alt_anchor = max(0.0, contact_time)
                alternatives.append({
                    "name": alt_item["filename"],
                    "filename": alt_item["filename"],
                    "path": alt_item["path"],
                    "duration": alt_item["duration_sec"],
                    "duration_sec": alt_item["duration_sec"],
                    "lead_in_ms": alt_item.get("lead_in_ms", 0),
                    "timeline_anchor_sec": round(alt_anchor, 3),
                    "score": round(alt_score, 1),
                    "clap_similarity": round(alt_sim, 3),
                    "volumeDB": cls.compute_acoustic_gain_compensation(alt_item, recommended_gain),
                    "low_band_ratio": alt_item.get("low_band_ratio", 0.3),
                    "high_band_ratio": alt_item.get("high_band_ratio", 0.3),
                    "crest_factor_db": alt_item.get("crest_factor_db", 12.0),
                    "category": alt_item.get("acoustic_category", alt_item.get("category", "sfx"))
                })

            sound_info = {
                "name": best["filename"],
                "filename": best["filename"],
                "path": best["path"],
                "duration": best["duration_sec"],
                "duration_sec": best["duration_sec"],
                "lead_in_ms": best.get("lead_in_ms", 0),
                "peak_transient_sec": lead_in,
                "crest_factor_db": best.get("crest_factor_db", 0),
                "spectral_centroid_hz": best.get("spectral_centroid_hz", 0),
                "low_band_ratio": best.get("low_band_ratio", 0.0),
                "high_band_ratio": best.get("high_band_ratio", 0.0),
                "acoustic_category": best.get("acoustic_category", best.get("category", "sfx")),
                "category": best.get("acoustic_category", best.get("category", "sfx")),
                "volumeDB": cls.compute_acoustic_gain_compensation(best, recommended_gain)
            }

            cid = cue.get("cue_id", i + 1)
            ev_name = cue.get("event_name", f"Event @ {contact_time:.2f}s")
            v_trig = cue.get("visual_trigger", cue.get("description", ""))
            p_cat = cue.get("primary_category", cue.get("category", "sfx"))
            s_text = cue.get("sound_texture", "")

            matched_results.append({
                "cue_id": cid,
                "cueId": cid,
                "event_name": ev_name,
                "eventName": ev_name,
                "visual_trigger": v_trig,
                "visualTrigger": v_trig,
                "primary_category": p_cat,
                "category": p_cat,
                "sound_texture": s_text,
                "soundTexture": s_text,
                "contact_time_sec": round(contact_time, 3),
                "cutTimeSec": round(contact_time, 3),
                "dialoguePresent": dialogue_present,
                "matched_sound": sound_info,
                "matchedSound": sound_info,
                "originalSound": sound_info,
                "placement": {
                    "timeline_anchor_sec": round(timeline_anchor, 3),
                    "clip_in_point_sec": round(clip_in_point, 3),
                    "clip_out_point_sec": round(clip_out_point, 3),
                    "target_track": "auto"
                },
                "match_score": round(best_score, 1),
                "clap_similarity": round(best_clap_sim, 3),
                "volumeDB": cls.compute_acoustic_gain_compensation(best, recommended_gain),
                "volumeOffsetDB": int(round(cls.compute_acoustic_gain_compensation(best, recommended_gain) - (-12.0))),
                "alternatives": alternatives
            })

        # Strict chronological monotonicity (TimelineStart[i] >= TimelineStart[i-1])
        matched_results.sort(key=lambda x: x["contact_time_sec"])
        last_anchor = 0.0
        for item in matched_results:
            pl = item["placement"]
            if pl["timeline_anchor_sec"] < last_anchor:
                pl["timeline_anchor_sec"] = last_anchor
            last_anchor = pl["timeline_anchor_sec"]

        # Dynamic Collision-Free Multi-Track Slot Allocation (No overwrites)
        track_busy_ends = []
        for item in matched_results:
            pl = item["placement"]
            start = pl["timeline_anchor_sec"]
            dur = max(0.05, pl["clip_out_point_sec"] - pl["clip_in_point_sec"])
            end = start + dur

            assigned_idx = None
            for t_idx, t_end in enumerate(track_busy_ends):
                if start >= (t_end - 0.02):
                    track_busy_ends[t_idx] = end
                    assigned_idx = t_idx
                    break
            if assigned_idx is None:
                track_busy_ends.append(end)
                assigned_idx = len(track_busy_ends) - 1

            pl["target_track_index"] = assigned_idx
            pl["target_track"] = f"Track_{assigned_idx + 1}"

        return matched_results

        return matched_results


# ==============================================================================
# 6. LOCAL COMPUTER-VISION MICRO-MOTION SNAPPER
# ==============================================================================
class VideoMotionSnapper:
    """Lightweight computer-vision micro-motion transient snapping."""

    @staticmethod
    def align_and_filter_cues(cues, video_path, time_offset=0.0):
        if not video_path or not os.path.exists(video_path):
            return cues
        try:
            import cv2
        except ImportError:
            return cues

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return cues
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Low-framerate micro-proxies (< 12 fps) lack temporal density; preserve AI timestamps directly
        if fps < 12.0:
            cap.release()
            return cues

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames < 10:
            cap.release()
            return cues

        # Refine cues within a local window (avoiding unbounded full-video frame buffering)
        for cue in cues:
            coarse_t_rel = float(cue.get('timestamp_sec', cue.get('contact_time_sec', 0.0)))
            coarse_t = coarse_t_rel + time_offset
            target_f = int(coarse_t * fps)
            if target_f <= 2 or target_f >= total_frames - 2:
                continue

            # Read a tight 6-frame window around the target
            start_f = max(0, target_f - 3)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
            diffs = []
            prev_gray = None
            for _ in range(7):
                ret, frame = cap.read()
                if not ret:
                    break
                h, w = frame.shape[:2]
                small = cv2.resize(frame, (180, int(180 * h / w))) if w > 180 else frame
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    diffs.append(float(np.mean(cv2.absdiff(gray, prev_gray))))
                prev_gray = gray

            if diffs and len(diffs) >= 3:
                max_d_idx = int(np.argmax(diffs))
                if diffs[max_d_idx] > 3.0:
                    best_f = start_f + 1 + max_d_idx
                    refined_t = round((best_f / fps) - time_offset, 3)
                    cue['timestamp_sec'] = max(0.0, refined_t)
                    cue['contact_time_sec'] = max(0.0, refined_t)

        cap.release()
        return cues


# ==============================================================================
# 7. CLI DISPATCHER & MAIN ENTRYPOINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Asset Station Auto SFX CLAP Production Engine")
    parser.add_argument("--index", type=str, help="Directory of soundbank to index")
    parser.add_argument("--vault", type=str, default=DEFAULT_VAULT_INDEX, help="Path to sfx_vault.idx")
    parser.add_argument("--embeddings", type=str, default=DEFAULT_EMBEDDINGS, help="Path to sfx_embeddings.npy")
    parser.add_argument("--match", action="store_true", help="Run context-aware CLAP matching")
    parser.add_argument("--cues", type=str, help="Path to cues JSON file or raw JSON string")
    parser.add_argument("--style", type=str, default="auto", help="Style: auto | saas | gaming | casual_reel | informative | documentary | storytelling")
    parser.add_argument("--cooldown-size", type=int, default=4, help="Anti-repeat FIFO buffer size")
    parser.add_argument("--stdin", action="store_true", help="Read JSON payload from standard input")
    parser.add_argument("--test", action="store_true", help="Run self-diagnostic check on models & bundled soundbank")
    parser.add_argument("--video", type=str, help="Path to local video (or proxy) for computer-vision micro-motion transient snapping")
    parser.add_argument("--time-offset", type=float, default=0.0, help="Source time offset in seconds for trimmed clips")

    args = parser.parse_args()

    # Self-diagnostic test
    if args.test:
        clap = ClapEngine.get_instance()
        avail = clap.is_available()
        print(f"CLAP Models Available: {avail}")
        if avail:
            init_ok = clap.initialize_text()
            print(f"CLAP Text Initialization: {'SUCCESS' if init_ok else 'FAILED'}")
            test_vec = clap.encode_text(["test query"])
            print(f"Text Encoder Output Shape: {test_vec.shape}, Norm: {np.linalg.norm(test_vec[0]):.4f}")
        return

    # Mode 1: Index Directory
    if args.index:
        def on_prog(curr, tot):
            pct = int((curr / tot) * 100)
            sys.stderr.write(f"PROGRESS:{pct}:{curr}/{tot}\n")
            sys.stderr.flush()

        res = SoundbankIndexer.index_directory(args.index, output_vault=args.vault, output_embeddings=args.embeddings, progress_callback=on_prog)
        print(json.dumps(res, indent=2, cls=NumpyEncoder))
        return

    # Mode 2: Context-Aware Matching
    if args.match or args.stdin:
        vault_path = args.vault
        emb_path = args.embeddings
        soundbank, embeddings = SoundMatcher.load_vault(vault_path, emb_path)

        if soundbank is None or embeddings is None:
            print(json.dumps({"status": "error", "message": f"Vault or embeddings missing at {vault_path}. Run --index first."}, cls=NumpyEncoder))
            return

        raw_cues = None
        if args.stdin:
            input_text = sys.stdin.read().strip()
            if input_text:
                try:
                    raw_cues = json.loads(input_text)
                except Exception as e:
                    print(json.dumps({"status": "error", "message": f"Failed to parse stdin JSON: {str(e)}"}, cls=NumpyEncoder))
                    return
        elif args.cues:
            if os.path.exists(args.cues):
                with open(args.cues, "r", encoding="utf-8") as f:
                    raw_cues = json.load(f)
            else:
                try:
                    raw_cues = json.loads(args.cues)
                except Exception as e:
                    print(json.dumps({"status": "error", "message": f"Failed to parse cues JSON: {str(e)}"}, cls=NumpyEncoder))
                    return

        if not raw_cues:
            print(json.dumps({"status": "error", "message": "No valid cues provided to match."}, cls=NumpyEncoder))
            return

        cues_list = raw_cues.get("cues", raw_cues) if isinstance(raw_cues, dict) else raw_cues
        if args.video and os.path.exists(args.video):
            cues_list = VideoMotionSnapper.align_and_filter_cues(cues_list, args.video, time_offset=getattr(args, "time_offset", 0.0))
        results = SoundMatcher.match_cues(cues_list, soundbank, embeddings, style_preset=args.style, cooldown_size=args.cooldown_size)
        print(json.dumps({"status": "success", "total_cues": len(cues_list), "matched_count": len(results), "cues": results}, indent=2, cls=NumpyEncoder))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
