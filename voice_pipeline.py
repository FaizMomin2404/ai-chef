"""
AI Chef - voice cooking companion backend.

Flow:
  browser text (typed / Web Speech API)  ->  command router
  command router -> recipe generation (Ollama) or session navigation
  result -> WebSocket text for the UI  +  spoken audio (Piper TTS)

The backend also exposes a local microphone endpoint using faster-whisper,
for browsers without speech recognition. The browser Web Speech API is the
primary STT because it is instant and needs no extra model on the hot path.
"""

import asyncio
import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Optional, Callable, Dict, List

import numpy as np
import aiohttp

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from recipe_engine import (
    RecipeUnavailable,
    RecipeValidationError,
    ambiguous_base,
    apply_variant,
    build_fallback,
    build_recipe_prompt,
    build_repair_prompt,
    clarify_dish,
    family_of,
    normalize_name,
    parse_json_loose,
    smart_title,
    suggest_from_ingredients,
    validate_recipe,
)
from state_manager import CookingSession, load_session, save_session, session
from timer_manager import TimerManager, timers
from app_config import (
    ALLOWED_ORIGINS,
    DEPLOYMENT_MODE,
    MAX_CONNECTIONS,
    MAX_MESSAGE_CHARS,
    MULTI_USER,
    PIPER_DIR,
    PIPER_VOICE,
    SESSION_PERSISTENCE,
    STT_MODE,
    TTS_ENABLED,
    TTS_MODE,
)
from llm_provider import (
    ProviderConfigError,
    ProviderError,
    check_provider,
    llm_chat,
    provider_status,
)
from intent_router import (
    classify_intent,
    conversational_response,
    IntentResult,
    GENERAL_CHAT,
    RECIPE_INFORMATION,
    RECIPE_REQUEST,
    INGREDIENT_BASED_SUGGESTION,
    START_COOKING,
    COOKING_FOLLOW_UP,
    TIMER_COMMAND,
    REPEAT_OR_CLARIFY,
    UNSAFE_OR_UNSUPPORTED_REQUEST,
)
from rag_system import retrieve_for_dish, retrieve, knowledge_stats


# ======================================================================
# Configuration
# ======================================================================
SAMPLE_RATE = 16000
WHISPER_MODEL_SIZE = "small"
WHISPER_COMPUTE = "int8"

OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "phi3:3.8b")

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
SESSION_FILE = os.environ.get("SESSION_FILE") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "session_state.json"
)


# ======================================================================
# System prompt
# ======================================================================
SYSTEM_PROMPT = """You are AI Chef, a warm, patient cooking teacher standing in the user's kitchen.

Rules:
- Teach one small action at a time. Keep every answer under 40 words.
- Reply in English, or in friendly Hinglish when the user speaks Hinglish.
- Use the session context (dish, servings, current step, tasks, conditions) in every answer.
- Never claim you can see, smell, or taste the food. You have no camera.
- Safety first: hot oil, boiling water, gas flames, pressure cookers, raw meat, sharp knives.
  Never suggest leaving active heating unattended.
- If the user lacks an ingredient, judge honestly whether the dish still works, suggest a
  sensible substitute, and say how taste or texture changes.
- Do not repeat the whole recipe. Answer the question asked.
- If the user is unclear, ask one short clarifying question instead of guessing.
"""


# ======================================================================
# Text to speech (Piper, with SAPI fallback)
# ======================================================================
class TTSEngine:
    """Piper speech output with a Windows SAPI fallback and a stop control."""

    def __init__(self):
        self.voice = None
        self.sample_rate = 22050
        self._lock = asyncio.Lock()
        self._gen_lock = threading.Lock()
        self._gen = 0
        self.available = False
        self.last_error = ""
        if TTS_MODE == "browser" or not TTS_ENABLED:
            print("[TTS] Server speech disabled; the browser will speak replies.")
            return
        self._load()

    def _load(self):
        try:
            from piper import PiperVoice
            path = self._find_model()
            if path is None:
                self._download()
                path = self._find_model()
            if path:
                self.voice = PiperVoice.load(path)
                print(f"[TTS] Piper voice loaded: {path}")
            else:
                print("[TTS] Piper voice missing, using Windows SAPI fallback")
        except Exception as e:
            self.last_error = str(e)
            print(f"[TTS] Piper unavailable ({e}); using SAPI fallback")
            self.voice = None

        import importlib.util
        sapi = importlib.util.find_spec("pyttsx3") is not None
        self.available = bool(self.voice is not None or sapi)
        if not self.available:
            print("[TTS] No speech output available. Text will still be shown.")

    def _download(self):
        try:
            from pathlib import Path
            from piper.download_voices import download_voice
            Path(PIPER_DIR).mkdir(parents=True, exist_ok=True)
            print(f"[TTS] Downloading voice {PIPER_VOICE} ...")
            download_voice(PIPER_VOICE, Path(PIPER_DIR))
        except Exception as e:
            self.last_error = str(e)
            print(f"[TTS] Voice download failed: {e}")

    def _find_model(self):
        import glob
        direct = os.path.join(PIPER_DIR, f"{PIPER_VOICE}.onnx")
        if os.path.exists(direct):
            return direct
        if os.path.isdir(PIPER_DIR):
            found = sorted(glob.glob(os.path.join(PIPER_DIR, "*.onnx")))
            if found:
                return found[0]
        return None

    def _synthesize(self, text: str):
        chunks = list(self.voice.synthesize(text))
        if not chunks:
            return None, 0
        sr = chunks[0].sample_rate
        parts = [np.asarray(c.audio_float_array, dtype=np.float32).reshape(-1) for c in chunks]
        return np.concatenate(parts).astype(np.float32), sr

    def _speak_sapi(self, text: str) -> bool:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 165)
            engine.setProperty("volume", 1.0)
            engine.say(text)
            engine.runAndWait()
            try:
                engine.stop()
            except Exception:
                pass
            return True
        except Exception as e:
            self.last_error = str(e)
            print(f"[TTS] SAPI failed: {e}")
            return False

    def stop(self):
        """Interrupt speech playback (used by the barge-in button)."""
        with self._gen_lock:
            self._gen += 1
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass

    def _speak_blocking(self, text: str) -> Optional[bool]:
        """Return True if spoken, False on failure, None if interrupted."""
        with self._gen_lock:
            gen = self._gen
        if self.voice is not None:
            try:
                audio, sr = self._synthesize(text)
                if audio is not None and audio.size > 0:
                    with self._gen_lock:
                        if gen != self._gen:
                            return None
                    import sounddevice as sd
                    sd.play(audio.reshape(-1, 1), samplerate=sr)
                    sd.wait()
                    with self._gen_lock:
                        return True if gen == self._gen else None
            except Exception as e:
                self.last_error = str(e)
                print(f"[TTS] Piper playback failed: {e}")
        with self._gen_lock:
            if gen != self._gen:
                return None
        ok = self._speak_sapi(text)
        with self._gen_lock:
            if gen != self._gen:
                return None
            return ok

    async def speak(self, text: str) -> Optional[bool]:
        if not TTS_ENABLED or not text.strip() or not self.available:
            return False
        async with self._lock:
            return await asyncio.to_thread(self._speak_blocking, text)


# ======================================================================
# Speech to text (faster-whisper, local microphone)
# ======================================================================
def ensure_microphone_unmuted():
    """
    Windows keeps a per-device mute flag. A muted microphone still opens
    normally and just returns silence, which looks like a broken app.
    Unmute the default capture device and give it full volume.
    """
    try:
        from pycaw.pycaw import AudioUtilities
        mic = AudioUtilities.CreateDevice(AudioUtilities.GetMicrophone())
        ev = mic.EndpointVolume
        if ev.GetMute():
            ev.SetMute(0, None)
            print("[STT] Windows microphone was muted - unmuted it.")
        if ev.GetMasterVolumeLevelScalar() < 0.5:
            ev.SetMasterVolumeLevelScalar(1.0, None)
            print("[STT] Raised microphone volume to 100%.")
    except Exception as e:
        print(f"[STT] Could not check the Windows microphone mute state: {e}")


class STTEngine:
    OVERRIDE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mic_device.json")

    def __init__(self):
        self.model = None
        self._input_device = None
        self._device_locked = False
        self.last_peak = 0.0
        self.last_device_name = ""
        if STT_MODE == "browser":
            print("[STT] Server microphone disabled; the browser handles speech recognition.")
            return
        ensure_microphone_unmuted()
        self._load_override()
        self._load()

    def _load(self):
        try:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type=WHISPER_COMPUTE)
            print(f"[STT] faster-whisper '{WHISPER_MODEL_SIZE}' loaded (CPU int8)")
        except Exception as e:
            print(f"[STT] Model load failed: {e}")
            self.model = None

    # -- device management ---------------------------------------------
    def _load_override(self):
        """Restore the microphone the user picked in the UI."""
        try:
            with open(self.OVERRIDE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            idx = int(data.get("index"))
            import sounddevice as sd
            info = sd.query_devices(idx)
            if info["max_input_channels"] > 0:
                peak = self._probe_bounded(idx)
                if peak is not None and peak > 0.0005:
                    self._input_device = idx
                    self._device_locked = True
                    print(f"[STT] Using saved microphone {idx} ({info['name']}, peak {peak:.4f})")
                else:
                    print(f"[STT] Saved microphone {idx} ({info['name']}) is silent; "
                          "using the Windows default instead.")
        except Exception:
            pass

    def _save_override(self, idx: int):
        try:
            with open(self.OVERRIDE_FILE, "w", encoding="utf-8") as f:
                json.dump({"index": int(idx)}, f)
        except Exception as e:
            print(f"[STT] Could not save microphone choice: {e}")

    @staticmethod
    def _is_loopback(name: str) -> bool:
        low = name.lower()
        return any(k in low for k in ("stereo mix", "pc speaker", "wave out",
                                      "what u hear", "line in", "output"))

    def devices(self):
        """List input devices with the metadata the UI needs."""
        import sounddevice as sd
        try:
            default_in = sd.default.device[0]
        except Exception:
            default_in = -1
        out = []
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] < 1:
                continue
            out.append({
                "index": i,
                "name": d["name"],
                "channels": int(d["max_input_channels"]),
                "sample_rate": int(d["default_samplerate"]),
                "is_default": i == default_in,
                "loopback": self._is_loopback(d["name"]),
            })
        return out

    def set_device(self, index: int) -> int:
        """Select a microphone explicitly and remember the choice."""
        import sounddevice as sd
        info = sd.query_devices(int(index))
        if info["max_input_channels"] < 1:
            raise ValueError("That device has no microphone input")
        self._input_device = int(index)
        self._device_locked = True
        self._save_override(self._input_device)
        print(f"[STT] Microphone chosen: {self._input_device} ({info['name']})")
        return self._input_device

    @staticmethod
    def _broken(audio: np.ndarray) -> bool:
        if audio is None or audio.size == 0:
            return True
        peak = float(np.max(np.abs(audio)))
        if (not np.isfinite(peak)) or peak > 4.0:
            return True
        # Broken Bluetooth drivers sometimes return a large DC offset instead
        # of audio. Real speech is centred around zero.
        if abs(float(np.mean(audio))) > 0.25:
            return True
        return False

    @staticmethod
    def _mixdown(audio, channels: int):
        """Turn a (frames, channels) recording into one strong mono channel."""
        arr = np.asarray(audio, dtype=np.float32)
        if arr.size == 0:
            return None
        arr = arr.reshape(-1, channels)
        if channels > 1:
            with np.errstate(over="ignore", invalid="ignore"):
                rms = np.sqrt(np.mean(arr.astype(np.float64) ** 2, axis=0))
            if not np.all(np.isfinite(rms)):
                return None
            # Some laptop arrays only populate one capsule; take the live one.
            return arr[:, int(np.argmax(rms))].copy()
        return arr.reshape(-1)

    def _probe(self, index: int, seconds: float = 0.5):
        """Record a short sample and return its peak, or None if broken."""
        import sounddevice as sd
        info = sd.query_devices(index)
        sr = int(info["default_samplerate"])
        ch = 2 if info["max_input_channels"] >= 2 else 1
        audio = sd.rec(int(seconds * sr), samplerate=sr, channels=ch,
                       dtype="float32", device=index)
        sd.wait()
        flat = self._mixdown(audio, ch)
        if flat is None or self._broken(flat):
            return None
        return float(np.max(np.abs(flat)))

    def _probe_bounded(self, index: int, seconds: float = 0.4, timeout: float = 2.0):
        """Probe a device without letting a broken driver hang startup."""
        import queue
        q: "queue.Queue" = queue.Queue()

        def run():
            try:
                q.put(self._probe(index, seconds))
            except Exception:
                q.put(None)

        t = threading.Thread(target=run, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            print(f"[STT] Device {index} did not respond to a probe; skipping it.")
            return None
        try:
            return q.get_nowait()
        except Exception:
            return None

    def pick_device(self):
        """Use the Windows default input. Probing every device can hang on
        broken Bluetooth drivers, so selection itself never records."""
        if self._input_device is not None:
            return self._input_device

        import sounddevice as sd
        try:
            default_in = sd.default.device[0]
            info = sd.query_devices(default_in)
            if info["max_input_channels"] > 0:
                self._input_device = int(default_in)
                self.last_device_name = info["name"]
                print(f"[STT] Using Windows default mic {default_in} ({info['name']})")
                return self._input_device
        except Exception as e:
            print(f"[STT] Default mic unavailable: {e}")

        for d in self.devices():
            if not d["loopback"]:
                self._input_device = d["index"]
                self.last_device_name = d["name"]
                print(f"[STT] Falling back to mic {d['index']} ({d['name']})")
                return self._input_device

        self._input_device = None
        return None

    # -- recording ------------------------------------------------------
    def record(self, seconds: float = 6.0, device: Optional[int] = None) -> Optional[np.ndarray]:
        """Record mono audio and return it resampled to 16 kHz float32."""
        import sounddevice as sd
        dev = int(device) if device is not None else self.pick_device()
        try:
            info = sd.query_devices(dev) if dev is not None else sd.query_devices(kind="input")
        except Exception as e:
            print(f"[STT] Cannot query device {dev}: {e}")
            return None

        sr = int(info["default_samplerate"])
        ch = 2 if info["max_input_channels"] >= 2 else 1
        import threading

        def _watchdog():
            # Broken audio drivers can make sd.wait() never return.
            try:
                sd.stop()
                print(f"[STT] Watchdog stopped a stuck recording on device {dev}")
            except Exception:
                pass

        try:
            audio = sd.rec(int(seconds * sr), samplerate=sr, channels=ch,
                           dtype="float32", device=dev)
            timer = threading.Timer(seconds + 4.0, _watchdog)
            timer.daemon = True
            timer.start()
            try:
                sd.wait()
            finally:
                timer.cancel()
        except Exception as e:
            print(f"[STT] Record failed on device {dev} ({sr} Hz, {ch} ch): {e}")
            try:
                sr, ch = SAMPLE_RATE, 1
                audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1,
                               dtype="float32", device=dev)
                sd.wait()
            except Exception as e2:
                print(f"[STT] Record retry failed: {e2}")
                return None

        flat = self._mixdown(audio, ch)
        if flat is None or self._broken(flat):
            print(f"[STT] Device {dev} returned invalid audio")
            return None

        self.last_peak = float(np.max(np.abs(flat)))
        self.last_device_name = info["name"]

        if sr != SAMPLE_RATE:
            try:
                from math import gcd
                from scipy.signal import resample_poly
                g = gcd(sr, SAMPLE_RATE)
                flat = resample_poly(flat, SAMPLE_RATE // g, sr // g).astype(np.float32)
            except Exception as e:
                print(f"[STT] Resample failed: {e}")
                return None
        return flat

    # -- transcription --------------------------------------------------
    def transcribe(self, audio: np.ndarray) -> Optional[str]:
        if self.model is None or audio is None or audio.size == 0:
            return None

        peak = float(np.max(np.abs(audio)))
        self.last_peak = peak
        if peak < 0.0015:
            print(f"[STT] Audio too quiet (peak {peak:.5f}); not sending to Whisper")
            return None

        # Quiet laptop mics produce peaks around 0.005. Normalise so Whisper
        # and its voice-activity filter see a healthy signal.
        if peak < 0.3:
            gain = min(100.0, 0.9 / max(peak, 1e-6))
            audio = np.clip(audio * gain, -1.0, 1.0).astype(np.float32)

        try:
            segments, _ = self.model.transcribe(
                audio,
                language="en",
                vad_filter=True,
                vad_parameters={"threshold": 0.35, "min_silence_duration_ms": 400},
                beam_size=5,
                condition_on_previous_text=False,
            )
            text = " ".join(s.text.strip() for s in segments).strip()
            low = text.lower().strip(" .,!?:;")
            if low in {
                "thank you", "thanks for watching", "thank you for watching",
                "subtitles by", "you", "bye", "yeah", "okay", "uh", "um",
                "tell us", "like", "so", "please subscribe",
            }:
                return None
            print(f"[STT] Heard (peak {peak:.4f}, {self.last_device_name}): {text!r}")
            return text or None
        except Exception as e:
            print(f"[STT] Transcribe failed: {e}")
            return None

    def listen(self, seconds: float = 6.0) -> Optional[str]:
        return self.transcribe(self.record(seconds))


# ======================================================================
# Connection manager
# ======================================================================
class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def send(self, ws: WebSocket, obj: dict):
        try:
            await ws.send_text(json.dumps(obj))
        except Exception:
            self.disconnect(ws)

    async def broadcast(self, obj: dict):
        for ws in list(self.active):
            await self.send(ws, obj)


# ======================================================================
# Pipeline / command router
# ======================================================================
class Pipeline:
    CONTROL_HINTS = (
        "timer", "pause", "resume", "stop", "cancel", "next", "previous",
        "repeat", "help", "what can i say", "never mind", "done",
    )

    def __init__(self, session_obj: Optional[CookingSession] = None,
                 timer_manager: Optional[TimerManager] = None,
                 persist: Optional[bool] = None):
        self.connections = ConnectionManager()
        self.session: CookingSession = session_obj if session_obj is not None else session
        self.timers = timer_manager if timer_manager is not None else timers
        self.timers.on_expire = self._timer_expired_threadsafe
        self._persist = SESSION_PERSISTENCE if persist is None else bool(persist)
        self.pending_dish: Optional[str] = None
        self.pending_servings: int = 2
        self.paused = False
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.llm_ready = False
        self.llm_error = ""
        self.tts_busy = False
        self.voice_warned = False
        self._tts = None
        self._stt = None
        self._watch_task = None
        self._last_proactive = 0.0
        self._llm_checked_at = 0.0

    # -- lazy speech engines (tests can inject fakes) --------------------
    @property
    def tts(self):
        if self._tts is None:
            self._tts = TTSEngine()
        return self._tts

    @tts.setter
    def tts(self, value):
        self._tts = value

    @property
    def stt(self):
        if self._stt is None:
            self._stt = STTEngine()
        return self._stt

    @stt.setter
    def stt(self, value):
        self._stt = value

    # -- startup --------------------------------------------------------
    def _load_models(self):
        if STT_MODE != "browser":
            try:
                _ = self.stt
            except Exception as e:
                print(f"[Startup] STT failed to load: {e}")
        if TTS_MODE != "browser" and TTS_ENABLED:
            try:
                _ = self.tts
            except Exception as e:
                print(f"[Startup] TTS failed to load: {e}")

    async def startup(self):
        self.loop = asyncio.get_running_loop()
        print("[Startup] Loading speech models...")
        await asyncio.to_thread(self._load_models)
        await self._check_llm()
        if self._persist and load_session(self.session, SESSION_FILE):
            print(f"[Session] Resumed {self.session.selected_dish} "
                  f"at step {self.session.current_step_index + 1}.")
        self._watch_task = asyncio.create_task(self._progress_watch_loop())

    async def startup_connection(self):
        """Lightweight startup for an isolated public WebSocket session."""
        self.loop = asyncio.get_running_loop()
        await self._check_llm()
        self._watch_task = asyncio.create_task(self._progress_watch_loop())

    async def shutdown_connection(self):
        if self._watch_task is not None:
            self._watch_task.cancel()
            self._watch_task = None
        try:
            self.timers.cancel_all()
        except Exception:
            pass
        try:
            self.session.active_timers.clear()
        except Exception:
            pass
        if self._tts is not None:
            try:
                self._tts.stop()
            except Exception:
                pass

    async def _check_llm(self):
        try:
            ok, reason = await check_provider()
            self.llm_ready = bool(ok)
            self.llm_error = "" if ok else reason
            self._llm_checked_at = time.time()
            status = provider_status()
            if ok:
                print(f"[LLM] {status.get('provider')} {status.get('model')} ready")
            else:
                print(f"[LLM] not ready: {reason}")
        except Exception as e:
            self.llm_ready = False
            self.llm_error = str(e)[:200]
            self._llm_checked_at = time.time()
            print(f"[LLM] check failed: {e}")

    async def refresh_llm_status(self, max_age: float = 15.0):
        """Keep the AI status badge truthful without hammering the provider."""
        if time.time() - self._llm_checked_at >= max_age:
            await self._check_llm()

    def _save(self):
        if not self._persist:
            return
        try:
            save_session(self.session, SESSION_FILE)
        except Exception as e:
            print(f"[Session] save failed: {e}")

    # -- status / services ----------------------------------------------
    def tts_mode(self) -> str:
        if not TTS_ENABLED:
            return "off"
        if TTS_MODE == "browser":
            return "browser"
        if TTS_MODE == "server":
            return "server" if self.tts.available else "browser"
        return "server" if self.tts.available else "browser"

    def stt_mode(self) -> str:
        if STT_MODE == "browser":
            return "browser"
        return "server" if getattr(self._stt, "model", None) is not None else "browser"

    def _services_payload(self) -> dict:
        status = provider_status()
        mode = self.tts_mode()
        return {
            "type": "services",
            "backend": True,
            "llm": self.llm_ready,
            "llm_model": status.get("model", ""),
            "llm_provider": status.get("provider", ""),
            "llm_configured": bool(status.get("configured")),
            "llm_error": self.llm_error if not self.llm_ready else "",
            "tts": mode in ("server", "browser"),
            "tts_mode": mode,
            "stt": bool(self._stt is not None and getattr(self._stt, "model", None) is not None),
            "stt_mode": self.stt_mode(),
            "tts_enabled": TTS_ENABLED,
            "multi_user": MULTI_USER,
            "mic_device": getattr(self._stt, "last_device_name", "") if self._stt else "",
        }

    async def send_services(self, ws: Optional[WebSocket] = None):
        payload = self._services_payload()
        if ws is not None:
            await self.connections.send(ws, payload)
        else:
            await self.connections.broadcast(payload)

    async def send_status(self, ws: Optional[WebSocket] = None):
        payload = {"type": "status", **self.session.to_dict()}
        if ws is not None:
            await self.connections.send(ws, payload)
        else:
            await self.connections.broadcast(payload)
        await self.connections.broadcast({"type": "timers", "timers": self.timers.list()})

    async def say(self, text: str, ws: Optional[WebSocket] = None, speak: bool = True):
        text = (text or "").strip()
        if not text:
            return
        payload = {"type": "ai_speak", "text": text}
        if ws is not None:
            await self.connections.send(ws, payload)
        else:
            await self.connections.broadcast(payload)

        if not speak:
            return
        mode = self.tts_mode()
        if mode == "browser":
            # The UI speaks this with the browser's SpeechSynthesis API and
            # controls its own busy/listening state. The server must not send
            # speak_start/speak_done, because it cannot know when audio ends.
            return
        if mode == "off" or not self.tts.available:
            if not self.voice_warned:
                self.voice_warned = True
                await self.connections.broadcast({
                    "type": "voice_unavailable",
                    "text": "Voice output is unavailable. Instructions are shown on screen.",
                })
            return

        await self.connections.broadcast({"type": "speak_start"})
        self.tts_busy = True
        try:
            ok = await self.tts.speak(text)
        except Exception as e:
            print(f"[TTS] speak failed: {e}")
            ok = False
        finally:
            self.tts_busy = False
            await self.connections.broadcast({"type": "speak_done"})
        if ok is False:
            await self.connections.broadcast({
                "type": "voice_error",
                "text": "Voice playback failed. The text is on screen.",
            })

    async def stop_speaking(self, ws: Optional[WebSocket] = None):
        try:
            self.tts.stop()
        except Exception:
            pass
        self.tts_busy = False
        await self.connections.broadcast({"type": "speak_done"})
        if ws is not None:
            await self.connections.send(ws, {"type": "voice_stopped", "text": "Speech stopped."})

    # -- timer expiry ---------------------------------------------------
    def _timer_expired_threadsafe(self, timer_id: str, label: str):
        # Remove the session record immediately, even if the event loop is not
        # running yet (unit tests, or a timer started during startup).
        self.session.remove_timer(timer_id)
        if self.loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._on_timer_done(timer_id, label), self.loop)

    async def _on_timer_done(self, timer_id: str, label: str):
        self.session.remove_timer(timer_id)
        await self.connections.broadcast({"type": "timers", "timers": self.timers.list()})
        if label:
            msg = f"Your {label} is done. Please check the dish now."
        else:
            msg = "Your timer is done. Please check the dish now."
        await self.connections.broadcast({"type": "timer_done", "id": timer_id,
                                          "label": label, "text": msg})
        await self.say(msg)
        await self.send_status()
        self._save()

    # -- websocket lifecycle -------------------------------------------
    async def on_connect(self, ws: WebSocket):
        is_first = len(self.connections.active) == 0
        await self.connections.connect(ws)
        await self.send_status(ws)
        await self.send_services(ws)
        text = ("Hi! I'm your AI Chef. What would you like to cook today? "
                "You can also tell me the ingredients you have.")
        await self.say(text, ws, speak=is_first)

    async def on_message(self, ws: WebSocket, data: dict):
        kind = data.get("type")
        if kind == "command":
            text = str(data.get("text") or "")[:MAX_MESSAGE_CHARS]
            await self.handle_text(text, ws)
        elif kind == "pause":
            self.paused = True
            self.session.paused = True
            await self.say("Paused. Take your time. Say resume when you are ready.", ws)
            await self.send_status(ws)
        elif kind == "resume":
            self.paused = False
            self.session.paused = False
            await self.say("Resumed. Say repeat to hear the current step.", ws)
            await self.send_status(ws)
        elif kind == "stop":
            await self.stop_session(ws)
        elif kind == "new_session":
            await self.new_session(ws)
        elif kind == "get_status":
            await self.send_status(ws)
            await self.send_services(ws)
        elif kind == "stop_voice":
            await self.stop_speaking(ws)
        elif kind == "start_dish":
            dish = (data.get("dish") or "").strip()
            servings = int(data.get("servings") or 2)
            if dish:
                await self.start_recipe(dish, servings, ws)
        elif kind == "ping":
            await self.connections.send(ws, {"type": "pong"})

    async def stop_session(self, ws: WebSocket):
        try:
            self.tts.stop()
        except Exception:
            pass
        self.tts_busy = False
        self.timers.cancel_all()
        self.session.active_timers.clear()
        self.session.end(stopped=True)
        self.paused = False
        self.session.paused = False
        await self.connections.broadcast({"type": "session_stopped"})
        await self.connections.broadcast({"type": "timers", "timers": []})
        await self.say("Session stopped. Timers are cleared. Say new dish when you want to cook again.", ws)
        await self.send_status(ws)
        self._save()

    async def new_session(self, ws: WebSocket):
        try:
            self.tts.stop()
        except Exception:
            pass
        self.timers.cancel_all()
        self.session.reset()
        self.pending_dish = None
        self.pending_servings = 2
        self.paused = False
        await self.connections.broadcast({"type": "session_reset"})
        await self.connections.broadcast({"type": "timers", "timers": []})
        await self.say("Fresh start. What would you like to cook?", ws)
        await self.send_status(ws)
        self._save()

    # ==================================================================
    # Main router
    # ==================================================================
    async def handle_text(self, raw: str, ws: WebSocket):
        text = (raw or "").strip()
        if not text:
            return
        low = text.lower()
        s = self.session
        s.last_user_response = text
        s.record_message("user", text)
        await self.connections.send(ws, {"type": "transcript", "speaker": "user", "text": text})

        # ---- New dish / reset ----
        if any(p in low for p in ("new dish", "start over", "new recipe",
                                  "cook something else", "naya dish", "nayi dish")):
            await self.new_session(ws)
            return

        # ---- Explicit request for a different dish, at any point ----
        # This runs before the awaiting-state branches. Without it, a request
        # made while a recipe is on screen gets treated as a confirmation or
        # cooking follow-up: the panel keeps the old dish while the chef says
        # something unrelated. That is the mismatch users see.
        if s.has_recipe:
            allow_bare = not (s.awaiting in ("variant", "servings", "timer"))
            new_dish = self._explicit_new_dish(text, allow_bare=allow_bare)
            if new_dish:
                current = normalize_name(s.selected_dish)
                new_key = normalize_name(new_dish)
                is_switch = (not current) or (
                    new_key != current
                    and new_key not in current
                    and current not in new_key
                )
                if is_switch:
                    servings = s.parse_servings(text) or s.servings or 2
                    # Old timers belong to the old dish.
                    self.timers.cancel_all()
                    s.active_timers.clear()
                    await self.connections.broadcast({"type": "timers", "timers": []})
                    await self.say(f"Switching to {smart_title(new_dish)}.", ws)
                    await self.start_recipe(new_dish, servings, ws)
                    return

        # ---- Awaiting states (clarification responses) ----
        # These take priority over intent routing since the user is
        # answering a specific question we asked.

        # ---- clarifying a variant ----
        if s.awaiting == "variant" and self.pending_dish:
            combined = apply_variant(self.pending_dish, text)
            if combined:
                servings = self.pending_servings
                self.pending_dish = None
                s.awaiting = "dish"
                await self.start_recipe(combined, servings, ws)
            else:
                await self.say(clarify_dish(self.pending_dish) or "Please choose one of those options.", ws)
            return

        # ---- waiting for a timer duration ----
        if s.awaiting == "timer":
            minutes = s.parse_timer_minutes(text)
            if minutes is None and text.strip().isdigit():
                minutes = int(text.strip())
            if minutes is None:
                await self.say("Just tell me the number of minutes, for example 10.", ws)
                return
            s.awaiting = "cooking"
            await self._create_timer(minutes, ws)
            return

        # ---- waiting for servings ----
        if s.awaiting == "servings" and self.pending_dish:
            if any(w in low for w in ("cancel", "never mind", "forget it", "stop")):
                self.pending_dish = None
                s.awaiting = "dish"
                await self.say("No problem. What would you like to cook?", ws)
                return
            bare = re.fullmatch(r"\s*(\d{1,2})\s*", text)
            servings = int(bare.group(1)) if bare else s.parse_servings(text)
            if servings is None:
                await self.say("Just tell me a number, like 4.", ws)
                return
            dish = self.pending_dish
            self.pending_dish = None
            s.awaiting = "dish"
            await self.start_recipe(dish, servings, ws)
            return

        # ---- recipe shown, waiting for confirmation ----
        if s.awaiting == "confirm":
            if self._asks_for_full_recipe(low):
                await self._speak_full_recipe(ws)
                return
            intent = classify_intent(text)
            if intent.intent in (GENERAL_CHAT, RECIPE_INFORMATION):
                direct = conversational_response(text, intent)
                answer = direct or await self._answer_general(text, ws)
                await self.say(answer, ws)
                await self.send_status(ws)
                return
            if any(p in low for p in ("ready", "start", "begin", "chalo", "shuru",
                                      "yes", "ok", "okay", "haan", "proceed", "cook")):
                await self._begin_cooking(ws)
            elif any(p in low for p in ("missing", "don't have", "do not have",
                                        "nahi hai", "out of", "khatam")):
                await self._handle_missing(text, low, ws)
            elif any(p in low for p in ("change", "different", "another dish", "cancel")):
                s.reset()
                self.pending_dish = None
                await self.say("Okay. What would you like to cook instead?", ws)
                await self.send_status(ws)
            else:
                await self.say("Say ready to start, or tell me which ingredient is missing.", ws)
            return

        # ---- confirming a skip ----
        if s.awaiting == "confirm_skip":
            if any(w in low for w in ("yes", "skip", "continue", "haan", "go ahead")):
                s.awaiting = "cooking"
                s.conditions.pop("current_task", None)
                await self._advance(ws, force=True)
            elif any(w in low for w in ("no", "wait", "not yet", "ruko")):
                s.awaiting = "cooking"
                await self.say("Okay, we will stay here. Tell me when it is ready.", ws)
            else:
                await self.say("Should I move to the next step anyway? Say yes or no.", ws)
            return

        # ---- finished session ----
        if s.awaiting == "finished":
            if any(w in low for w in ("new", "another", "yes", "something else", "aur")):
                await self.new_session(ws)
            else:
                await self.say("Thanks for cooking with me. Say new dish whenever you are ready.", ws)
            return

        # ---- If there's an active cooking session, route to cooking commands first ----
        if s.session_started and s.has_recipe:
            await self.handle_cooking_command(text, low, ws)
            return

        # ---- Intent routing (only when not in active cooking) ----
        intent = classify_intent(text)

        # ---- Unsafe requests ----
        if intent.intent == UNSAFE_OR_UNSUPPORTED_REQUEST:
            await self.say("I can't advise leaving cooking unattended. For safety, "
                           "never leave hot oil, boiling water, or an open flame unwatched.",
                           ws)
            return

        # ---- General conversation (no cooking context) ----
        if intent.intent == GENERAL_CHAT:
            # Direct responses for common greetings
            direct = conversational_response(text, intent)
            if direct:
                await self.say(direct, ws)
                await self.send_status(ws)
                return
            # Let the LLM handle general questions without cooking context
            answer = await self._answer_general(text, ws)
            await self.say(answer, ws)
            await self.send_status(ws)
            return

        # ---- Recipe information (no cooking session started) ----
        if intent.intent == RECIPE_INFORMATION:
            dish = intent.dish or self._extract_dish(text)
            if dish:
                # Try RAG first
                rag_results = retrieve_for_dish(dish, top_k=3)
                if rag_results:
                    context = "\n".join(f"- {r['text'][:300]}" for r in rag_results[:2])
                    answer = await self._answer_with_rag(text, dish, context, ws)
                else:
                    answer = await self._answer_general(text, ws, dish_context=dish)
                await self.say(answer, ws)
                await self.send_status(ws)
                return
            # No specific dish mentioned, general cooking question
            answer = await self._answer_general(text, ws)
            await self.say(answer, ws)
            await self.send_status(ws)
            return

        # ---- Recipe request ----
        if intent.intent == RECIPE_REQUEST:
            # Use the full extracted dish name (includes variant words like "chicken")
            dish = self._extract_dish(text) or intent.dish
            servings = intent.servings or s.parse_servings(text) or 2
            if not dish or self._is_generic_dish(dish):
                question = clarify_dish(dish) if dish else None
                if question:
                    self.pending_dish = ambiguous_base(dish) or dish
                    self.pending_servings = servings
                    s.awaiting = "variant"
                    await self.say(question, ws)
                    return
                await self.say("Which dish would you like to cook? For example, "
                               "chicken biryani, pasta, poha, or dal.", ws)
                return
            question = clarify_dish(dish)
            if question and not self._has_variant(dish):
                self.pending_dish = ambiguous_base(dish) or dish
                self.pending_servings = servings
                s.awaiting = "variant"
                await self.say(question, ws)
                return
            await self.start_recipe(dish, servings, ws)
            return

        # ---- Ingredient-first suggestions ----
        if intent.intent == INGREDIENT_BASED_SUGGESTION:
            await self.suggest_dishes(text, ws)
            return

        # ---- Timer commands (no active session) ----
        if intent.intent == TIMER_COMMAND:
            if intent.timer_minutes:
                await self._create_timer(intent.timer_minutes, ws)
            else:
                s.awaiting = "timer"
                await self.say("How many minutes should the timer run?", ws)
            return

        # ---- Repeat / clarify ----
        if intent.intent == REPEAT_OR_CLARIFY:
            await self.say("I'm not in a cooking session yet. What would you like to cook?", ws)
            return

        # ---- Cooking follow-up without active session ----
        if intent.intent == COOKING_FOLLOW_UP:
            await self.say("We are not cooking right now. Tell me what you'd like to cook, "
                           "or ask me about a dish.", ws)
            await self.send_status(ws)
            return

        if self.paused:
            if any(p in low for p in ("resume", "continue", "move on", "chalo", "aage")):
                self.paused = False
                s.paused = False
                msg = "Resumed. "
                if s.current_step:
                    msg += f"Current step: {s.current_step}"
                else:
                    msg += "What would you like to cook?"
                await self.say(msg, ws)
                await self.send_status(ws)
            elif any(p in low for p in ("stop", "end session", "band")):
                await self.stop_session(ws)
            else:
                await self.say("Still paused. Say resume when you are ready.", ws)
            return

        # ---- clarifying a variant ----
        if s.awaiting == "variant" and self.pending_dish:
            combined = apply_variant(self.pending_dish, text)
            if combined:
                servings = self.pending_servings
                self.pending_dish = None
                s.awaiting = "dish"
                await self.start_recipe(combined, servings, ws)
            else:
                await self.say(clarify_dish(self.pending_dish) or "Please choose one of those options.", ws)
            return

        # ---- waiting for a timer duration ----
        if s.awaiting == "timer":
            minutes = s.parse_timer_minutes(text)
            if minutes is None and text.strip().isdigit():
                minutes = int(text.strip())
            if minutes is None:
                await self.say("Just tell me the number of minutes, for example 10.", ws)
                return
            s.awaiting = "cooking"
            await self._create_timer(minutes, ws)
            return

        # ---- waiting for servings ----
        if s.awaiting == "servings" and self.pending_dish:
            if any(w in low for w in ("cancel", "never mind", "forget it", "stop")):
                self.pending_dish = None
                s.awaiting = "dish"
                await self.say("No problem. What would you like to cook?", ws)
                return
            bare = re.fullmatch(r"\s*(\d{1,2})\s*", text)
            servings = int(bare.group(1)) if bare else s.parse_servings(text)
            if servings is None:
                await self.say("Just tell me a number, like 4.", ws)
                return
            dish = self.pending_dish
            self.pending_dish = None
            s.awaiting = "dish"
            await self.start_recipe(dish, servings, ws)
            return

        # ---- recipe shown, waiting for confirmation ----
        if s.awaiting == "confirm":
            if self._asks_for_full_recipe(low):
                await self._speak_full_recipe(ws)
                return
            intent = classify_intent(text)
            if intent.intent in (GENERAL_CHAT, RECIPE_INFORMATION):
                direct = conversational_response(text, intent)
                answer = direct or await self._answer_general(text, ws)
                await self.say(answer, ws)
                await self.send_status(ws)
                return
            if any(p in low for p in ("ready", "start", "begin", "chalo", "shuru",
                                      "yes", "ok", "okay", "haan", "proceed", "cook")):
                await self._begin_cooking(ws)
            elif any(p in low for p in ("missing", "don't have", "do not have",
                                        "nahi hai", "out of", "khatam")):
                await self._handle_missing(text, low, ws)
            elif any(p in low for p in ("change", "different", "another dish", "cancel")):
                s.reset()
                self.pending_dish = None
                await self.say("Okay. What would you like to cook instead?", ws)
                await self.send_status(ws)
            else:
                await self.say("Say ready to start, or tell me which ingredient is missing.", ws)
            return

        # ---- confirming a skip ----
        if s.awaiting == "confirm_skip":
            if any(w in low for w in ("yes", "skip", "continue", "haan", "go ahead")):
                s.awaiting = "cooking"
                s.conditions.pop("current_task", None)
                await self._advance(ws, force=True)
            elif any(w in low for w in ("no", "wait", "not yet", "ruko")):
                s.awaiting = "cooking"
                await self.say("Okay, we will stay here. Tell me when it is ready.", ws)
            else:
                await self.say("Should I move to the next step anyway? Say yes or no.", ws)
            return

        # ---- finished session ----
        if s.awaiting == "finished":
            if any(w in low for w in ("new", "another", "yes", "something else", "aur")):
                await self.new_session(ws)
            else:
                await self.say("Thanks for cooking with me. Say new dish whenever you are ready.", ws)
            return

        # ---- no recipe yet: dish-first or ingredients-first ----
        if not s.has_recipe:
            if self._looks_like_ingredients(low):
                await self.suggest_dishes(text, ws)
                return
            if any(h in low for h in self.CONTROL_HINTS):
                await self.say("Let's pick a dish first. Tell me what you'd like to cook, "
                               "or list the ingredients you have.", ws)
                return
            dish = self._extract_dish(text)
            if not dish or self._is_generic_dish(dish):
                await self.say("Which dish would you like to cook? For example, "
                               "chicken biryani, pasta, poha, or dal.", ws)
                return
            servings = s.parse_servings(text)
            question = clarify_dish(dish)
            if question and not self._has_variant(dish):
                self.pending_dish = ambiguous_base(dish) or dish
                self.pending_servings = servings or 2
                s.awaiting = "variant"
                await self.say(question, ws)
                return
            if servings is None:
                self.pending_dish = dish
                self.pending_servings = 2
                s.awaiting = "servings"
                await self.say(f"{smart_title(dish)} sounds great. For how many people?", ws)
                return
            await self.start_recipe(dish, servings, ws)
            return

        # ---- cooking state: navigate and answer ----
        await self.handle_cooking_command(text, low, ws)

    # ==================================================================
    # Recipe generation
    # ==================================================================
    async def start_recipe(self, dish: str, servings: int, ws: WebSocket,
                           missing: Optional[List[str]] = None):
        dish = (dish or "").strip()
        if not dish:
            await self.say("Tell me which dish you want to cook.", ws)
            return
        servings = max(1, min(20, int(servings or 2)))

        question = clarify_dish(dish)
        if question and not self._has_variant(dish):
            self.pending_dish = ambiguous_base(dish) or dish
            self.pending_servings = servings
            self.session.awaiting = "variant"
            await self.say(question, ws)
            return

        # Replacing a different dish: clear the old recipe panel and timers
        # first, so the screen can never keep showing dish A while the chef
        # talks about dish B (including when generation fails).
        if self.session.has_recipe and \
                normalize_name(self.session.selected_dish) != normalize_name(dish):
            self.timers.cancel_all()
            self.session.active_timers.clear()
            await self.connections.broadcast({"type": "timers", "timers": []})
            self.session.reset()
            await self.send_status(ws)

        await self.say(f"Let me put together {smart_title(dish)} for {servings}. One moment.", ws)
        try:
            recipe = await self.generate_recipe(dish, servings)
        except RecipeUnavailable as e:
            self.session.awaiting = "dish"
            await self.connections.send(ws, {"type": "error", "text": str(e)})
            await self.say(str(e), ws)
            await self.send_status(ws)
            return
        except Exception as e:
            print(f"[Recipe] unexpected failure: {e}")
            msg = "I could not build a reliable recipe just now. Please try again."
            self.session.awaiting = "dish"
            await self.connections.send(ws, {"type": "error", "text": msg})
            await self.say(msg, ws)
            return

        self.session.prepare(
            dish=recipe["dish"],
            servings=recipe["servings"],
            ingredients=recipe["ingredients"],
            steps=recipe["steps"],
            duration_minutes=recipe.get("duration_minutes", 0),
            prep_time_minutes=recipe.get("prep_time_minutes", 0),
            cook_time_minutes=recipe.get("cook_time_minutes", 0),
            total_time_minutes=recipe.get("total_time_minutes", recipe.get("duration_minutes", 0)),
            safety=recipe.get("safety", []),
            source=recipe.get("source", "ai"),
            missing=missing,
        )
        await self.connections.send(ws, {
            "type": "recipe",
            "dish": recipe["dish"],
            "servings": recipe["servings"],
            "ingredients": recipe["ingredients"],
            "steps": recipe["steps"],
            "duration_minutes": recipe.get("duration_minutes", 0),
            "total_time_minutes": recipe.get("total_time_minutes", recipe.get("duration_minutes", 0)),
            "prep_time_minutes": recipe.get("prep_time_minutes", 0),
            "cook_time_minutes": recipe.get("cook_time_minutes", 0),
            "waiting_time_minutes": recipe.get("waiting_time_minutes", 0),
            "safety": recipe.get("safety", []),
            "source": recipe.get("source", "ai"),
            "source_type": recipe.get("source_type", "ai_generated"),
            "validation_status": recipe.get("validation_status", "validated"),
            "missing": missing or [],
        })
        await self.send_status(ws)

        honesty = ""
        if recipe.get("source") != "ai":
            honesty = (f" I could not get a reliable AI recipe, so I am using my "
                       f"tested built-in {recipe['dish']} recipe.")
        ing = ", ".join(f"{i['quantity']} {i['name']}" for i in recipe["ingredients"][:6])
        safety = recipe["safety"][0] if recipe.get("safety") else ""
        msg = (f"Here is what you need for {recipe['dish']}, "
               f"for {recipe['servings']} servings: {ing}."
               f"{honesty} Tell me if anything is missing, or say ready to start.")
        if safety:
            msg += " " + safety
        await self.say(msg, ws)
        await self.send_status(ws)
        self._save()

    async def generate_recipe(self, dish: str, servings: int) -> dict:
        req_id = uuid.uuid4().hex[:8]
        print(f"[Recipe:{req_id}] start dish={dish!r} servings={servings}")

        # Ground the model in local culinary knowledge when, and only when,
        # the retrieved documents genuinely mention this dish.
        context = ""
        try:
            rag_hits = retrieve_for_dish(dish, top_k=3)
            print(f"[Recipe:{req_id}] rag_results={len(rag_hits)}")
            if rag_hits:
                context = "\n".join(
                    f"- {str(r.get('text', ''))[:400]}" for r in rag_hits[:2]
                )
        except Exception as e:
            print(f"[Recipe:{req_id}] rag unavailable: {e}")

        prompt = build_recipe_prompt(dish, servings, context=context)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            out = await llm_chat(messages, json_mode=True, max_tokens=900,
                                 temperature=0.2)
            self.llm_ready = True
        except Exception as e:
            print(f"[Recipe:{req_id}] provider failure: {e}")
            self.llm_ready = False
            return self._fallback_or_raise(dish, servings, ai_failed=True,
                                           request_id=req_id)

        data = parse_json_loose(out)
        if data is None:
            print(f"[Recipe:{req_id}] provider returned non-JSON output")
        try:
            recipe = validate_recipe(data, dish, servings)
            print(f"[Recipe:{req_id}] path=ai status=ok")
            return recipe
        except RecipeValidationError as e:
            reason = str(e)
            print(f"[Recipe:{req_id}] ai validation failed: {reason}")

        # One bounded repair attempt, naming the exact failure.
        try:
            repaired = await llm_chat(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": build_repair_prompt(
                     dish, servings, out, reason=reason)}],
                json_mode=True, max_tokens=900, temperature=0.1,
            )
            recipe = validate_recipe(parse_json_loose(repaired), dish, servings)
            print(f"[Recipe:{req_id}] path=ai_repair status=ok")
            return recipe
        except Exception as e:
            print(f"[Recipe:{req_id}] repair failed: {e}")

        return self._fallback_or_raise(dish, servings, ai_failed=False,
                                       request_id=req_id)

    def _fallback_or_raise(self, dish: str, servings: int, ai_failed: bool,
                           request_id: str = "") -> dict:
        fallback = build_fallback(dish, servings)
        if fallback is not None:
            print(f"[Recipe:{request_id}] path=builtin status=ok dish={fallback['dish']!r}")
            return fallback
        if ai_failed:
            print(f"[Recipe:{request_id}] path=none status=provider_error")
            raise RecipeUnavailable(
                f"I could not reach the AI that builds new recipes, and I do not "
                f"have a tested built-in recipe for {smart_title(dish)}. "
                f"{self._provider_hint()} Then try again."
            )
        print(f"[Recipe:{request_id}] path=none status=invalid_output")
        raise RecipeUnavailable(
            f"The AI returned an unreliable recipe for {smart_title(dish)}, and I do "
            "not have a tested built-in version. Please try rephrasing the dish or retry."
        )

    @staticmethod
    def _provider_hint() -> str:
        status = provider_status()
        if status.get("provider") == "ollama":
            return "Please check that Ollama is running."
        return "The hosted AI service is temporarily unavailable; please retry shortly."

    # ==================================================================
    # Ingredient-first suggestions
    # ==================================================================
    async def suggestions_for(self, ingredients: str) -> List[dict]:
        suggestions: List[dict] = []
        try:
            prompt = (
                "Return ONLY valid JSON: "
                '{"suggestions":[{"dish":"name","uses":["item"],"missing":["item"]}]}.'
                f" The user has: {ingredients}. Give 3 simple home dishes that use those "
                "ingredients. Only list truly missing essentials."
            )
            out = await llm_chat(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": prompt}],
                json_mode=True, max_tokens=350, temperature=0.4,
            )
            self.llm_ready = True
            data = parse_json_loose(out) or {}
            for s in data.get("suggestions", [])[:3]:
                dish = str(s.get("dish") or "").strip()
                if dish and not self._is_generic_dish(dish):
                    suggestions.append({
                        "dish": dish,
                        "uses": [str(x) for x in s.get("uses", [])][:6],
                        "missing": [str(x) for x in s.get("missing", [])][:6],
                    })
        except Exception as e:
            print(f"[Suggest] LLM failed: {e}")
            self.llm_ready = False

        for fallback in suggest_from_ingredients(ingredients):
            if len(suggestions) >= 3:
                break
            if all(normalize_name(fallback["dish"]) != normalize_name(x["dish"])
                   for x in suggestions):
                suggestions.append(fallback)
        return suggestions[:3]

    async def suggest_dishes(self, ingredients: str, ws: WebSocket):
        suggestions = await self.suggestions_for(ingredients)
        await self.connections.send(ws, {"type": "suggestions", "suggestions": suggestions})
        names = ", ".join(s["dish"] for s in suggestions)
        await self.say(f"You could make {names}. Which one sounds good?", ws)

    # ==================================================================
    # Cooking commands
    # ==================================================================
    async def _begin_cooking(self, ws: WebSocket):
        s = self.session
        s.begin()
        self.paused = False
        await self.send_status(ws)
        msg = f"Great. Step 1 of {s.step_count}. {s.current_step}"
        if s.safety:
            msg += " " + s.safety[0]
        await self.say(msg, ws)
        self._save()

    async def _advance(self, ws: WebSocket, force: bool = False):
        s = self.session
        task = s.current_task
        if (not force and s.conditions.get("current_task") == "not_ready"
                and task is not None and task["status"] != "done"):
            s.awaiting = "confirm_skip"
            await self.say("You said this is not ready yet. Do you want to move on anyway?", ws)
            return
        s.conditions.pop("current_task", None)
        was_last = s.current_step_index >= s.step_count - 1
        new_step = s.next_step()
        await self.send_status(ws)
        if was_last or s.finished:
            await self.say("That was the last step. Turn off the heat, plate the dish and "
                           "taste it. Say new dish to cook again.", ws)
        else:
            await self.say(f"Step {s.current_step_index + 1} of {s.step_count}. {new_step}", ws)
        self._save()

    async def _next_guidance(self, ws: WebSocket):
        s = self.session
        if not s.steps:
            await self.say("Let's pick a dish first.", ws)
            return
        idx = s.current_step_index
        current = (s.current_step or "").rstrip(".")
        msg = f"Right now: {current}."
        if idx + 1 < len(s.steps):
            following = s.steps[idx + 1].rstrip(".")
            msg += f" Then: {following}."
        await self.say(msg, ws)

    async def _create_timer(self, minutes: int, ws: WebSocket, label: Optional[str] = None):
        total = int(minutes * 60)
        label = label or f"{minutes}-minute timer"
        tid = self.timers.create(total, label=label)
        self.session.add_timer(tid, total, label)
        # Show the timer immediately; do not make the UI wait for the speech.
        await self.connections.broadcast({"type": "timers", "timers": self.timers.list()})
        await self.say(f"Timer set for {minutes} minute{'s' if minutes != 1 else ''}. "
                       "I will tell you when it is done.", ws)
        await self.send_status(ws)
        self._save()

    async def _handle_missing(self, text: str, low: str, ws: WebSocket):
        s = self.session
        m = re.search(
            r"(?:don'?t have|do not have|out of|no more|nahi hai|khatam|without)\s+"
            r"([a-zA-Z ]{2,40})",
            low,
        )
        ingredient = (m.group(1).strip() if m else "").strip(" .,!")
        if ingredient:
            s.missing_ingredients.append(ingredient)
        answer = await self.ask_llm(text, ws, substitution=True)
        await self.say(answer, ws)
        await self.send_status(ws)

    _TYPO_WORDS = {
        "nxt": "next", "nxtf": "next", "nxtt": "next", "nex": "next",
        "nextt": "next", "nxtstep": "next step", "nxet": "next",
        "prevoius": "previous", "prevous": "previous", "privious": "previous",
        "prev": "previous", "backk": "back", "repat": "repeat",
        "repeatt": "repeat", "repet": "repeat", "repeta": "repeat",
        "finsh": "finish", "finishh": "finish", "finnish": "finish",
        "ingrediants": "ingredients", "ingridents": "ingredients",
        "ingredents": "ingredients", "recipee": "recipe", "reciepe": "recipe",
        "timmer": "timer", "timr": "timer", "stpo": "stop", "stopp": "stop",
    }

    @classmethod
    def _normalise_typos(cls, low: str) -> str:
        words = []
        for token in low.split():
            cleaned = re.sub(r"[^a-z0-9']", "", token)
            words.append(cls._TYPO_WORDS.get(cleaned, token))
        return " ".join(words)

    @staticmethod
    def _asks_for_full_recipe(low: str) -> bool:
        asks = any(w in low for w in ("give", "show", "tell", "list", "what are",
                                      "read", "see", "again", "full"))
        subject = ("ingredient" in low or "recipe" in low or "step" in low
                   or "instruction" in low)
        return asks and subject

    async def _speak_full_recipe(self, ws: WebSocket):
        s = self.session
        if not s.has_recipe:
            await self.say("We do not have a recipe yet. Tell me what you would like to cook.", ws)
            return
        ing = "; ".join(f"{i.get('quantity', '')} {i.get('name', '')}".strip()
                        for i in s.ingredients)
        steps = " ".join(f"Step {i + 1}: {step}" for i, step in enumerate(s.steps))
        await self.say(f"{s.selected_dish} for {s.servings} servings. Ingredients: {ing}.", ws)
        await self.say(f"Method: {steps}", ws)
        await self.send_status(ws)

    async def _finish_dish(self, ws: WebSocket):
        """Guide the remaining steps in order without losing the real position."""
        s = self.session
        if not s.has_recipe:
            await self.say("Let's pick a dish first.", ws)
            return
        idx = max(0, min(s.current_step_index, len(s.steps) - 1))
        remaining = s.steps[idx:]
        if not remaining:
            await self.say("That was the last step. Your dish is ready to plate.", ws)
            return
        spoken = " ".join(
            f"Step {idx + i + 1}: {step}" for i, step in enumerate(remaining)
        )
        await self.say(
            f"Continuing from step {idx + 1} of {len(s.steps)}. {spoken} "
            "Say next as you complete each one.", ws
        )
        await self.send_status(ws)
        self._save()

    async def handle_cooking_command(self, text: str, low: str, ws: WebSocket):
        s = self.session
        low = self._normalise_typos(low)

        # ---- finish / wrap up, starting from the real current step ----
        if re.search(r"\b(let'?s\s+)?(finish|wrap up|complete)\b", low) \
                or "finish this dish" in low or "finish the dish" in low:
            if any(w in low for w in ("finish", "wrap", "complete")):
                await self._finish_dish(ws)
                return

        # ---- finish the whole dish ----
        if re.search(r"\b(i'?m done|i am done|finished cooking|dish is ready|"
                     r"khaana ready|sab ho gaya)\b", low):
            s.end()
            self.timers.cancel_all()
            s.active_timers.clear()
            await self.connections.broadcast({"type": "timers", "timers": []})
            await self.say("Wonderful work! Your dish is ready. Turn off the heat and taste it. "
                           "Say new dish to cook again.", ws)
            await self.send_status(ws)
            self._save()
            return

        # ---- timers ----
        sec_match = re.search(r"(\d+)\s*seconds?\b", low)
        if "timer" in low and any(w in low for w in ("remaining", "left", "status", "how much")):
            rows = self.timers.list()
            if not rows:
                await self.say("No timers are running right now.", ws)
            else:
                bits = []
                for t in rows:
                    secs = int(t["remaining"])
                    bits.append(f"{t['label']} with {secs // 60} minutes {secs % 60} seconds left")
                await self.say("You have " + "; ".join(bits) + ".", ws)
            return
        if "cancel timer" in low or "stop timer" in low:
            self.timers.cancel_all()
            s.active_timers.clear()
            await self.connections.broadcast({"type": "timers", "timers": []})
            await self.say("All timers cancelled.", ws)
            await self.send_status(ws)
            self._save()
            return

        # ---- stop the whole session (typed or voice) ----
        if low.strip(" .!") in ("stop", "stop cooking", "stop session", "end session") \
                or any(p in low for p in ("stop cooking", "end session", "band karo")):
            await self.stop_session(ws)
            return
        if "timer" in low and ("set" in low or "start" in low or sec_match
                               or s.parse_timer_minutes(text)):
            if sec_match:
                total_seconds = int(sec_match.group(1))
                label = f"{total_seconds}-second timer"
                spoken = f"{total_seconds} second{'s' if total_seconds != 1 else ''}"
                tid = self.timers.create(total_seconds, label=label)
                s.add_timer(tid, total_seconds, label)
                # Show the timer immediately; do not make the UI wait for the speech.
                await self.connections.broadcast({"type": "timers", "timers": self.timers.list()})
                await self.say(f"Timer set for {spoken}. I will tell you when it is done.", ws)
                await self.send_status(ws)
                self._save()
                return
            minutes = s.parse_timer_minutes(text)
            if not minutes:
                s.awaiting = "timer"
                await self.say("How many minutes should the timer run?", ws)
                return
            await self._create_timer(minutes, ws)
            return

        # ---- pause / a short break ----
        if any(p in low for p in ("wait", "give me a minute", "hold on", "pause",
                                  "ruko", "ek minute", "two minutes", "2 minutes")) \
                and not text.strip().endswith("?"):
            minutes = s.parse_timer_minutes(text)
            self.paused = True
            s.paused = True
            if minutes:
                tid = self.timers.create(minutes * 60, label=f"{minutes}-minute break")
                s.add_timer(tid, minutes * 60, f"{minutes}-minute break")
                await self.connections.broadcast({"type": "timers", "timers": self.timers.list()})
                await self.say(f"Okay, I will wait {minutes} minutes. I will tell you when "
                               "the time is up. Say resume if you are ready sooner.", ws)
            else:
                await self.say("No problem. Take your time. Say resume when you are ready.", ws)
            await self.send_status(ws)
            self._save()
            return

        # ---- full ingredients + method from the authoritative session ----
        if self._asks_for_full_recipe(low):
            await self._speak_full_recipe(ws)
            return

        # ---- normal conversation during cooking (never advances the step) ----
        intent = classify_intent(text)
        if intent.intent == GENERAL_CHAT:
            direct = conversational_response(text, intent)
            if direct:
                await self.say(direct, ws)
                await self.send_status(ws)
                return

        # ---- task confirmations ----
        cue = s.detect_task_cue(text)
        if cue == "not_ready":
            task = s.current_task
            if task is not None:
                task["status"] = "started"
                if not task["started_at"]:
                    task["started_at"] = time.time()
                task["note"] = "reported not ready"
                task["asked"] = 0
                task["last_ask"] = None
                task["ask_after"] = max(120, task["ask_after"] // 2)
            s.set_condition("current_task", "not_ready")
            await self.say("Okay, no rush. Give it a little more time. Make sure the flame is "
                           "on and never leave it unattended. I will not move on yet.", ws)
            await self.send_status(ws)
            return

        if cue == "started":
            task = s.start_task(note=text[:80])
            minutes = max(1, round((task["ask_after"] if task else 180) / 60))
            await self.say(f"Got it. I have noted the start time. I will check back in about "
                           f"{minutes} minutes.", ws)
            await self.send_status(ws)
            self._save()
            return

        if cue == "done" and self._mentions_task_subject(low):
            idx = self._matching_task_index(low)
            s.complete_task(idx)
            for ing in s.ingredients:
                base = normalize_name(ing["name"].split(",")[0])
                if base and base in normalize_name(low):
                    s.mark_prepared(ing["name"])
                    break
            if idx == s.current_step_index:
                if s.current_step_index + 1 < len(s.steps):
                    s.next_action = s.steps[s.current_step_index + 1]
                    msg = (f"Great, that is noted. Next up: {s.next_action}. "
                           "Say next when you are ready.")
                else:
                    s.next_action = None
                    msg = "Great, that is noted. Say done when the dish is ready."
            else:
                msg = f"Noted. Right now we are on: {s.current_step}"
            await self.say(msg, ws)
            await self.send_status(ws)
            self._save()
            return

        # ---- next guidance (must come before the bare "next" advance) ----
        if any(p in low for p in ("what should i do next", "what's next", "whats next",
                                  "next action", "aage kya", "ab kya")):
            await self._next_guidance(ws)
            return

        # ---- done with the current step / next ----
        if any(p in low for p in ("next step", "next", "done with this", "move on",
                                  "aage", "continue")) or low.strip(" .!") in ("done", "ok", "okay"):
            await self._advance(ws)
            return

        # ---- previous ----
        if any(p in low for p in ("previous", "go back", "back a step", "pichla")) \
                and not any(p in low for p in ("repeat", "instruction", "say that again")):
            step = s.previous_step()
            await self.say(f"Back to step {s.current_step_index + 1}. {step}", ws)
            await self.send_status(ws)
            self._save()
            return

        # ---- repeat ----
        if any(p in low for p in ("previous instruction", "last instruction")) \
                or ("repeat" in low and any(w in low for w in ("previous", "last", "pichla"))):
            step = s.previous_instruction()
            await self.say(f"Previous instruction: {step}", ws)
            return
        if any(p in low for p in ("repeat", "again", "say that again", "what did you say",
                                  "forgot", "dobara")):
            step = s.repeat()
            await self.say(f"Step {s.current_step_index + 1}. {step}", ws)
            return

        # ---- missing ingredient / substitution ----
        if re.search(r"(don'?t have|do not have|out of|no more|nahi hai|khatam|skip)", low):
            await self._handle_missing(text, low, ws)
            return

        # ---- seasoning mistake ----
        if "too much salt" in low or "extra salt" in low or "namak" in low:
            await self.say("Do not add more salt. Stir in a little sugar or cream, or serve with "
                           "plain rice or bread to balance it. Taste after each addition.", ws)
            return

        # ---- explain slowly ----
        if any(p in low for p in ("slowly", "simpler", "explain", "samajh", "detail")) \
                and s.current_step:
            answer = await self.ask_llm(
                "Explain the current step in very simple words, in two short sentences: "
                + s.current_step, ws)
            await self.say(answer, ws)
            return

        # ---- ingredients ----
        if "ingredient" in low or "what do i need" in low:
            ing = ", ".join(f"{i['quantity']} {i['name']}" for i in s.ingredients)
            await self.say(f"For {s.servings} servings you need: {ing}.", ws)
            return

        # ---- any other question ----
        # General-knowledge questions must not inherit the cooking teacher's
        # "continue the recipe" bias. Answer them conversationally and leave the
        # authoritative step untouched. Cooking questions still get full context.
        cooking_words = (
            "cook", "recipe", "ingredient", "bake", "boil", "fry", "steam",
            "oven", "pan", "heat", "substitute", "spice", "masala", "dish",
            "food", "taste", "season", "measure", "temperature", "timer",
            "step", "kitchen", "serve", "dough", "batter", "oil", "butter",
            "salt", "sugar", "flour", "rice", "chicken", "paneer", "egg",
            "dal", "curry", "roti", "idli", "poha", "pancake", "shawarma",
        )
        if intent.intent == RECIPE_INFORMATION and not intent.dish \
                and not any(word in low for word in cooking_words):
            answer = await self._answer_general(text, ws)
        else:
            answer = await self.ask_llm(text, ws)
        await self.say(answer, ws)

    async def ask_llm(self, question: str, ws: WebSocket, substitution: bool = False) -> str:
        s = self.session
        history = "\n".join(
            f"- {m['role']}: {m['text'][:160]}" for m in s.conversation[-6:]
            if m.get("text")
        )
        conditions = "; ".join(f"{k}: {v}" for k, v in s.conditions.items())
        context = (
            f"Dish: {s.selected_dish} for {s.servings} servings.\n"
            f"Current step {s.current_step_index + 1} of {s.step_count}: {s.current_step}\n"
            f"Next action: {s.next_action}\n"
            f"Ingredients: {', '.join(i['name'] for i in s.ingredients)}\n"
            f"Prepared: {', '.join(sorted(s.prepared)) or 'nothing confirmed yet'}\n"
            f"Missing: {', '.join(s.missing_ingredients) or 'none reported'}\n"
            f"Conditions: {conditions or 'none'}\n"
            f"Recent conversation:\n{history}\n"
        )
        if substitution:
            context += ("The user is missing an ingredient. Judge honestly whether the dish "
                        "still works, suggest a substitute, and describe the taste or texture "
                        "change. Keep it under 40 words.\n")
        try:
            answer = await llm_chat(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": context + "User: " + question}],
                max_tokens=160, temperature=0.35,
            )
            self.llm_ready = True
            return answer or "I did not catch that. Could you say it again?"
        except Exception as e:
            print(f"[LLM] answer failed: {e}")
            self.llm_ready = False
            self.llm_error = str(e)[:200]
            return ("My language model is not responding right now. "
                    f"{self._provider_hint()} Then ask me again.")

    async def _answer_general(self, text: str, ws: WebSocket,
                              dish_context: Optional[str] = None) -> str:
        """
        Answer a general conversation question without triggering cooking context.
        Uses the LLM with a conversational system prompt (not the cooking teacher prompt).
        """
        conv_prompt = """You are AI Chef, a friendly cooking companion.
        Answer the user's question naturally and conversationally.
        Keep responses short (under 60 words), warm, and helpful.
        Do NOT ask about servings, ingredients, or cooking steps unless the user
        explicitly asks to cook something.
        Do NOT mention the current dish, recipe progress, or cooking steps when
        the user asks about a non-cooking subject.
        Do NOT invent recipes. Do NOT start a cooking session.
        Reply in English or Hinglish matching the user's language.
        """
        dish_part = f"\nThe user mentioned this dish: {dish_context}." if dish_context else ""
        try:
            answer = await llm_chat(
                [{"role": "system", "content": conv_prompt + dish_part},
                 {"role": "user", "content": text}],
                max_tokens=160, temperature=0.5,
            )
            self.llm_ready = True
            return answer or "I'm here to help with cooking. What would you like to know?"
        except Exception as e:
            print(f"[LLM] general answer failed: {e}")
            self.llm_ready = False
            return ("I'm having trouble thinking right now. Please try again in a moment.")

    async def _answer_with_rag(self, text: str, dish: str,
                               context: str, ws: WebSocket) -> str:
        """
        Answer a recipe information question using RAG context.
        """
        rag_prompt = f"""You are AI Chef, a knowledgeable cooking teacher.
        Answer the user's question about {dish} using the provided knowledge context.
        Keep responses short (under 80 words), warm, and helpful.
        Do NOT start a cooking session or ask for servings unless the user
        explicitly says they want to cook.
        If the context doesn't contain the answer, say so honestly.
        Reply in English or Hinglish matching the user's language.

        Knowledge context:
        {context}

        User question: {text}
        """
        try:
            answer = await llm_chat(
                [{"role": "system", "content": rag_prompt},
                 {"role": "user", "content": text}],
                max_tokens=200, temperature=0.4,
            )
            self.llm_ready = True
            return answer or f"I don't have detailed information about {dish} right now. Would you like to cook it instead?"
        except Exception as e:
            print(f"[LLM] RAG answer failed: {e}")
            self.llm_ready = False
            return f"I'm having trouble retrieving information about {dish}. Please try again."

    # ==================================================================
    # Proactive progress checks
    # ==================================================================
    async def _progress_watch_loop(self):
        while True:
            try:
                await asyncio.sleep(8)
                await self._maybe_ask_progress()
            except asyncio.CancelledError:
                return
            except Exception as e:
                print(f"[Progress] watcher error: {e}")

    async def _maybe_ask_progress(self):
        s = self.session
        if (not self.connections.active or self.paused or self.tts_busy
                or not s.session_started or not s.current_step):
            return
        now = time.time()
        if now - s.last_activity < 20:
            return
        if now - self._last_proactive < 90:
            return
        for key, task in s.tasks.items():
            if task.get("status") != "started":
                continue
            if task.get("asked", 0) >= 2:
                continue
            started = task.get("started_at") or now
            if now - started < task.get("ask_after", 180):
                continue
            if task.get("last_ask") and now - task["last_ask"] < 60:
                continue
            task["asked"] = task.get("asked", 0) + 1
            task["last_ask"] = now
            self._last_proactive = now
            await self.say(self._task_question(task), speak=True)
            await self.send_status()
            return

    @staticmethod
    def _task_question(task: dict) -> str:
        low = (task.get("label") or "").lower()
        if "boil" in low or "water" in low:
            return "Is the water boiling yet? Tell me when it starts bubbling."
        if "chop" in low or "cut" in low:
            return "Are the onions chopped yet? Tell me when they are ready."
        if "fry" in low or "sauté" in low or "saute" in low or "golden" in low:
            return "Are they golden yet? Keep stirring so they do not burn."
        if "simmer" in low or "cook" in low:
            return "Is it cooked yet? Check that it is soft and not sticking."
        return "How is this step going? Tell me when it is done."

    # ==================================================================
    # Small text helpers
    # ==================================================================
    @staticmethod
    def _looks_like_ingredients(low: str) -> bool:
        if any(p in low for p in ("i have", "i've got", "ive got", "what can i make",
                                  "what can you make", "ingredients are", "left with",
                                  "mere paas", "ghar mein")):
            return True
        return low.count(",") >= 2 and not any(w in low for w in ("make", "cook", "recipe"))

    # Words that may appear before the actual dish name in a natural request.
    # They are stripped one token at a time so that mixed forms such as
    # "can you help me make idli", "i would like to cook pancakes" or
    # "mujhe paneer banao" all reduce to the dish.
    _DISH_LEAD_WORDS = {
        "please", "pls", "kindly", "hey", "hi", "hello", "ok", "okay", "so",
        "can", "could", "would", "will", "you", "u", "help", "me", "us",
        "i", "i'd", "id", "im", "i'm", "want", "wanna", "need", "like",
        "love", "to", "make", "cook", "prepare", "fix", "create", "try",
        "how", "do", "does", "recipe", "recipes", "for", "of", "teach",
        "show", "give", "get", "let's", "lets", "let", "some",
        "mujhe", "maine", "hum", "humein", "hume", "mujhko",
        "banao", "banani", "banaye", "banado", "bana",
        "kaise", "kese", "banate", "a", "an", "the",
        # NOTE: "banana" is deliberately NOT stripped here because it is also
        # a food word ("banana pancakes"); the trailing "... banana hai" form
        # is handled by a separate rule above.
    }
    _DISH_TRAIL_WORDS = {"recipe", "recipes", "please", "pls", "today", "tonight"}

    @classmethod
    def _extract_dish(cls, text: str) -> str:
        t = (text or "").strip()
        # Drop servings phrases first so they cannot leak into the dish name.
        t = re.sub(
            r"\b(?:for\s+)?(\d+|one|two|three|four|five|six|seven|eight|nine|ten|"
            r"do|teen|char|chaar|panch|paanch|chhe|che|saat|aath|nau|das)\s*"
            r"(servings?|people|persons?|portions?|log|logo)\b",
            "", t, flags=re.I,
        )
        t = re.sub(r"\bserves?\s+\d+\b", "", t, flags=re.I)
        t = re.sub(r"\b(servings?|people|persons?|portions?|log|logo)\s*[:=]?\s*\d+",
                   "", t, flags=re.I)
        # Bare "for 2" / "for two" is a serving phrase, not part of the dish.
        t = re.sub(
            r"\bfor\s+(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|"
            r"do|teen|char|chaar|panch|paanch|chhe|che|saat|aath|nau|das)\b",
            "", t, flags=re.I,
        )
        # Hinglish trailing verb forms ("... banana hai", "... banao").
        t = re.sub(r"\b(?:banani|banana|banane)\s+hai\b.*$", "", t, flags=re.I)
        t = re.sub(r"\b(?:banao|bana\s*do|banaye|bana\s*de)\b.*$", "", t, flags=re.I)
        t = re.sub(r"[.?!]+$", "", t).strip()

        tokens = t.split()
        # Strip leading politeness / request / cooking verbs one token at a time.
        while tokens and tokens[0].strip(",;:!?").lower() in cls._DISH_LEAD_WORDS:
            tokens.pop(0)
        # Strip trailing filler such as "idli recipe please".
        while tokens and tokens[-1].strip(",;:!?").lower() in cls._DISH_TRAIL_WORDS:
            tokens.pop()
        t = " ".join(tokens).strip(" ,;:-")
        if len(t) < 3:
            return ""
        return t[:80]

    @staticmethod
    def _is_generic_dish(dish: str) -> bool:
        return normalize_name(dish) in {
            "something", "anything", "food", "dinner", "lunch", "breakfast",
            "a dish", "dish", "recipe", "surprise", "whatever", "kuch", "kuch bhi",
        }

    # Words that can never be a dish name; they usually mean the user is
    # talking about cooking technique or issuing a navigation command.
    _NON_DISH_WORDS = {
        "next", "previous", "back", "repeat", "pause", "resume", "stop",
        "start", "ready", "done", "timer", "yes", "no", "ok", "okay",
        "continue", "skip", "wait", "again", "it", "this", "that", "these",
        "those", "them", "more", "less", "spicy", "hot", "mild", "gravy",
        "sauce", "thick", "thicker", "thin", "thinner", "sweet", "salty",
        "finish", "finishh", "finsh", "finnish", "wrap", "complete", "finishing",
        "are", "is", "was", "were", "am", "be", "been", "being", "chopped",
        "cut", "diced", "sliced", "washed", "boiled", "fried", "added",
        "put", "mixed", "stirred", "cooking", "cook", "make", "made",
        "how", "what", "why", "when", "where", "can", "could", "should",
        "would", "do", "does", "did", "my", "me", "you", "your", "its",
        "and", "or", "with", "on", "in", "at", "for", "to", "of", "a", "an",
    }

    _REQUEST_PHRASE_RE = re.compile(
        r"\b(?:i want to (?:make|cook|eat|have)|i'?d like to (?:make|cook)|"
        r"i would like to (?:make|cook)|make|cook|prepare|recipe for|"
        r"banana hai|banana chahiye|banao|bana do|pakana hai|"
        r"how (?:do|to) i (?:make|cook))\b"
    )

    def _explicit_new_dish(self, text: str, allow_bare: bool = True) -> str:
        """Return a genuinely new dish requested in `text`, else ''.

        This prevents normal cooking talk ("make it spicy", "the onions are
        chopped") from being mistaken for a new recipe, while still letting
        the user switch dishes naturally at any time.
        """
        low = (text or "").lower().strip()
        if not low:
            return ""
        request_like = bool(self._REQUEST_PHRASE_RE.search(low))
        if not request_like and not allow_bare:
            return ""
        dish = self._extract_dish(text)
        if not dish or self._is_generic_dish(dish):
            return ""
        tokens = normalize_name(dish).split()
        if not tokens:
            return ""
        mapped = [self._TYPO_WORDS.get(t, t) for t in tokens]
        if any(t in self._NON_DISH_WORDS for t in tokens + mapped):
            return ""
        if not request_like:
            # A bare dish is only accepted when it is short and noun-like.
            if len(tokens) > 4:
                return ""
        return dish

    @staticmethod
    def _has_variant(dish: str) -> bool:
        low = normalize_name(dish)
        return any(w in low for w in (
            "chicken", "veg", "vegetable", "paneer", "mutton", "egg", "fish",
            "white sauce", "red sauce", "tomato", "cream",
        ))

    def _mentions_task_subject(self, low: str) -> bool:
        return any(w in low for w in (
            "onion", "water", "rice", "potato", "chop", "boil", "fry", "saute",
            "sauté", "mix", "knead", "heat", "pan", "stove", "gas", "masala",
            "dal", "chicken", "vegetable", "tomato", "ginger", "garlic",
            "roast", "bake", "simmer", "pressure",
        ))

    def _matching_task_index(self, low: str) -> int:
        s = self.session
        words = [w for w in re.findall(r"[a-z]{4,}", low) if w not in {
            "done", "that", "this", "with", "have", "chopped", "ready",
            "finished", "golden", "boiled", "cooked", "soft", "tender",
        }]
        for key, task in s.tasks.items():
            label = (task.get("label") or "").lower()
            if any(w in label for w in words):
                try:
                    return int(key)
                except ValueError:
                    continue
        return s.current_step_index


# ======================================================================
# FastAPI app
# ======================================================================
app = FastAPI(title="AI Chef", description="Voice cooking companion")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

pipeline = Pipeline()
ACTIVE_RUNTIMES: set = set()


@app.on_event("startup")
async def _startup():
    await pipeline.startup()


@app.get("/")
async def root():
    """Serve the cooking UI."""
    index = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"message": "AI Chef is running", "ui": "/static/index.html"}


@app.get("/healthz")
async def healthz():
    """Cheap liveness probe: no provider request, safe for frequent monitoring."""
    return {"status": "ok"}


@app.get("/health")
async def health():
    # Detailed status is for humans/dashboards. Do not spend a free-tier
    # provider request on every container/monitor probe; five minutes is enough
    # for a truthful badge without exhausting Groq's daily request quota.
    await pipeline.refresh_llm_status(max_age=300.0)
    stt = pipeline._stt
    tts_mode = pipeline.tts_mode()
    status = provider_status()
    return {
        "status": "ok",
        "stt_ready": bool(stt is not None and getattr(stt, "model", None) is not None),
        "stt_mode": pipeline.stt_mode(),
        "tts_ready": tts_mode in ("server", "browser"),
        "tts_mode": tts_mode,
        "llm_ready": pipeline.llm_ready,
        "llm_model": status.get("model", ""),
        "llm_provider": status.get("provider", ""),
        "llm_configured": bool(status.get("configured")),
        "llm_error": "" if pipeline.llm_ready else (pipeline.llm_error or status.get("error", "")),
        "dish": pipeline.session.selected_dish,
        "step": pipeline.session.current_step_index + 1,
        "steps": pipeline.session.step_count,
        "progress_percent": pipeline.session.progress_percent,
        "timers": pipeline.timers.list(),
        "tts_enabled": TTS_ENABLED,
        "multi_user": MULTI_USER,
        "input_device": getattr(stt, "_input_device", None) if stt else None,
    }


def _mic_unavailable():
    if pipeline.stt_mode() == "server":
        return None
    return {
        "ok": False,
        "unavailable": True,
        "error": ("The server microphone is not available here. Use browser "
                  "Hands-Free or type your command."),
    }


@app.post("/api/voice/stop")
async def api_voice_stop():
    """Stop speech playback immediately (barge-in)."""
    await pipeline.stop_speaking()
    return {"ok": True}


@app.get("/api/mic/devices")
async def api_mic_devices():
    """List microphones so the user can pick the one that actually hears them."""
    blocked = _mic_unavailable()
    if blocked:
        return blocked
    return {
        "ok": True,
        "devices": pipeline.stt.devices(),
        "selected": pipeline.stt.pick_device(),
        "device_name": pipeline.stt.last_device_name,
    }


@app.post("/api/mic/select")
async def api_mic_select(body: dict):
    blocked = _mic_unavailable()
    if blocked:
        return blocked
    try:
        idx = int(body.get("index"))
        selected = await asyncio.to_thread(pipeline.stt.set_device, idx)
        return {"ok": True, "selected": selected}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/api/mic/level")
async def api_mic_level(seconds: float = 2.0, device: Optional[int] = None):
    """Record briefly and report the level, to verify the microphone."""
    blocked = _mic_unavailable()
    if blocked:
        return blocked

    def _run():
        audio = pipeline.stt.record(min(5.0, max(0.5, seconds)), device=device)
        if audio is None:
            return {"ok": False, "peak": 0.0, "rms": 0.0,
                    "device": pipeline.stt._input_device,
                    "device_name": pipeline.stt.last_device_name,
                    "level": "error", "error": "No audio captured"}
        peak = float(np.max(np.abs(audio)))
        rms = float(np.sqrt(np.mean(audio ** 2)))
        level = "good" if peak >= 0.05 else "low" if peak >= 0.005 else "silent"
        return {"ok": peak >= 0.003, "peak": round(peak, 4), "rms": round(rms, 4),
                "device": pipeline.stt._input_device,
                "device_name": pipeline.stt.last_device_name, "level": level}

    return await asyncio.to_thread(_run)


@app.post("/api/listen")
async def api_listen(seconds: float = 6.0):
    """Record from the host microphone and return the transcript."""
    blocked = _mic_unavailable()
    if blocked:
        return blocked

    def _run():
        audio = pipeline.stt.record(min(15.0, max(1.0, seconds)))
        if audio is None:
            return {"ok": False, "text": "", "peak": 0.0,
                    "device": pipeline.stt._input_device,
                    "device_name": pipeline.stt.last_device_name,
                    "error": "No audio captured. The microphone may be muted or blocked by Windows."}
        peak = float(np.max(np.abs(audio)))
        text = pipeline.stt.transcribe(audio)
        print(f"[STT] listen: device={pipeline.stt._input_device} "
              f"peak={peak:.4f} text={text!r}", flush=True)
        return {"ok": bool(text), "text": text or "", "peak": round(peak, 4),
                "device": pipeline.stt._input_device,
                "device_name": pipeline.stt.last_device_name}

    return await asyncio.to_thread(_run)


@app.get("/api/mic-test")
async def api_mic_test():
    blocked = _mic_unavailable()
    if blocked:
        return blocked

    def _run():
        audio = pipeline.stt.record(3.0)
        if audio is None:
            return {"ok": False, "error": "No audio captured (muted, privacy blocked, or broken device)"}
        peak = float(np.max(np.abs(audio)))
        return {
            "ok": peak >= 0.003,
            "peak": round(peak, 4),
            "rms": round(float(np.sqrt(np.mean(audio ** 2))), 4),
            "device": pipeline.stt._input_device,
            "device_name": pipeline.stt.last_device_name,
            "transcript": pipeline.stt.transcribe(audio) if peak >= 0.003 else None,
        }

    return await asyncio.to_thread(_run)


@app.post("/api/recipe/dish-first")
async def api_dish_first(body: dict):
    dish = (body.get("dish") or "").strip()
    try:
        servings = max(1, min(20, int(body.get("servings") or 2)))
    except Exception:
        servings = 2
    if not dish:
        return {"ok": False, "error": "dish required"}
    try:
        recipe = await pipeline.generate_recipe(dish, servings)
        return {"ok": True, "recipe": recipe}
    except RecipeUnavailable as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/recipe/ingredients-first")
async def api_ingredients_first(body: dict):
    ingredients = (body.get("ingredients") or "").strip()
    if not ingredients:
        return {"ok": False, "error": "ingredients required"}
    try:
        suggestions = await pipeline.suggestions_for(ingredients)
        return {"ok": True, "suggestions": suggestions}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    runtime = pipeline
    created = False
    if MULTI_USER:
        if len(ACTIVE_RUNTIMES) >= MAX_CONNECTIONS:
            await ws.close(code=1013, reason="Server is full; please retry shortly.")
            return
        runtime = Pipeline(
            session_obj=CookingSession(),
            timer_manager=TimerManager(),
            persist=False,
        )
        await runtime.startup_connection()
        ACTIVE_RUNTIMES.add(runtime)
        created = True
    try:
        await runtime.on_connect(ws)
        while True:
            raw = await ws.receive_text()
            try:
                data = json.loads(raw)
            except Exception:
                data = {"type": "command", "text": raw}
            await runtime.on_message(ws, data)
    except WebSocketDisconnect:
        pass
    except RuntimeError as e:
        # Starlette raises this if the client vanishes between accept() and
        # receive_text(); it is a normal disconnect, not a server error.
        if "not connected" not in str(e).lower():
            print(f"[WS] runtime error: {e}", flush=True)
    except Exception as e:
        import traceback
        print(f"[WS] error: {e}\n{traceback.format_exc()}", flush=True)
    finally:
        runtime.connections.disconnect(ws)
        if created:
            ACTIVE_RUNTIMES.discard(runtime)
            await runtime.shutdown_connection()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
        proxy_headers=True,
    )
