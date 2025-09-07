# voice_test.py
import sounddevice as sd
import numpy as np
import whisper

_MODEL = None

def _get_model(name: str = "base"):
    global _MODEL
    if _MODEL is None:
        _MODEL = whisper.load_model(name)  # cached after first load
    return _MODEL

def record_audio_buffer(duration=5, fs=16000):
    """Record mono float32 audio into memory (no file)."""
    print(f"[voice] Recording {duration}s @ {fs}Hz …")
    rec = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()
    buf = rec[:, 0].astype(np.float32)
    # ensure [-1, 1]
    np.clip(buf, -1.0, 1.0, out=buf)
    return buf, fs

def transcribe_buffer(buf: np.ndarray, fs=16000, model_name="base", language=None):
    """Transcribe an in-memory float32 buffer with Whisper (CPU-friendly)."""
    mdl = _get_model(model_name)
    # Whisper accepts a float32 numpy array (16kHz mono). Keep fp16 off for CPU/Windows.
    result = mdl.transcribe(buf, fp16=False, language=language)
    text = (result.get("text") or "").strip()
    if not text:
        raise RuntimeError("Empty transcript (try speaking louder/closer).")
    print("[voice] Transcribed:", text)
    return text

def record_and_transcribe_mem(duration=5, fs=16000, model_name="base", language=None):
    """Convenience: record into memory and return text."""
    buf, _ = record_audio_buffer(duration=duration, fs=fs)
    return transcribe_buffer(buf, fs=fs, model_name=model_name, language=language)

# Backward-compatible file-based helpers (only used if import fallback happens)
import scipy.io.wavfile as wavfile
def record_audio(duration=5, fs=16000, filename="temp.wav"):
    print(f"[voice] Recording {duration}s @ {fs}Hz …")
    rec = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()
    pcm16 = (rec[:,0] * 32767.0).astype(np.int16)
    wavfile.write(filename, fs, pcm16)
    print(f"[voice] Saved WAV -> {filename}")
    return filename

def transcribe_audio(filepath, model_name="base", language=None):
    mdl = _get_model(model_name)
    result = mdl.transcribe(filepath, fp16=False, language=language)
    txt = (result.get("text") or "").strip()
    print("[voice] Transcribed:", txt)
    if not txt:
        raise RuntimeError("Empty transcript")
    return txt

def record_and_transcribe(duration=5, fs=16000, model_name="base", language=None):
    return transcribe_audio(record_audio(duration, fs), model_name, language)

if __name__ == "__main__":
    print("🎤 Test (in-memory)…")
    print(record_and_transcribe_mem())
