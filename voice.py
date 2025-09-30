# voice.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Tuple

WHISPER_DIR = Path(os.getenv("WHISPER_MODEL_DIR", r"F:\foodchat_clean\assets\models--openai--whisper-small"))

def _have_whisper_files() -> bool:
    # naive check for model files in the directory
    return WHISPER_DIR.exists() and any(WHISPER_DIR.glob("**/*.pt"))

def transcribe_wav(path: Path) -> Tuple[str, str | None]:
    """
    Returns (text, error). If whisper isn't available, returns ("", "error message").
    """
    try:
        import whisper  # pip install openai-whisper
    except Exception:
        return "", "Whisper not installed. pip install openai-whisper"

    try:
        if _have_whisper_files():
            # Force whisper to use local dir cache
            os.environ["XDG_CACHE_HOME"] = str(WHISPER_DIR.parent)
            os.environ["TRANSFORMERS_CACHE"] = str(WHISPER_DIR.parent)
            model = whisper.load_model("small")  # should pick from local cache
        else:
            model = whisper.load_model("small")  # will try to download if internet; likely offline
        r = model.transcribe(str(path), fp16=False, language=None)
        return (r.get("text") or "").strip(), None
    except Exception as e:
        return "", f"whisper failed: {e}"
