# tools/ensure_assets.py
from __future__ import annotations
import os, sys, shutil
from pathlib import Path
import hashlib
import urllib.request

APP_ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))  # works in PyInstaller
ASSETS = APP_ROOT / "assets"
MODELS = APP_ROOT / "models"

# TODO: replace these with your own release URLs (GitHub Release assets, S3, etc.)
URLS = {
    # facebook/m2m100_418M (flat layout as you use now)
    "m2m_config":            ("assets/translation_models/config.json", "https://example.com/m2m/config.json", ""),
    "m2m_model":             ("assets/translation_models/model.safetensors", "https://example.com/m2m/model.safetensors", ""),
    "m2m_tokenizer":         ("assets/translation_models/tokenizer_config.json", "https://example.com/m2m/tokenizer_config.json", ""),
    "m2m_spm":               ("assets/translation_models/sentencepiece.bpe.model", "https://example.com/m2m/sentencepiece.bpe.model", ""),
    "m2m_vocab":             ("assets/translation_models/vocab.json", "https://example.com/m2m/vocab.json", ""),
    "m2m_added":             ("assets/translation_models/added_tokens.json", "https://example.com/m2m/added_tokens.json", ""),
    "m2m_special":           ("assets/translation_models/special_tokens_map.json", "https://example.com/m2m/special_tokens_map.json", ""),

    # whisper-small local cache (optional—otherwise Whisper downloads once with internet)
    # Put your snapshot files somewhere and list them here if you want fully offline voice.

    # mistral gguf
    "mistral_gguf":          ("models/mistral-7b-instruct-v0.2/mistral-7b-instruct-v0.2.Q4_K_M.gguf", "https://example.com/gguf/mistral.Q4_K_M.gguf", ""),
}

def _download(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    print("[assets] downloading:", dst.name)
    with urllib.request.urlopen(url) as r, open(dst, "wb") as f:
        shutil.copyfileobj(r, f)

def ensure_all():
    ok = True
    for key, (rel, url, sha) in URLS.items():
        dst = APP_ROOT / rel
        if dst.exists() and dst.stat().st_size > 1024:
            continue
        if not url:
            print("[assets] missing URL for", key)
            ok = False
            continue
        try:
            _download(url, dst)
            if sha:
                h = hashlib.sha256(dst.read_bytes()).hexdigest()
                if h.lower() != sha.lower():
                    print("[assets] hash mismatch for", dst)
                    ok = False
        except Exception as e:
            print("[assets] download failed:", dst, e)
            ok = False
    return ok

if __name__ == "__main__":
    sys.exit(0 if ensure_all() else 1)
