# src/paths.py
from pathlib import Path
import sys
import os

def app_root() -> Path:
    # Where code lives. When frozen, this is the PyInstaller temp dir (_MEIPASS).
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[1]

def runtime_root() -> Path:
    # Where the user runs the app from. When frozen, this is the folder
    # that contains FoodChat.exe (e.g., dist\). In dev, it's the project root.
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]

def assets_dir() -> Path:
    # 1) Allow override for both dev & prod
    override = os.environ.get("FOODCHAT_ASSETS_DIR")
    if override:
        return Path(override)

    root = runtime_root()
    # 2) Prefer ./assets if it exists (nice for dev)
    if (root / "assets").exists():
        return root / "assets"
    # 3) Otherwise use ./data (your repo/exe layout)
    return root / "data"


# Convenience helpers (optional but handy)
def data_file(name: str) -> Path:
    return assets_dir() / name

def model_dir(name: str) -> Path:
    return assets_dir() / name
