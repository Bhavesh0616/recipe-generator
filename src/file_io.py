import os, json, time, re
from pathlib import Path
from typing import Any, Dict, Optional
from src.paths import runtime_root  # <<< single source of truth

BASE = runtime_root()               # project root in dev, exe folder in prod
DATA_DIR = BASE / "data"
CHATS_DIR = DATA_DIR / "chats"
USER_DIR = DATA_DIR / "user"
HEALTH_FILTERS_PATH = DATA_DIR / "health_filters.json"
RECIPES_DB_PATH = DATA_DIR / "recipes_db.json"
LAST_OPENED_PATH = USER_DIR / "last_opened_chat.json"

def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHATS_DIR.mkdir(parents=True, exist_ok=True)
    USER_DIR.mkdir(parents=True, exist_ok=True)

def _atomic_write(path: Path, obj: Any):
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)

def read_json(path: Path, default: Any):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default

def write_json(path: Path, obj: Any):
    _atomic_write(path, obj)

def slugify(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", (name or "").strip().lower()).strip("-") or time.strftime("%H%M%S")

def create_new_chat(title: Optional[str] = None) -> Path:
    ensure_dirs()
    ts = time.strftime("%Y%m%d-%H%M%S")
    slug = slugify(title or "chat")
    path = CHATS_DIR / f"chat-{ts}-{slug}.json"
    doc = {"schema_version": 1, "title": title or f"Chat {ts}", "created_at": ts, "messages": [], "currency": "EUR"}
    write_json(path, doc)
    write_json(LAST_OPENED_PATH, {"active_chat": str(path.relative_to(BASE))})
    return path

def get_active_chat_path() -> Optional[Path]:
    ensure_dirs()
    meta = read_json(LAST_OPENED_PATH, default={})
    rel = meta.get("active_chat")
    if not rel:
        return None
    p = (BASE / rel) if not Path(rel).is_absolute() else Path(rel)
    return p if p.exists() else None

def set_active_chat(path: Path):
    path = Path(path)
    write_json(LAST_OPENED_PATH, {"active_chat": str(path if path.is_absolute() else path.resolve().relative_to(BASE))})

def append_message(chat_path: Path, role: str, content: str, extra: Optional[Dict]=None):
    doc = read_json(chat_path, default=None)
    if not doc: return
    msg = {"role": role, "content": content, "ts": time.strftime("%Y%m%d-%H%M%S")}
    if extra: msg["meta"] = extra
    doc.setdefault("messages", []).append(msg)
    write_json(chat_path, doc)

def load_health_filters() -> Dict:
    return read_json(HEALTH_FILTERS_PATH, default={
        "diabetes": False,
        "hypertension": False,
        "allergies": [],
        "low_sodium": False,
        "vegetarian": False,
        "vegan": False
    })

def save_health_filters(filters: Dict):
    write_json(HEALTH_FILTERS_PATH, filters)

def load_recipes_db() -> Dict:
    return read_json(RECIPES_DB_PATH, default={"recipes": []})
