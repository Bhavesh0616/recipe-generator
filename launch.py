# main / launch
from __future__ import annotations
import os, json, time
from pathlib import Path
from typing import Dict, Any, List
from flask import Flask, request, jsonify, send_from_directory

from recipe import suggest_recipes, get_recipe_detail
from trans import normalize_query, safe_lang_code
from nutrition import nutrition_from_ingredients
from voice import transcribe_wav
from healthcheck import health_check_text
from trans import safe_lang_code  # (already imported elsewhere)


# -------------------- config --------------------
APP_ROOT = Path(os.getenv("APP_ROOT", Path(__file__).parent)).resolve()
DATA_DIR = APP_ROOT / "data"
CHATS_DIR = APP_ROOT / "chats"
CHATS_DIR.mkdir(parents=True, exist_ok=True)

INDEX_HTML = os.getenv("INDEX_HTML", "frontend_food_chat.html")

# create Flask app ONCE
app = Flask(__name__, static_folder=str(APP_ROOT))

@app.get("/")
def index():
    # serve the attached HTML from your app folder
    return send_from_directory(app.static_folder, INDEX_HTML)

@app.get("/favicon.ico")
def favicon():
    return send_from_directory(app.static_folder, "favicon.ico")

def _active_path_file() -> Path:
    return CHATS_DIR / "_active.txt"

def _now_iso():
    return time.strftime("%Y%m%dT%H%M%S")

# ---------- tiny chat store ----------
def _chat_list() -> List[Dict[str, str]]:
    chats = []
    for p in sorted(CHATS_DIR.glob("*.json")):
        if p.name.startswith("_"): 
            continue
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
            chats.append({"title": j.get("title") or p.stem, "path": str(p)})
        except Exception:
            chats.append({"title": p.stem, "path": str(p)})
    return chats

def _active_chat_path() -> Path | None:
    f = _active_path_file()
    if f.exists():
        p = Path(f.read_text(encoding="utf-8").strip() or "")
        return p if p.exists() else None
    # else: pick latest or create one
    lst = list(CHATS_DIR.glob("*.json"))
    return lst[-1] if lst else None

def _ensure_chat(title: str = "Ingredients") -> Path:
    p = _active_chat_path()
    if p: 
        return p
    path = CHATS_DIR / f"{_now_iso()}_{title.replace(' ','_')}.json"
    path.write_text(json.dumps({"title": title, "messages": []}, ensure_ascii=False, indent=2), encoding="utf-8")
    _active_path_file().write_text(str(path), encoding="utf-8")
    return path

def _append_msg(path: Path, role: str, content: str, meta: Dict[str, Any] | None = None):
    try:
        j = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        j = {"title": path.stem, "messages": []}
    j.setdefault("messages", []).append({
        "role": role,
        "content": content,
        "ts": _now_iso(),
        "meta": meta or {}
    })
    path.write_text(json.dumps(j, ensure_ascii=False, indent=2), encoding="utf-8")

# ----------------- static icon for mic -----------------
@app.get("/icons8-microphone-48.png")
def mic_png():
    return send_from_directory(app.static_folder, "icons8-microphone-48.png")

# ----------------- chats API --------------------------
@app.get("/api/chats")
def list_chats():
    return jsonify(_chat_list())

@app.post("/api/chats")
def create_chat():
    title = (request.get_json(force=True) or {}).get("title") or "Ingredients"
    path = CHATS_DIR / f"{_now_iso()}_{title.replace(' ','_')}.json"
    path.write_text(json.dumps({"title": title, "messages": []}, ensure_ascii=False, indent=2), encoding="utf-8")
    return jsonify({"title": title, "path": str(path)})

@app.get("/api/active_chat")
def get_active_chat():
    p = _active_chat_path()
    return jsonify({"path": str(p) if p else ""})

@app.post("/api/active_chat")
def set_active_chat():
    path = (request.get_json(force=True) or {}).get("path")
    if path and Path(path).exists():
        _active_path_file().write_text(path, encoding="utf-8")
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "invalid path"})

@app.get("/api/chat_history")
def chat_history():
    p = _active_chat_path()
    if not p: 
        return jsonify([])
    try:
        j = json.loads(p.read_text(encoding="utf-8"))
        msgs = j.get("messages", [])
        # front-end expects flat list
        return jsonify(msgs[::-1])  # newest first is handled client-side too
    except Exception:
        return jsonify([])

# ----------------- health filters save -----------------
@app.post("/api/health_filters")
def save_health_filters():
    payload = request.get_json(force=True) or {}
    # flatten to a simple list like ["diabetes", "low_sodium"]
    filters: List[str] = []
    if payload.get("diabetes"): filters.append("diabetes")
    if payload.get("hypertension"): filters.append("hypertension")
    if payload.get("low_sodium"): filters.append("low_sodium")
    if payload.get("vegetarian"): filters.append("vegetarian")
    if payload.get("vegan"): filters.append("vegan")
    if payload.get("gluten_free"): filters.append("gluten_free")
    if payload.get("dairy_free"): filters.append("dairy_free")
    if payload.get("keto"): filters.append("keto")
    if payload.get("paleo"): filters.append("paleo")
    if payload.get("heart_health"): filters.append("heart_health")
    if payload.get("renal"): filters.append("renal")
    if payload.get("weight_loss"): filters.append("weight_loss")
    if payload.get("high_protein"): filters.append("high_protein")
    if payload.get("low_fat"): filters.append("low_fat")
    if payload.get("low_carb"): filters.append("low_carb")
    if payload.get("pregnancy"): filters.append("pregnancy")
    if payload.get("children"): filters.append("children")
    if payload.get("senior"): filters.append("senior")
    for a in payload.get("allergies") or []:
        filters.append(f"allergy_{a}".lower())

    p = _ensure_chat()
    try:
        j = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        j = {"title": p.stem, "messages": []}
    j["health_filters"] = filters
    p.write_text(json.dumps(j, ensure_ascii=False, indent=2), encoding="utf-8")
    return jsonify({"ok": True, "filters": filters})

# ----------------- core: find recipes ------------------
# /api/find_recipes
@app.post("/api/find_recipes")
def api_find_recipes():
    data = request.get_json(force=True) or {}
    raw = (data.get("ingredients") or "").strip()
    display_lang = safe_lang_code(data.get("display_lang") or "auto")

    print(f"[FIND] raw='{raw}' display_lang='{display_lang}'")  # <--- log raw

    query_norm = normalize_query(raw, display_lang)
    print(f"[FIND] transliterated/normalized='{query_norm}'")    # <--- log transliteration

    p = _ensure_chat()
    _append_msg(p, "user", raw)

    list_text = suggest_recipes(query_norm, display_lang=display_lang)
    print(f"[FIND] suggestions len={len(list_text)}\n{list_text}\n")  # <--- log suggestions returned

    _append_msg(p, "assistant", list_text)
    return jsonify({"ok": True})


# /api/get_recipe_detail
@app.post("/api/get_recipe_detail")
def api_get_recipe_detail():
    data = request.get_json(force=True) or {}
    title = (data.get("title") or "").strip()
    display_lang = safe_lang_code(data.get("display_lang") or "auto")

    print(f"[DETAIL] ui_title='{title}' display_lang='{display_lang}'")  # <--- log clicked title

    p = _ensure_chat()
    _append_msg(p, "user", f"Selected recipe: {title}")

    try:
        detail = get_recipe_detail(title, display_lang=display_lang)
    except Exception as e:
        print("[DETAIL] get_recipe_detail ERROR:", e)
        detail = {"ingredients": [], "steps": []}

    ings = detail.get("ingredients", []) or []
    steps = detail.get("steps", []) or []

    print(f"[DETAIL] returned ings={len(ings)} steps={len(steps)}")       # <--- log counts
    if ings[:2] or steps[:2]:
        print(f"[DETAIL] sample ing='{ings[:2]}' sample step='{steps[:2]}'")  # <--- sample

    from trans import section_labels
    ING_H, STEP_H = section_labels(display_lang)
    txt = f"{ING_H}:\n- " + "\n- ".join(ings) + "\n\n" + f"{STEP_H}:\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))

    meta = {"payload": {"recipe_detail": {"title": title, "ingredients": ings, "steps": steps}}}
    try:
        _append_msg(p, "assistant", txt, meta=meta)
    except Exception as ee:
        print("[DETAIL] chat save error:", ee)

    return jsonify({"found": True, "title": title, "ingredients": ings, "steps": steps, "raw": txt})


# ---------------- nutrition from model -----------------
@app.post("/api/check_nutrition")
def api_check_nutrition():
    data = request.get_json(force=True) or {}
    ingredients = data.get("ingredients") or []
    if isinstance(ingredients, str):
        ingredients = [x.strip() for x in ingredients.split(",") if x.strip()]
    text = nutrition_from_ingredients(ingredients)
    return jsonify({"text": text})

# ---------------- health check suggestions -----------------
# ---------------- health check suggestions -----------------
@app.post("/api/health_check")
def api_health_check():
    data = request.get_json(force=True) or {}
    display_lang = safe_lang_code(data.get("display_lang") or "auto")

    p = _ensure_chat()
    try:
        j = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return jsonify({"ok": False, "error": "no active chat"}), 400

    filters = j.get("health_filters", [])

    # latest recipe detail payload from chat history
    ingredients, steps, title = [], [], ""
    for m in reversed(j.get("messages", [])):
        meta = (m.get("meta") or {})
        payload = (meta.get("payload") or {})
        rd = payload.get("recipe_detail")
        if rd:
            title = rd.get("title", "")
            ingredients = rd.get("ingredients", []) or []
            steps = rd.get("steps", []) or []
            break

    if not ingredients and not steps:
        return jsonify({"ok": False, "error": "no recipe detail to analyze"}), 400

    text = health_check_text(ingredients, steps, filters, display_lang)

    # Save as an assistant message so it appears in the chat (like Nutrition)
    block = f"🩺 Health Check for: {title or 'current recipe'}\n\n{text}"
    meta = {"payload": {"health_check": {"filters": filters, "text": text}}}
    try:
        _append_msg(p, "assistant", block, meta=meta)
    except Exception as ee:
        print("[health_check] warn: failed to save chat message:", ee)

    return jsonify({"ok": True, "text": text})


# -------------------- voice (optional) -----------------
@app.post("/api/voice_input")
def api_voice_input():
    f = request.files.get("audio")
    if not f:
        return jsonify({"error": "no audio"}), 400
    tmp = APP_ROOT / "_tmp_audio.wav"
    f.save(str(tmp))
    text, err = transcribe_wav(tmp)
    try:
        tmp.unlink(missing_ok=True)
    except Exception:
        pass
    if err:
        return jsonify({"error": err})
    return jsonify({"text": text})

# -------------------- run ------------------------------
if __name__ == "__main__":
    app.run(host=os.getenv("HOST", "127.0.0.1"),
            port=int(os.getenv("PORT", "0") or "0"),
            debug=False)