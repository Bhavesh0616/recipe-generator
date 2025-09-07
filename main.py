import os
os.environ.setdefault("FLASK_SKIP_DOTENV", "1")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

from flask import Flask, request, jsonify, send_from_directory
from pathlib import Path
import json, time

# use those variables when you load files / models
from src.recipe_search import normalize_ingredients, search_recipes, init_search_engine, get_recipe_by_title
from src.file_io import ensure_dirs, create_new_chat, get_active_chat_path, set_active_chat, append_message, CHATS_DIR
from src.paths import data_file


from src.recipe_search import normalize_ingredients, search_recipes, init_search_engine

# your local modules (unchanged)
from src.file_io import (
    ensure_dirs, create_new_chat, get_active_chat_path, set_active_chat,
    append_message, CHATS_DIR
)

# NEW: import just the functions from split files
from src.generator import generate_recipes, extract_keywords
from src.translator import from_en, healthy as trn_healthy, process_user_input

# ---------- Flask setup ----------
app = Flask(__name__, static_folder=None)
ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend_food_chat.html"


from src.paths import data_file
FULL_DATASET = data_file("full_dataset.csv")
CLEANED_CSV = data_file("cleaned_ingredients.csv")


from src.recipe_search import init_search_engine

def _append_msg(chat_path: Path, role: str, content: str, meta: dict | None = None):
    chat_path = Path(chat_path)
    try:
        j = json.loads(chat_path.read_text(encoding="utf-8"))
    except Exception:
        j = {}
    if not isinstance(j, dict):
        j = {}
    j.setdefault("schema_version", 1)
    j.setdefault("title", chat_path.stem)
    j.setdefault("created_at", j.get("created_at", time.strftime("%Y%m%d-%H%M%S")))
    j.setdefault("messages", [])
    msg = {"role": role, "content": content, "ts": time.strftime("%Y%m%d-%H%M%S")}
    if meta:
        msg["meta"] = meta
    j["messages"].append(msg)
    if role == "assistant":
        j["last_payload"] = meta.get("payload") if meta and "payload" in meta else {"suggestions": content}
    chat_path.write_text(json.dumps(j, indent=2, ensure_ascii=False), encoding="utf-8")

def parse_user_input(text: str) -> str:
    t = (text or "").strip().lower()
    if t.startswith(("what","how","which","suggest","give","show","list")) or " for " in t:
        return "question"
    if "," in t or len(t.split()) <= 5:
        return "ingredients"
    return "other"

def _fallback_titles(ingredients_text: str):
    txt = (ingredients_text or "").lower()
    toks = [s.strip() for s in txt.replace("\n", ",").split(",") if s.strip()]
    if not toks:
        return ["Mixed Veg Bowl", "Quick Stir-Fry", "Healthy Salad"]
    base = [toks[0].capitalize()] + [toks[1].capitalize()] if len(toks) > 1 else [toks[0].capitalize()]
    if len(base) == 1:
        return [f"{base[0]} Delight", f"Quick {base[0]} Bowl", f"Homestyle {base[0]}"]
    return [f"{base[0]} & {base[1]}", f"Quick {base[0]} {base[1]} Mix", f"{base[0]} {base[1]} Skillet"]

import re

def _requested_k(text: str, default_k: int = 5, hard_cap: int = 10) -> int:
    """Pull an integer like '6' from 'suggest 6 recipes'. Falls back to default."""
    m = re.search(r'\b(\d{1,2})\b', text or '')
    if not m:
        return default_k
    k = int(m.group(1))
    return max(1, min(k, hard_cap))

def _bullets_n(titles, n: int):
    seen = set()
    unique_titles = []
    for t in titles:
        if t not in seen:
            unique_titles.append(t)
            seen.add(t)
        if len(unique_titles) == n:
            break
    return "• " + "\n• ".join(unique_titles) if unique_titles else ""

# inside main.py (near the route) add:
def _translate_titles(titles, display_lang):
    if display_lang not in ("en", "auto") and titles:
        try:
            from src.translator import from_en
            return [from_en(t, display_lang) for t in titles]
        except Exception as e:
            print("[list translate] error:", e)
    return titles

def _pairs_and_bullets(df, display_lang, k):
    # unique English titles from the CSV result
    titles_en = (df['title'].dropna().astype(str).drop_duplicates().tolist())[:k]
    titles_ui = _translate_titles(titles_en, display_lang)
    pairs = [{"en": e, "ui": u} for e, u in zip(titles_en, titles_ui)]
    bullets = "• " + "\n• ".join(p["ui"] for p in pairs)
    return pairs, bullets


# ---------- Routes ----------
@app.route("/", methods=["GET"])
def index():
    return send_from_directory(FRONTEND.parent, FRONTEND.name)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "app": "ok",
        "generator_loaded": True,
        "translator_loaded": trn_healthy()
    })

@app.route("/api/chats", methods=["GET"])
def api_list_chats():
    ensure_dirs()
    items = []
    if CHATS_DIR.exists():
        for fp in sorted(CHATS_DIR.glob("*.json")):
            title = fp.stem
            try:
                j = json.loads(fp.read_text(encoding="utf-8"))
                if isinstance(j, dict) and j.get("title"):
                    title = str(j["title"])
            except Exception:
                pass
            items.append({"title": title, "path": str(fp)})
    return jsonify(items)

@app.route("/api/chats", methods=["POST"])
def api_create_chat():
    ensure_dirs()
    data = request.get_json(force=True) or {}
    title = (data.get("title") or "New Chat").strip() or "New Chat"
    ts = time.strftime("%Y%m%d-%H%M%S")
    slug = "-".join(title.lower().split())[:32] or "chat"
    fp = CHATS_DIR / f"chat-{ts}-{slug}.json"
    fp.write_text(json.dumps({"title": title, "messages": []}, indent=2), encoding="utf-8")
    set_active_chat(fp)
    return jsonify({"title": title, "path": str(fp)})

@app.route("/api/chat_last", methods=["GET"])
def api_chat_last():
    p = get_active_chat_path()
    if not p or not Path(p).exists():
        return jsonify({"suggestions": "", "top": []})
    try:
        j = json.loads(Path(p).read_text(encoding="utf-8"))
        payload = j.get("last_payload") or {}
        return jsonify({
            "suggestions": payload.get("suggestions", ""),
            "top": payload.get("top", []),
            "currency": payload.get("currency", "USD"),
            "filters": payload.get("filters", {}),
        })
    except Exception:
        return jsonify({"suggestions": "", "top": []})

@app.route("/api/active_chat", methods=["GET"])
def api_get_active_chat():
    p = get_active_chat_path()
    return jsonify({"path": str(p) if p else ""})

@app.route("/api/active_chat", methods=["POST"])
def api_set_active_chat():
    data = request.get_json(force=True) or {}
    path = data.get("path")
    if not path:
        return jsonify({"ok": False, "error": "path required"}), 400
    p = Path(path)
    if not p.exists():
        return jsonify({"ok": False, "error": "file not found"}), 404
    set_active_chat(p)
    return jsonify({"ok": True})

@app.route("/api/chat_history", methods=["GET"])
def api_chat_history():
    p = get_active_chat_path()
    if not p or not Path(p).exists():
        return jsonify([])  # no active chat
    try:
        j = json.loads(Path(p).read_text(encoding="utf-8"))
        return jsonify(j.get("messages", []))
    except Exception:
        return jsonify([])  # fallback

from src.generator import generate_recipes, extract_keywords, extract_recipe_title
from src.recipe_search import normalize_ingredients, search_recipes, get_recipe_by_title

@app.route("/api/find_recipes", methods=["POST"])
def api_find_recipes():
    data = request.get_json(force=True) or {}
    raw_input = (data.get("ingredients") or "").strip()
    display_lang = data.get("display_lang") or "auto"

    if display_lang == "auto":
        from src.translator import detect_lang_by_script
        display_lang = detect_lang_by_script(raw_input)

    user_input, translated_ui = process_user_input(raw_input, display_lang)
    ask_k = _requested_k(user_input, default_k=5)

    input_type = parse_user_input(user_input)

    # ✅ Force title/keyword mode for these languages (so “北京烤鸭”, “สะเต๊ะ” aren’t treated as ingredient lists)
    if display_lang in ("zh", "th", "vi", "tr"):
        input_type = "question"

    # 🔹 NEW: Build an ENGLISH query for the CSV search (we’ll still display results in the UI language)
    query_en = None
    if display_lang not in ("en", "auto"):
        try:
            from src.translator import to_en
            query_en = to_en(raw_input, display_lang)
        except Exception:
            query_en = None

    bullets = ""
    df = None  # Always initialize df

    try:
        # (Optional pre-pass you already had) — switch to use the EN base if we have it
        if display_lang not in ("en", "auto") and len(user_input.split()) > 1:
            base = (query_en or user_input)  # 🔹 NEW
            df_exact = search_recipes([base], top_k=ask_k)
            if not df_exact.empty:
                df = df_exact
            else:
                raw_ings = [w.strip().lower() for w in base.split() if w.strip()]  # 🔹 NEW (base)
                norm_ings = normalize_ingredients(raw_ings)
                df = search_recipes(norm_ings, top_k=ask_k)

            # ✅ Punjabi-specific fallback if everything failed (unchanged)
            if (df is None or df.empty) and display_lang == "pa":
                try:
                    from src.translator import transliterate_text
                    from difflib import get_close_matches
                    from src.recipe_search import DF

                    title_en = transliterate_text(user_input, "pa")
                    print("[Punjabi match] Transliteration:", title_en)

                    if DF is None:
                        from src.recipe_search import init_search_engine
                        init_search_engine()

                    title_list = DF['title'].dropna().astype(str).str.lower().tolist()
                    closest_titles = get_close_matches(title_en.lower(), title_list, n=3, cutoff=0.5)
                    print("[Punjabi match] Found:", closest_titles)

                    if closest_titles:
                        bullets = "• " + "\n• ".join(closest_titles[:ask_k])
                        payload = {"suggestions": bullets, "top": [], "currency": "USD", "filters": {}}
                        return jsonify(payload)

                except Exception as e:
                    print("[Punjabi fallback error]", e)

        def _titles_from_df(df):
            return df['title'].dropna().astype(str).drop_duplicates().tolist()

        if input_type == "ingredients":
            # 🔹 Use English query if available (better TF-IDF match), else the romanized user_input
            base = (query_en or user_input)  # 🔹 NEW
            raw_ings = [w.strip().lower() for w in base.replace(",", " ").split() if w.strip()]
            norm_ings = normalize_ingredients(raw_ings)
            print(f"🥘 Searching by ingredients: {norm_ings}")

            df = search_recipes(norm_ings, top_k=ask_k)
            titles = _titles_from_df(df)
            titles = _translate_titles(titles, display_lang)
            bullets = _bullets_n(titles, ask_k)

        else:
            from src.generator import extract_keywords_simple
            base = (query_en or user_input)  # 🔹 NEW
            kws = extract_keywords_simple(base)
            norm_kws = normalize_ingredients(kws)
            print(f"🔎 Derived keywords: {norm_kws}")

            df = search_recipes(norm_kws or base.split(), top_k=ask_k)
            if not df.empty:
                titles = _titles_from_df(df)
                titles = _translate_titles(titles, display_lang)
                bullets = _bullets_n(titles, ask_k)
            else:
                # Fallback to GPT generation
                result = generate_recipes(base)  # 🔹 NEW: seed generator with base as well
                text_en = result[0] if result else "No recipes generated."

                if display_lang not in ("en", "auto"):
                    from src.translator import from_en
                    text_en = from_en(text_en, display_lang)

                lines = [ln.strip() for ln in text_en.split("\n") if ln.strip()]
                suggestions = [ln for ln in lines if not ln.lower().startswith(("ingredients", "directions", "title:"))]
                titles = _translate_titles(suggestions, display_lang)
                bullets = "• " + "\n• ".join(titles[:ask_k])

    except Exception as e:
        print(f"[search error] {e}")
        bullets = _bullets_n(_fallback_titles(user_input), ask_k)

    payload = {"suggestions": bullets, "top": [], "currency": "USD", "filters": {}}

    try:
        chat_path = get_active_chat_path()
        if chat_path:
            _append_msg(chat_path, "user", translated_ui)
            _append_msg(chat_path, "assistant", bullets, meta={"payload": payload})
    except Exception as e:
        print("[chat save] error:", e)

    return jsonify(payload)

from flask import jsonify
from src.translator import transliterate_punjabi, match_punjabi_recipe, translate_text
from src.recipe_search import get_recipe_by_title


from src.recipe_search import get_recipe_by_title

@app.route("/api/get_recipe_detail", methods=["POST"])
def get_recipe_detail():
    import ast, re
    from src.translator import transliterate_text, from_en

    def _coerce_list(x):
        if x is None:
            return []
        if isinstance(x, list):
            return [str(i).strip() for i in x]
        s = str(x).strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            try:
                val = ast.literal_eval(s)
                if isinstance(val, (list, tuple)):
                    return [str(i).strip() for i in val]
            except Exception:
                pass
        parts = re.split(r"\r?\n|;", s)
        return [p.strip(" \t\"'") for p in parts if p.strip(" \t\"'")]

    def _dedupe(seq):
        seen, out = set(), []
        for i in seq:
            k = re.sub(r"\s+", " ", i.strip())
            if k and k not in seen:
                seen.add(k)
                out.append(k)
        return out

    data = request.json or {}
    title_ui = (data.get("title") or "").strip()
    display_lang = data.get("display_lang", "en")

    print("[Detail] UI title:", title_ui)
    print("[Detail] Display lang:", display_lang)

    # ✅ Transliterate title back to English for searching
    title_en = transliterate_text(title_ui, display_lang) if display_lang not in ("en", "auto") else title_ui
    print("[Detail] Transliterated title:", title_en)
    print("[Transliteration Debug] Raw title_en:", repr(title_en)) 

    # 🪙 Punjabi-specific fallback using transliteration match
    if display_lang == "pa":
        from src.recipe_search import match_punjabi_recipe, get_recipe_by_title
        try:
            matched_titles = match_punjabi_recipe(title_en)
            if matched_titles:
                print("[Punjabi match] Found:", matched_titles)
            rows = get_recipe_by_title(matched_titles[0])
            if rows and isinstance(rows, list) and isinstance(rows[0], dict):
                row = rows[0]
                result = {
                    "title": row.get("title", ""),
                    "ingredients": row.get("ingredients", "").split("\n"),
                    "steps": row.get("directions", "").split("\n"),
                    "link": row.get("link", "")
                }
        except Exception as e:
            print("[Punjabi fallback error]", e)

    from difflib import get_close_matches
    from src.recipe_search import DF
    if DF is None:
        from src.recipe_search import init_search_engine
        init_search_engine()

    # Get top 5 closest matches from CSV titles
    title_list = DF['title'].dropna().astype(str).str.lower().tolist()
    closest_titles = get_close_matches(title_en.lower(), title_list, n=5, cutoff=0.5)

    from src.translator import transliterate_text, from_en, to_en

    # Prefer true translation for CJK/Thai; otherwise romanize
    if display_lang in ("en", "auto"):
        title_en = title_ui
    else:
        if display_lang in ("zh", "ja", "ko", "th"):
            title_en = to_en(title_ui, display_lang) or title_ui
        else:
            title_en = transliterate_text(title_ui, display_lang) or title_ui

    print("[Detail] Title EN (via translate/romanize):", title_en)

    from difflib import get_close_matches
    from src.recipe_search import DF, init_search_engine
    if DF is None:
        init_search_engine()

    title_list = DF['title'].dropna().astype(str).str.lower().tolist()
    cand_a = get_close_matches(title_en.lower(), title_list, n=5, cutoff=0.5)
    roman = transliterate_text(title_ui, display_lang) if display_lang not in ("en", "auto") else ""
    cand_b = get_close_matches(roman.lower(), title_list, n=5, cutoff=0.5) if roman else []
    closest_titles = list(dict.fromkeys(cand_a + cand_b))[:5]

    print("[Detail] Candidates (A=translation):", cand_a)
    print("[Detail] Candidates (B=roman):", cand_b)
    print("[Detail] Candidates (merged):", closest_titles)



    result = None
    if closest_titles:
        best_match = closest_titles[0]
        print("[Detail] Closest match candidates:", closest_titles)
        print("[Detail] Using best match:", best_match)

        row = DF[DF['title'].str.lower() == best_match].iloc[0]
        result = {
            "title": row["title"],
            "ingredients": row["ingredients"].split("\n"),
            "steps": row["directions"].split("\n"),
            "link": row.get("link", "")
        }

    if result:
        ingredients = _coerce_list(result.get("ingredients"))
        steps       = _coerce_list(result.get("steps"))

        strip_junk = lambda s: s.strip().strip("[](){}\"'“”‘’,。！？!?")
        clean_line = lambda s: re.sub(r"\s+", " ", s).strip()

        ingredients = [clean_line(strip_junk(i)) for i in ingredients]
        steps       = [clean_line(strip_junk(s)) for s in steps]

        ingredients = _dedupe([i for i in ingredients if len(i) > 1])
        steps       = _dedupe([s for s in steps if len(s) > 1])


        if display_lang not in ("en", "auto"):
            try:
                def clean_for_translation(lines):
                    cleaned = []
                    for s in lines:
                        s = re.sub(r"[^a-zA-Z0-9가-힣\s\.,:()\-\+\/]", "", s)  # Remove junk symbols
                        s = re.sub(r"\s+", " ", s).strip()  # Normalize spaces
                        if not s.endswith("."):
                            s += "."  # Ensure sentence ends with a period for context
                        cleaned.append(s)
                    return "\n".join(cleaned)
                
                

                ingredients_text = clean_for_translation(ingredients)
                steps_text = clean_for_translation(steps)

                translated_ingredients = from_en(ingredients_text, display_lang).replace("<unk>", "").split("\n")
                translated_text = from_en(steps_text, display_lang).replace("<unk>", "")

                translated_steps = re.split(r"\d+\.\s*", translated_text)  # splits at "1. ", "2. ", etc.
                translated_steps = [s.strip() for s in translated_steps if len(s.strip()) > 1]


                ingredients = _dedupe([s.strip() for s in translated_ingredients if len(s.strip()) > 1])
                steps = _dedupe([s.strip() for s in translated_steps if len(s.strip()) > 1])

                # if display_lang not in ("en", "auto"):
                #     try:
                #         translated_ingredients = [from_en(i, display_lang) for i in ingredients]
                #         translated_steps = [from_en(s, display_lang) for s in steps]
                #         ingredients = _dedupe([s.strip() for s in translated_ingredients if len(s.strip()) > 1])
                #         steps = _dedupe([s.strip() for s in translated_steps if len(s.strip()) > 1])
                #     except Exception as e:
                #         print("[get_recipe_detail] batch translation error:", e)


                print("[Detail] Ingredients (translated):", ingredients)
                print("[Detail] Steps (translated):", steps)

            except Exception as e:
                print("[get_recipe_detail] batch translation error:", e)

        raw_text = (
            f"{title_ui}\nIngredients:\n" + "\n".join(ingredients) +
            "\n\nSteps:\n" + "\n".join(steps)
        )
    else:
        ingredients = ["No ingredients found for this recipe."]
        steps = ["No steps found for this recipe."]
        raw_text = "No recipe found in the database."

    try:
        chat_path = get_active_chat_path()
        if chat_path:
            _append_msg(chat_path, "user", f"Selected recipe: {title_ui}")
            detail_text = (
                "Ingredients:\n- " + "\n- ".join(ingredients) +
                "\n\nSteps:\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
            )
            _append_msg(chat_path, "assistant", detail_text, meta={
                "payload": {"recipe_detail": {"title": title_ui, "ingredients": ingredients, "steps": steps}}
            })
    except Exception as e:
        print("[get_recipe_detail] chat save error:", e)

    return jsonify({"found": bool(result), "ingredients": ingredients, "steps": steps, "raw": raw_text})

init_search_engine(str(FULL_DATASET))

@app.route('/icons8-microphone-48.png')
def serve_mic_icon():
    return send_from_directory(ROOT, "icons8-microphone-48.png")

from src.voice_test import record_and_transcribe_mem, record_and_transcribe

from flask import request
import tempfile
import os
import numpy as np
import sounddevice as sd
from src.voice_test import transcribe_buffer

@app.route("/api/voice_input", methods=["POST"])
def api_voice_input():
    try:
        # Record audio directly into memory buffer (7 seconds here)
        duration = 7  # seconds
        fs = 16000    # sample rate

        print(f"[voice] Recording {duration}s at {fs}Hz")
        audio_buffer = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()  # wait until recording is finished

        audio_buffer = audio_buffer[:, 0]  # mono channel data as 1D numpy array
        np.clip(audio_buffer, -1, 1, out=audio_buffer)  # ensure valid range

        # Transcribe from buffer (no files, no ffmpeg)
        transcript = transcribe_buffer(audio_buffer, fs=fs, model_name="base")  # specify model size if needed
        
        if not transcript.strip():
            raise RuntimeError("Empty transcript (try speaking louder/closer).")

        return jsonify({"text": transcript})

    except RuntimeError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "Voice processing failed: " + str(e)}), 500

@app.route("/api/get_nutrition", methods=["POST"])
def get_nutrition():
    import json, re
    from transformers import pipeline, set_seed
    data = request.get_json(force=True) or {}
    title = (data.get("title") or "").strip()

    # 1) Get ingredients by title (your helper)
    result = get_recipe_by_title(title)
    if not result:
        fallback = "Recipe not found."
        if data.get("display_lang") not in ("en", "auto"):
            fallback = from_en(fallback, data["display_lang"])
        return jsonify({"text": fallback})

    ingredients = result.get("ingredients") or []

    from src.paths import app_root

    # 2) Ask GPT-2 for JSON ONLY
    set_seed(42)
    from src.paths import model_dir
    gen = pipeline("text-generation", model=str(model_dir("gpt2")))

    schema = (
        '{"calories_kcal": <number>, "protein_g": <number>, '
        '"carbs_g": <number>, "fat_g": <number>, "fiber_g": <number>}'
    )
    prompt = (
        "Estimate macronutrients per 1 serving based ONLY on the ingredients.\n"
        "Return JSON ONLY (no extra text), numeric values only.\n"
        f"Schema example: {schema}\n\n"
        "Ingredients:\n" + "\n".join(f"- {i}" for i in ingredients[:20]) + "\n\nJSON:"
    )

    out = gen(
        prompt,
        max_new_tokens=90,
        do_sample=False,            # make it deterministic
        temperature=0.2,
        top_p=0.9,
        repetition_penalty=1.2,
        eos_token_id=gen.tokenizer.eos_token_id
    )[0]["generated_text"]

    # 3) Parse first JSON-looking block
    m = re.search(r"\{[^\{\}]*\}", out, re.S)
    nutrition_obj = {}
    if m:
        try:
            obj = json.loads(m.group(0))
            # keep only numeric fields
            for k, v in obj.items():
                try:
                    nutrition_obj[k] = float(v)
                except Exception:
                    pass
        except Exception:
            pass

    # 4) Pretty print (or fallback text)
    labels = {
        "calories_kcal": "Calories (kcal)",
        "protein_g": "Protein (g)",
        "carbs_g": "Carbs (g)",
        "fat_g": "Fat (g)",
        "fiber_g": "Fiber (g)"
    }
    if nutrition_obj:
        lines = [f"{labels.get(k,k)}: {round(v,2)}" for k, v in nutrition_obj.items()]
        text = "\n".join(lines)
    else:
        text = "Nutrition info not found (gpt2)."

    return jsonify({"text": text, "nutrition": nutrition_obj})

def extract_keywords(ingredient: str):
    s = ingredient.lower()
    s = re.sub(r"[\d\./]+", "", s)  # remove numbers and fractions
    s = re.sub(r"\b(c|cup|tbsp|tsp|oz|can|pkg|envelope|g|ml|l|pound|lb|dash|slice|slices|inch|cm)\b", "", s)
    s = re.sub(r"[()\[\]{},.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    parts = s.split()
    keywords = []

    # full phrase, sliding phrases, individual words
    if len(parts) >= 2:
        keywords.append(" ".join(parts))
    for i in range(len(parts)):
        for j in range(i+1, len(parts)+1):
            phrase = " ".join(parts[i:j])
            if len(phrase) >= 3:
                keywords.append(phrase)
    keywords += parts  # add single words at end
    return list(dict.fromkeys(keywords))  # dedupe

from src.file_io import get_active_chat_path, read_json

@app.route("/api/check_nutrition", methods=["POST"])
def check_nutrition():
    import pandas as pd, re, json
    from src.file_io import get_active_chat_path, read_json

    chat_path = get_active_chat_path()
    if not chat_path or not chat_path.exists():
        return jsonify({"text": "❌ No chat path provided."})

    data = request.get_json(force=True) or {}
    ingredients = data.get("ingredients")
    display_lang = data.get("display_lang", "en")

    # 🧠 Translate ingredients to English if not in English
    if display_lang not in ("en", "auto") and ingredients:
            try:
                if display_lang not in ("en", "auto") and ingredients:
                    from src.translator import transliterate_text
                    ingredients = [transliterate_text(ing, display_lang) for ing in ingredients]
            except Exception as e:
                print("[check_nutrition] translation failed:", e)

            # ✅ Always run this — even for English input
            split_ingredients = []
            for ing in ingredients:
                if len(ing) > 100 and ',' not in ing:
                    parts = re.split(r"\b(?:and|or|with|,|;|\n)\b", ing)
                    split_ingredients.extend([p.strip() for p in parts if len(p.strip()) > 2])
                else:
                    split_ingredients.append(ing)
            ingredients = split_ingredients

            
            print("[check_nutrition] translation failed:", e)

    if not ingredients:
        # fallback to last_payload if no explicit ingredients passed
        doc = read_json(chat_path, default={})
        ingredients = (doc.get("last_payload", {})
                        .get("recipe_detail", {})
                        .get("ingredients", []))


    if not ingredients:
        return jsonify({"text": "❌ No ingredients found in chat file."})

    # === Load CSV once ===
    global _KAGGLE_NUTRITION_DF
    try:
        _KAGGLE_NUTRITION_DF
    except NameError:
        _KAGGLE_NUTRITION_DF = pd.read_csv("cleaned_ingredients.csv")
        _KAGGLE_NUTRITION_DF["ingredient_name_lc"] = (
            _KAGGLE_NUTRITION_DF["Descrip"].astype(str).str.lower().str.strip()
        )

    df = _KAGGLE_NUTRITION_DF

    # --- Helper: clean & extract keywords ---
    def extract_main_words(text):
        text = text.lower()
        text = re.sub(r"[\d\./]+", " ", text)  # remove numbers/fractions
        text = re.sub(r"\b(cup|cups|tablespoon|tbsp|teaspoon|tsp|oz|g|ml|lb|pound|envelopes?|can|pkg)\b", " ", text)
        text = re.sub(r"[^a-z\s]", " ", text)  # keep only letters
        words = [w.strip() for w in text.split() if len(w.strip()) > 2]
        return words

    totals = {"Calories (kcal)": 0, "Carbs (g)": 0, "Fat (g)": 0,
              "Protein (g)": 0, "Sugar (g)": 0, "Sodium (mg)": 0}

    matched = []

    for raw_ing in ingredients:
        keywords = extract_main_words(raw_ing)
        found_any = False
        for kw in keywords:
            matches = df[df["ingredient_name_lc"].str.contains(rf"\b{kw}\b", regex=True)]
            if not matches.empty:
                row = matches.iloc[0]
                found_any = True
                matched.append(kw)
                # Sum nutrients
                for col, label in {
                    "Energy_kcal": "Calories (kcal)",
                    "Carb_g": "Carbs (g)",
                    "Fat_g": "Fat (g)",
                    "Protein_g": "Protein (g)",
                    "Sugar_g": "Sugar (g)",
                    "Sodium_mg": "Sodium (mg)"
                }.items():
                    if col in row and pd.notna(row[col]):
                        totals[label] += float(row[col])
        if not found_any:
            matched.append(f"{raw_ing} ❌ No match")

    # --- Format output ---
    lines = [f"{k}: {round(v,2)}" for k,v in totals.items()]
    text = "🍎 Combined Nutrition (per recipe approx):\n" + "\n".join(lines)

    if matched:
        text += "\n\n🔍 Matched Ingredients:\n" + ", ".join(matched)

        if display_lang not in ("en", "auto"):
            from src.translator import from_en
            text = from_en(text, display_lang)


    return jsonify({"text": text, "nutrition": totals})

if __name__ == "__main__":
    import webbrowser, threading, socket
    def pick_free_port():
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        return port

    ensure_dirs()
    port = pick_free_port()
    url = f"http://127.0.0.1:{port}"
    threading.Timer(0.7, lambda: webbrowser.open(url)).start()
    print(f"🚀 Offline server at {url} (models will load lazily on first request)")
    app.run(host="127.0.0.1", port=port, debug=False)
