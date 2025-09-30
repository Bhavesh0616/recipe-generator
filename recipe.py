# recipe.py
from __future__ import annotations
import os, threading
from typing import Dict, List
from llama_cpp import Llama

from trans import (
    safe_lang_code,
    normalize_query,                  # if you call it here
    mt_enabled,
    translate_bullets_m2m,
    translate_list_m2m,
    translate_text_m2m_cached,        # for title back-translation
    section_labels,                   # for localized headers (if you build text here)
)

import re
from typing import List
from trans import (
    safe_lang_code, mt_enabled,
    translate_text_m2m_cached,
)



# -------- model path resolution (no hardcoding required) ----------
DEFAULT_REL = os.path.join("models", "mistral-7b-instruct-v0.2.Q4_K_M.gguf")
FALLBACK = r"F:\foodchat_clean\models\mistral-7b-instruct-v0.2\mistral-7b-instruct-v0.2.Q4_K_M.gguf"

MODEL_PATH = (
    os.getenv("LLAMA_GGUF_PATH")
    or (DEFAULT_REL if os.path.exists(DEFAULT_REL) else (FALLBACK if os.path.exists(FALLBACK) else ""))
)
if not MODEL_PATH:
    raise FileNotFoundError("No GGUF model found. Set LLAMA_GGUF_PATH or put it under ./models/")

# -------- single instance + single-file global lock ---------------
_LLM: Llama | None = None
_LOAD_LOCK = threading.Lock()
_REQ_LOCK = threading.Lock()   # <-- serialize each completion

def _llm() -> Llama:
    global _LLM
    if _LLM is not None:
        return _LLM
    with _LOAD_LOCK:
        if _LLM is None:
            _LLM = Llama(
                model_path=MODEL_PATH,
                n_ctx=int(os.getenv("LLAMA_CTX", "2048")),
                n_threads=max(2, (os.cpu_count() or 8) - 2),
                n_batch=int(os.getenv("LLAMA_N_BATCH", "128")),
                n_gpu_layers=int(os.getenv("LLAMA_GPU_LAYERS", "0")),  # keep 0 for Windows CPU stability
                use_mmap=True,
                mlock=False,
                seed=0,
                chat_format="mistral-instruct",
                verbose=False,
            )
    return _LLM

# ----------------- shared system prompt ----------------------------
SYS = (
    "You are an offline culinary assistant. "
    "Be concise and always return clean, readable text. Use bullet lists when listing multiple items."
)

def _chat(user: str, max_tokens=512, temperature=0.3) -> str:
    """All llama calls go through this function; it is serialized and safe."""
    with _REQ_LOCK:
        out = _llm().create_chat_completion(
            messages=[{"role": "system", "content": SYS},
                      {"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.95,
            repeat_penalty=1.1,
        )
    return out["choices"][0]["message"]["content"]

# -------------- public helpers ------------------------------------
def _extract_titles(en_text: str) -> List[str]:
    """
    Robustly extract titles from a variety of bullet styles that GGUF may return.
    We strip markdown and keep only the title part.
    """
    items: List[str] = []
    # split by lines and by '*' bullets that sometimes come inline
    raw_lines = []
    for line in en_text.splitlines():
        s = line.strip()
        if not s:
            continue
        # break inline bullets into separate lines
        if " * " in s:
            raw_lines.extend([p.strip() for p in s.split(" * ") if p.strip()])
        else:
            raw_lines.append(s)

    for s in raw_lines:
        # remove leading bullet marks
        s = re.sub(r'^[\-\*\u2022]+\s*', '', s)
        # drop bold markdown wrappers
        s = s.replace("**", "").strip()
        # if there's a colon with a description, keep the left side as the title
        s = s.split(":", 1)[0].strip()
        # remove surrounding quotes
        s = s.strip(' "“”')
        if s:
            items.append(s)
    # keep 6 max
    return items[:6]

def suggest_recipes(query: str, display_lang: str = "auto") -> str:
    code = safe_lang_code(display_lang)
    print(f"[SUGGEST] code='{code}' mt_enabled={mt_enabled()} query_for_gguf='{query}'")

    # 1) Ask GGUF for titles only, one per line
    prompt = (
        "You are a culinary assistant. "
        "User query: " + query + "\n\n"
        "Return 6 recipe TITLES only, one per line. "
        "No descriptions, no extra text, no numbering, no markdown bold. Example:\n"
        "- Vegetarian Sushi Roll\n- Salmon Sushi Bowls\n- Maki-zushi\n- Temaki Hand Rolls\n- Sushi Rice\n- Tofu Sushi Cake\n"
    )
    try:
        text_en = _chat(prompt, max_tokens=220, temperature=0.5)
    except Exception as e:
        print("[SUGGEST] llama error:", e)
        text_en = "- Vegetarian Sushi Roll\n- Salmon Sushi Bowls\n- Maki-zushi\n- Temaki Hand Rolls\n- Sushi Rice\n- Tofu Sushi Cake"

    print("[SUGGEST] gguf_output_en:\n" + text_en)

    # 2) Parse to clean English titles
    titles_en = _extract_titles(text_en)
    print(f"[SUGGEST] parsed_en_titles={titles_en}")

    # 3) Translate each title (preserves bullet lines), if needed
    if code != "en" and mt_enabled():
        titles_loc = []
        for t in titles_en:
            try:
                t_loc = translate_text_m2m_cached(t, code, src_code="en")
            except Exception as te:
                print("[SUGGEST] title translate fail:", te, "| title:", t)
                t_loc = t
            titles_loc.append(t_loc)
        titles = titles_loc
        print(f"[SUGGEST] translated_titles[{code}]={titles}")
    else:
        if code != "en":
            print("[SUGGEST] MT disabled; returning English titles")
        titles = titles_en

    # 4) Rebuild in the exact clickable format the UI expects: - "Title"
    output = "\n".join(f'- "{t}"' for t in titles)
    print("[SUGGEST] final_output:\n" + output)
    return output


def get_recipe_detail(title: str, display_lang: str = "auto") -> Dict[str, List[str]]:
    code = safe_lang_code(display_lang)
    print(f"[DETAIL] incoming title_ui='{title}' lang='{code}' mt_enabled={mt_enabled()}")  # <---

    # Optional: back-translate the UI title to English so GGUF is stable
    title_en = title
    if code != "en" and mt_enabled():
        from trans import translate_text_m2m_cached
        try:
            title_en = translate_text_m2m_cached(title, "en", src_code=code)
            print(f"[DETAIL] title_ui -> en: '{title}' -> '{title_en}'")  # <---
        except Exception as e:
            print("[DETAIL] title back-translation fail:", e)

    prompt = (
        f"Write the recipe for '{title_en}' in English.\n"
        "Format strictly as:\n"
        "Ingredients:\n"
        "- item 1\n- item 2\n...\n\n"
        "Steps:\n"
        "1. step\n2. step\n..."
    )
    try:
        text = _chat(prompt, max_tokens=700, temperature=0.4)
    except Exception as e:
        print("[DETAIL] llama error:", e)  # <---
        return {
            "ingredients": [
                "2 cups sushi rice", "2 1/2 cups water", "1/2 cup rice vinegar",
                "2 tbsp sugar", "1 tsp salt", "nori sheets", "fillings of choice"
            ],
            "steps": [
                "Rinse rice until water is clear; drain.",
                "Cook rice with water; rest 10 min.",
                "Warm vinegar, sugar, salt; fold into rice.",
                "Roll with nori and fillings; slice and serve."
            ],
        }

    print("[DETAIL] gguf_raw_en:\n" + text[:600] + ("\n...[truncated]" if len(text) > 600 else ""))  # <---

    # Parse EN sections
    ingredients: List[str] = []
    steps: List[str] = []
    section = None
    for line in text.splitlines():
        s = line.strip()
        low = s.lower()
        if low.startswith("ingredients"):
            section = "ing"; continue
        if low.startswith("steps") or low.startswith("method"):
            section = "steps"; continue
        if section == "ing" and (s.startswith("-") or s.startswith("•")):
            ingredients.append(s.lstrip("-• ").strip())
        elif section == "steps":
            if s and (s[0].isdigit() and "." in s[:4]):
                s = s.split(".", 1)[1].strip()
            if s:
                steps.append(s)

    if not ingredients and not steps:
        print("[DETAIL] parse fallback engaged")  # <---
        for l in text.splitlines():
            t = l.strip()
            if t.startswith(("-", "•")):
                ingredients.append(t.lstrip("-• ").strip())
            elif len(t) > 2:
                steps.append(t)
        steps, ingredients = steps[:20], ingredients[:40]

    print(f"[DETAIL] parsed_en ings={len(ingredients)} steps={len(steps)}")  # <---
    if ingredients[:2] or steps[:2]:
        print(f"[DETAIL] sample_ing='{ingredients[:2]}' sample_step='{steps[:2]}'")  # <---

    # Translate EN -> UI language
    if code != "en" and mt_enabled():
        try:
            ingredients = translate_list_m2m(ingredients, code, src_code="en")
            steps = translate_list_m2m(steps, code, src_code="en")
            print(f"[DETAIL] translated_to_{code} ings[0:2]={ingredients[:2]} steps[0:2]={steps[:2]}")  # <---
        except Exception as e:
            print("[MT] detail translate fail:", e)

    return {"ingredients": ingredients[:40], "steps": steps[:20]}



