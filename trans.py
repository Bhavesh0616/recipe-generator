# trans.py
from __future__ import annotations
import unicodedata as ud

# Languages we show in the UI (expand as you like)
SUPPORTED = {
    "auto","en","fr","de","es","it","pt","ru","ja","ko","zh","th","tr","vi",
    "hi","bn","mr","ta","te","gu","kn","ml","pa",
    # Add UI extras safely – they’ll fall back to English if MT lacks them:
    "sw","lg","ln","rw","so","am","ha","yo","zu","xh","st","sn","ne","si","ur"
}

# The M2M100-418M model’s *known* 2-letter codes (not exhaustive, but safe).
# If a code is not here, we won’t attempt MT for it.
MT_SUPPORTED = {
    "en","fr","de","es","it","pt","ru","ja","ko","zh","tr","vi","th",
    "hi","bn","mr","ta","te","gu","kn","ml","pa","ur",
    "sw","lg","ln","yo","am","ha","so","ne","si","rw","sn","zu","xh","st"
}

def _m2m_can(code: str) -> bool:
    return code in MT_SUPPORTED


def safe_lang_code(code: str) -> str:
    c = (code or "auto").lower().strip()
    return c if c in SUPPORTED else "auto"

def strip_accents(s: str) -> str:
    # basic ASCII fold for Latin-like scripts
    nk = ud.normalize("NFKD", s)
    return "".join(ch for ch in nk if not ud.combining(ch))

def _romanize_cyrillic(s: str) -> str:
    table = {
        "А":"A","Б":"B","В":"V","Г":"G","Д":"D","Е":"E","Ё":"Yo","Ж":"Zh","З":"Z","И":"I","Й":"Y","К":"K","Л":"L","М":"M","Н":"N","О":"O","П":"P","Р":"R","С":"S","Т":"T","У":"U","Ф":"F","Х":"Kh","Ц":"Ts","Ч":"Ch","Ш":"Sh","Щ":"Shch","Ы":"Y","Э":"E","Ю":"Yu","Я":"Ya",
        "а":"a","б":"b","в":"v","г":"g","д":"d","е":"e","ё":"yo","ж":"zh","з":"z","и":"i","й":"y","к":"k","л":"l","м":"m","н":"n","о":"o","п":"p","р":"r","с":"s","т":"t","у":"u","ф":"f","х":"kh","ц":"ts","ч":"ch","ш":"sh","щ":"shch","ы":"y","э":"e","ю":"yu","я":"ya",
    }
    return "".join(table.get(ch, ch) for ch in s)

def _romanize_greek(s: str) -> str:
    table = {
        "Α":"A","Β":"V","Γ":"G","Δ":"D","Ε":"E","Ζ":"Z","Η":"I","Θ":"Th","Ι":"I","Κ":"K","Λ":"L","Μ":"M","Ν":"N","Ξ":"X","Ο":"O","Π":"P","Ρ":"R","Σ":"S","Τ":"T","Υ":"Y","Φ":"F","Χ":"Ch","Ψ":"Ps","Ω":"O",
        "α":"a","β":"v","γ":"g","δ":"d","ε":"e","ζ":"z","η":"i","θ":"th","ι":"i","κ":"k","λ":"l","μ":"m","ν":"n","ξ":"x","ο":"o","π":"p","ρ":"r","σ":"s","ς":"s","τ":"t","υ":"y","φ":"f","χ":"ch","ψ":"ps","ω":"o",
    }
    return "".join(table.get(ch, ch) for ch in s)

def _romanize_hira_kata(s: str) -> str:
    # Very small gojūon map (enough to keep app stable); extend if needed.
    table = {
        # hiragana
        "あ":"a","い":"i","う":"u","え":"e","お":"o",
        "か":"ka","き":"ki","く":"ku","け":"ke","こ":"ko",
        "さ":"sa","し":"shi","す":"su","せ":"se","そ":"so",
        "た":"ta","ち":"chi","つ":"tsu","て":"te","と":"to",
        "な":"na","に":"ni","ぬ":"nu","ね":"ne","の":"no",
        "は":"ha","ひ":"hi","ふ":"fu","へ":"he","ほ":"ho",
        "ま":"ma","み":"mi","む":"mu","め":"me","も":"mo",
        "や":"ya","ゆ":"yu","よ":"yo",
        "ら":"ra","り":"ri","る":"ru","れ":"re","ろ":"ro",
        "わ":"wa","を":"o","ん":"n",
        "が":"ga","ぎ":"gi","ぐ":"gu","げ":"ge","ご":"go",
        "ざ":"za","じ":"ji","ず":"zu","ぜ":"ze","ぞ":"zo",
        "だ":"da","ぢ":"ji","づ":"zu","で":"de","ど":"do",
        "ば":"ba","び":"bi","ぶ":"bu","べ":"be","ぼ":"bo",
        "ぱ":"pa","ぴ":"pi","ぷ":"pu","ぺ":"pe","ぽ":"po",
        # katakana (subset)
        "ア":"a","イ":"i","ウ":"u","エ":"e","オ":"o",
        "カ":"ka","キ":"ki","ク":"ku","ケ":"ke","コ":"ko",
        "サ":"sa","シ":"shi","ス":"su","セ":"se","ソ":"so",
        "タ":"ta","チ":"chi","ツ":"tsu","テ":"te","ト":"to",
        "ナ":"na","ニ":"ni","ヌ":"nu","ネ":"ne","ノ":"no",
        "ハ":"ha","ヒ":"hi","フ":"fu","ヘ":"he","ホ":"ho",
        "マ":"ma","ミ":"mi","ム":"mu","メ":"me","モ":"mo",
        "ヤ":"ya","ユ":"yu","ヨ":"yo",
        "ラ":"ra","リ":"ri","ル":"ru","レ":"re","ロ":"ro",
        "ワ":"wa","ヲ":"o","ン":"n",
    }
    return "".join(table.get(ch, ch) for ch in s)

import re

def query_to_english_for_search(text: str, lang_code: str) -> str:
    """Prefer MT to EN; if source code not supported or MT fails, fall back to transliteration."""
    lang = safe_lang_code(lang_code)
    s = (text or "").strip()
    if not s:
        return s
    if lang != "en" and _m2m_can(lang):
        try:
            en = translate_text_m2m_cached(s, "en", src_code=lang)
            if en and re.search(r"[A-Za-z]", en):
                print(f"[TRANS] query MT {lang}->en: '{s}' -> '{en}'")
                return en
        except Exception as e:
            print("[TRANS] query MT fail:", e)
    # fallback: your romanizer/stripper
    out = normalize_query(s, lang)
    print(f"[TRANS] query translit {lang}->ASCII: '{s}' -> '{out}'")
    return out


def normalize_query(text: str, lang: str) -> str:
    lang = safe_lang_code(lang)
    s = text or ""
    if not s: 
        print(f"[TRANS] normalize: empty text lang='{lang}'")  # <---
        return s
    # cheap paths first
    if lang in ("auto", "en", "fr", "de", "es", "it", "pt", "tr", "vi"):
        out = strip_accents(s)
        print(f"[TRANS] normalize: latin-ish '{s}' -> '{out}' lang='{lang}'")  # <---
        return out
    if lang == "ru":
        out = _romanize_cyrillic(s); print(f"[TRANS] normalize: ru '{s}' -> '{out}'"); return out
    if lang in ("el","gr"):
        out = _romanize_greek(s); print(f"[TRANS] normalize: el/gr '{s}' -> '{out}'"); return out
    if lang in ("ja",):
        out = _romanize_hira_kata(s); print(f"[TRANS] normalize: ja '{s}' -> '{out}'"); return out
    out = strip_accents(s)
    print(f"[TRANS] normalize: fallback '{s}' -> '{out}' lang='{lang}'")    # <---
    return out


def _ensure_m2m():
    if _MT["model"] is not None:
        return _MT["tok"], _MT["model"]
    if not _mt_enabled():
        print("[MT] disabled (TRANSLATION_BACKEND != 'm2m')")  # <---
        raise RuntimeError("M2M backend not enabled (set TRANSLATION_BACKEND=m2m)")
    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
    model_dir = _mt_model_dir()
    print("[MT] Loading M2M100 from:", model_dir)  # <---
    _MT["tok"] = M2M100Tokenizer.from_pretrained(model_dir, local_files_only=True)
    _MT["model"] = M2M100ForConditionalGeneration.from_pretrained(model_dir, local_files_only=True)
    return _MT["tok"], _MT["model"]

# --- optional MT backend: facebook/m2m100_418M via Transformers ---
# Enable by setting env: TRANSLATION_BACKEND=m2m
# Model path can be ENV TRANSLATION_MODEL_DIR, else ./assets/translation_models/facebook/m2m100_418M

from functools import lru_cache
import os
from pathlib import Path

# --- HARD SETTINGS (edit if you ever move the folder) ---
_M2M_BASE = Path(r"F:\foodchat_clean\assets\translation_models")  # parent folder
_ALWAYS_ENABLE_MT = True  # force translation backend ON

_MT = {"tok": None, "model": None, "dir": None}

# ================== HARD-CODED M2M100 TRANSLATION BACKEND ==================
# No env vars required. If files are missing, we print a warning and skip MT.

from functools import lru_cache
import os
from pathlib import Path

# --- HARD SETTINGS (edit if you ever move the folder) ---
_M2M_BASE = Path(r"F:\foodchat_clean\assets\translation_models")  # parent folder
_ALWAYS_ENABLE_MT = True  # force translation backend ON

_MT = {"tok": None, "model": None, "dir": None}

def _mt_enabled() -> bool:
    return _ALWAYS_ENABLE_MT

def mt_enabled() -> bool:  # public
    return _mt_enabled()

def _mt_model_dir() -> str:
    """
    Accepts either:
      F:\foodchat_clean\assets\translation_models\facebook\m2m100_418M\
    or a flat folder with the model files directly in ...\translation_models\
    """
    base = _M2M_BASE
    flat = base
    nested = base / "facebook" / "m2m100_418M"

    def has_files(p: Path) -> bool:
        return (p / "config.json").exists() and (
            (p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists()
        )

    if has_files(flat):
        return str(flat)
    if has_files(nested):
        return str(nested)
    # Neither exists -> return flat path to trigger a clear error in loader
    return str(flat)

def _ensure_m2m():
    if _MT["model"] is not None:
        return _MT["tok"], _MT["model"]

    if not _mt_enabled():
        raise RuntimeError("M2M backend disabled")

    model_dir = _mt_model_dir()
    _MT["dir"] = model_dir

    try:
        from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer  # type: ignore
        import sentencepiece  # noqa: F401
    except Exception as e:
        print("[MT] transformers/sentencepiece not installed in venv:", e)
        raise

    if not Path(model_dir).exists():
        raise FileNotFoundError(
            f"[MT] Model folder not found: {model_dir}\n"
            "Put m2m100_418M files either directly in this folder or in 'facebook/m2m100_418M/'."
        )

    print("[MT] Loading M2M100 from:", model_dir)
    try:
        tok = M2M100Tokenizer.from_pretrained(model_dir, local_files_only=True)
        model = M2M100ForConditionalGeneration.from_pretrained(model_dir, local_files_only=True)
    except Exception as e:
        print("[MT] Failed to load model files from:", model_dir)
        raise

    _MT["tok"], _MT["model"] = tok, model
    return tok, model

@lru_cache(maxsize=512)
def translate_text_m2m_cached(text: str, tgt_code: str, src_code: str = "en") -> str:
    tgt = safe_lang_code(tgt_code)
    src = safe_lang_code(src_code)
    if not _m2m_can(tgt) or (src and not _m2m_can(src)):
        # MT not supported – return original text
        print(f"[MT] skip (unsupported): src={src} tgt={tgt}")
        return text
    tok, model = _ensure_m2m()
    tok.src_lang = src or "en"
    forced_id = tok.get_lang_id(tgt)
    inputs = tok(text, return_tensors="pt")
    gen = model.generate(**inputs, forced_bos_token_id=forced_id, max_new_tokens=min(1024, len(text)//2 + 200))
    return tok.batch_decode(gen, skip_special_tokens=True)[0]

def translate_list_m2m(items: list[str], tgt_code: str, src_code: str = "en") -> list[str]:
    if not _m2m_can(safe_lang_code(tgt_code)) or (src_code and not _m2m_can(safe_lang_code(src_code))):
        print(f"[MT] list skip (unsupported): src={src_code} tgt={tgt_code}")
        return items
    out = []
    for it in items:
        t = (it or "").strip()
        try:
            out.append(translate_text_m2m_cached(t, tgt_code, src_code))
        except Exception as e:
            print("[MT] item translate fail:", e, "| item:", repr(it))
            out.append(it)
    return out

def translate_bullets_m2m(text_block: str, tgt_code: str, src_code: str = "en") -> str:
    if not _m2m_can(safe_lang_code(tgt_code)) or (src_code and not _m2m_can(safe_lang_code(src_code))):
        print(f"[MT] bullets skip (unsupported): src={src_code} tgt={tgt_code}")
        return text_block
    try:
        return translate_text_m2m_cached(text_block, tgt_code, src_code)
    except Exception as e:
        print("[MT] bullets translate fail:", e)
        return text_block



# ---------- tiny language helpers you already added earlier ----------
LANG_NAMES = {
    "en": "English", "ja": "Japanese", "ko": "Korean", "zh": "Chinese",
    "hi": "Hindi", "fr": "French", "de": "German", "es": "Spanish",
    "it": "Italian", "pt": "Portuguese", "tr": "Turkish", "vi": "Vietnamese",
    "th": "Thai", "ru": "Russian",
}

def target_language_name(code: str) -> str:
    c = safe_lang_code(code)
    return LANG_NAMES.get(c, "English")

def section_labels(code: str) -> tuple[str, str]:
    c = safe_lang_code(code)
    if c == "ja": return "材料", "作り方"
    if c == "hi": return "सामग्री", "विधि"
    if c == "zh": return "食材", "做法"
    if c == "ko": return "재료", "만드는 법"
    if c == "es": return "Ingredientes", "Pasos"
    if c == "fr": return "Ingrédients", "Étapes"
    if c == "de": return "Zutaten", "Zubereitung"
    if c == "it": return "Ingredienti", "Preparazione"
    if c == "pt": return "Ingredientes", "Modo de preparo"
    if c == "tr": return "Malzemeler", "Adımlar"
    if c == "vi": return "Nguyên liệu", "Cách làm"
    if c == "th": return "ส่วนผสม", "วิธีทำ"
    return "Ingredients", "Steps"

def translate_via_m2m(text: str, tgt_code: str, src_code: str | None = None) -> str:
    if not text.strip():
        return text
    tok, model = _ensure_m2m()
    # M2M expects standard ISO codes; our UI already uses 'en','ja','ko','zh',...
    if src_code:
        tok.src_lang = src_code
    else:
        # leave to model auto handling; M2M isn’t perfect at auto but OK for many cases
        tok.src_lang = "en"  # safe default if we mostly post-translate English
    forced_id = tok.get_lang_id(tgt_code)
    inputs = tok(text, return_tensors="pt")
    gen = model.generate(**inputs, forced_bos_token_id=forced_id, max_new_tokens=min(1024, len(text)//2 + 200))
    return tok.batch_decode(gen, skip_special_tokens=True)[0]

def looks_like_lang(text: str, code: str) -> bool:
    t = text[:200]
    if code == "ja":
        # contains some CJK and few Latin letters
        import re
        cjk = len(re.findall(r"[\u3040-\u30ff\u3400-\u9fff]", t))
        latin = len(re.findall(r"[A-Za-z]", t))
        return cjk >= 10 and cjk > latin
    if code in ("zh","ko"):
        import re
        cjk = len(re.findall(r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af\u3400-\u9fff]", t))
        return cjk >= 10
    # simple heuristic for non-Latin scripts can be added as needed
    return True

def post_edit_mt_noise(s: str, code: str) -> str:
    if not s: return s
    if code == "ja":
        for k,v in {"スーシー":"寿司","ソース ソース":"醤油","ソースソース":"醤油","ナイジェリ":"にぎり","カクマー":"きゅうり","サルモン":"サーモン"}.items():
            s = s.replace(k, v)
        s = re.sub(r"\s+([、。・])", r"\1", s)
    elif code == "ko":
        for k,v in {"수시":"초밥","컵":"볼","마리화나":"재워"}.items():
            s = s.replace(k, v)
    elif code in ("zh","zh-cn","zh-tw"):
        s = s.replace("<unk>", "")
        s = re.sub(r"\s+([，。；：、])", r"\1", s)
    return s

def post_edit_list(items: list[str], code: str) -> list[str]:
    return [post_edit_mt_noise(x, code) for x in items]
