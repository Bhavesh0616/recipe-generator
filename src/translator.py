# src/translator.py
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# use those variables when you load files / models
from src.paths import assets_dir, app_root
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FULL_DATASET_CSV = DATA_DIR / "full_dataset.csv"

TRANS_DIR = str(assets_dir() / "translation_models")


_trans_tok = None
_trans_mdl = None

_SUPPORTED_LANGS = set([
'af', 'am', 'ar', 'ast', 'az', 'be', 'bem', 'ber', 'bg', 'bn', 'br', 'bs',
'ca', 'ceb', 'cs', 'cy', 'da', 'de', 'dv', 'dz', 'el', 'en', 'eo', 'es',
'et', 'eu', 'fa', 'fi', 'fo', 'fr', 'fr_CA', 'ga', 'gd', 'gl', 'gu', 'ha',
'haw', 'he', 'hi', 'hr', 'ht', 'hu', 'hy', 'id', 'ig', 'is', 'it', 'ja',
'jv', 'ka', 'kab', 'kk', 'kl', 'km', 'kn', 'ko', 'ku', 'ky', 'la', 'lb',
'lo', 'lt', 'lv', 'mg', 'mk', 'ml', 'mn', 'mr', 'ms', 'mt', 'my', 'ne',
'nl', 'no', 'ny', 'or', 'pa', 'pl', 'ps', 'pt', 'qu', 'ro', 'ru', 'rw',
'sd', 'si', 'sk', 'sl', 'so', 'sq', 'sr', 'su', 'sv', 'sw', 'ta', 'te',
'tg', 'th', 'ti', 'tk', 'tl', 'tr', 'tt', 'ug', 'uk', 'ur', 'uz', 'vi',
'wo', 'xh', 'yi', 'yo', 'zh', 'zh_cn', 'zh_tw', 'zu'
])

import pandas as pd
from difflib import get_close_matches
DF = None

punjabi_to_roman_map = {
    'ਅ': 'a', 'ਆ': 'aa', 'ਇ': 'i', 'ਈ': 'ee', 'ਉ': 'u', 'ਊ': 'oo', 'ੲ': 'i', 'ਐ': 'ai', 'ਏ': 'e', 'ਓ': 'o', 'ਔ': 'au',
    'ਕ': 'k', 'ਖ': 'kh', 'ਗ': 'g', 'ਘ': 'gh', 'ਙ': 'ng',
    'ਚ': 'ch', 'ਛ': 'chh', 'ਜ': 'j', 'ਝ': 'jh', 'ਞ': 'ny',
    'ਟ': 't', 'ਠ': 'th', 'ਡ': 'd', 'ਢ': 'dh', 'ਣ': 'n',
    'ਤ': 't', 'ਥ': 'th', 'ਦ': 'd', 'ਧ': 'dh', 'ਨ': 'n',
    'ਪ': 'p', 'ਫ': 'ph', 'ਬ': 'b', 'ਭ': 'bh', 'ਮ': 'm',
    'ਯ': 'y', 'ਰ': 'r', 'ਲ': 'l', 'ਵ': 'v',
    'ਸ਼': 'sh', 'ਸ': 's', 'ਹ': 'h', 'ਲ਼': 'l',
    # Dependent vowels
    'ਾ': 'a', 'ਿ': 'i', 'ੀ': 'ee', 'ੁ': 'u', 'ੂ': 'oo',
    'ੇ': 'e', 'ੈ': 'ai', 'ੋ': 'o', 'ੌ': 'au',
    'ਂ': 'n', 'ਃ': 'h', 'ੱ': '', '': '', 'ੜ': 'r',
    ' ': ' '
}

def transliterate_punjabi(text: str) -> str:
    result = []
    skip_next = False
    for i, ch in enumerate(text):
        if skip_next:
            skip_next = False
            continue
        # Combine vowel signs (rudimentary)
        if i + 1 < len(text) and text[i + 1] in ['ਾ','ਿ','ੀ','ੁ','ੂ','ੇ','ੈ','ੋ','ੌ']:
            result.append(punjabi_to_roman_map.get(ch, ch) + punjabi_to_roman_map.get(text[i + 1], ''))
            skip_next = True
        else:
            result.append(punjabi_to_roman_map.get(ch, ch))
    return ''.join(result)

from src.paths import app_root

def match_punjabi_recipe(text: str, csv_path=None) -> list[str]:
    if csv_path is None:
        csv_path = str(app_root() / "appdata" / "full_dataset.csv")
    # Transliterate Punjabi input to Roman
    roman_text = transliterate_punjabi(text)

    # Ensure the global search DF is loaded
    import src.recipe_search as rs
    if rs.DF is None:
        rs.init_search_engine(csv_path)

    titles = [str(t) for t in rs.DF['title'].fillna('')]
    lowered = [t.lower() for t in titles]

    matches = get_close_matches(roman_text.lower(), lowered, n=3, cutoff=0.5)
    return [titles[lowered.index(m)] for m in matches]


def safe_lang_code(code):
    code = (code or '').strip().lower()
    if code not in _SUPPORTED_LANGS:
        raise ValueError(f"Unsupported language code: {code}")
    return code

def _ensure_loaded():
    global _trans_tok, _trans_mdl
    if _trans_tok is None or _trans_mdl is None:
        _trans_tok = M2M100Tokenizer.from_pretrained(TRANS_DIR, local_files_only=True)
        _trans_mdl = M2M100ForConditionalGeneration.from_pretrained(
            TRANS_DIR, local_files_only=True, low_cpu_mem_usage=True
        )

def translate_any(text: str, src_lang: str, tgt_lang: str) -> str:
    if not text or src_lang == tgt_lang:
        return text
    try:
        _ensure_loaded()
        _trans_tok.src_lang = src_lang
        enc = _trans_tok(text, return_tensors="pt")
        out = _trans_mdl.generate(**enc, forced_bos_token_id=_trans_tok.get_lang_id(tgt_lang))
        return _trans_tok.batch_decode(out, skip_special_tokens=True)[0]
    except Exception as e:
        import traceback
        print("[from_en] fail:", e)
        traceback.print_exc()
        return text

def to_en(text: str, src_lang: str) -> str:
    if src_lang in ("auto", "en", "", None):
        return text
    return translate_any(text, src_lang, "en")



# def from_en(text: str, lang_code: str) -> str:
#     text = text.strip()
#     if not text or lang_code in ("en", "auto"):
#         return text

#     inputs = tokenizer(text, return_tensors="pt", truncation=True)
#     generated_tokens = model.generate(
#         **inputs,
#         forced_bos_token_id=tokenizer.lang_code_to_id[lang_code],
#         max_new_tokens=128
#     )
#     translated = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
#     return translated

# REMOVE the second MODEL_PATH/tokenizer/model block.
# Re-implement from_en using the same _ensure_loaded() pair:

def from_en(text: str, lang_code: str) -> str:
    text = (text or "").strip()
    if not text or lang_code in ("en", "auto"): return text
    try:
        _ensure_loaded()
        _trans_tok.src_lang = "en"
        enc = _trans_tok(text, return_tensors="pt", truncation=True)
        out = _trans_mdl.generate(
            **enc, forced_bos_token_id=_trans_tok.get_lang_id(lang_code), max_new_tokens=256
        )
        return _trans_tok.batch_decode(out, skip_special_tokens=True)[0]
    except Exception as e:
        print("[from_en] fail:", e)
        return text


def healthy():
    try:
        _ensure_loaded()
        return True
    except Exception as e:
        print(f"[translator] health fail: {e}")
        return False

import re
# New helper for Hindi
def transliterate_hindi(text: str) -> str:
    from indic_transliteration.sanscript import transliterate, DEVANAGARI, ITRANS
    try:
        return re.sub(r"\s+", " ", transliterate(text, DEVANAGARI, ITRANS)).lower().strip()
    except Exception:
        return text


# ---------------- Transliteration logic ----------------
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from indic_transliteration import sanscript



from pypinyin import pinyin, Style
import jieba
import unicodedata

def _capitalize_word(w: str) -> str:
    return w[0].upper() + w[1:] if w else w

def zh_to_ascii_words(text: str) -> str:
    """Chinese -> ASCII pinyin without tones, word-spaced."""
    words = [w for w in jieba.lcut(text) if w.strip()]
    roman_words = []
    for w in words:
        # NORMAL gives no tone marks; join syllables per word
        syl = [s[0] for s in pinyin(w, style=Style.NORMAL, strict=False)]
        roman_words.append("".join(syl))
    return " ".join(roman_words)

import re
from pythainlp import word_tokenize
from pythainlp.transliterate import romanize

def th_to_ascii_words(text: str) -> str:
    # Thai has no spaces; segment first
    toks = [t for t in word_tokenize(text, engine="newmm") if t.strip()]
    romans = []
    for t in toks:
        r = romanize(t, engine="thai2rom")  # nice Latin output
        if not r:
            r = romanize(t, engine="royin")  # fallback (RTGS)
        # keep only ASCII letters/digits; drop hyphens, tone marks, dots, etc.
        r = re.sub(r"[^A-Za-z0-9]+", "", r)
        if r:
            romans.append(r.lower())
    return " ".join(romans)  # e.g., "sa te" -> "sate" (depending on engine models)

def transliterate_text(text: str, lang: str) -> str:
    if not text or lang in ("en", "auto"):
        return text
    try:
        if lang == "hi":
            return transliterate_hindi(text)

        # For ko/ja you currently translate-to-English (OK for your CSV)
        if lang in ("ko", "ja"):
            return to_en(text, lang)

        if lang == "zh":
            # ASCII word-spaced pinyin: "北京烤鸭" -> "beijing kaoya"
            return zh_to_ascii_words(text)

        if lang == "th":
            return th_to_ascii_words(text)

        if lang in ("tr", "vi"):
            # strip accents; both are Latin scripts already
            return "".join(
                c for c in unicodedata.normalize("NFD", text)
                if unicodedata.category(c) != "Mn"
            )

        if lang == "ru":
            from transliterate import translit
            return translit(text, 'ru', reversed=True)

        if lang == "ur":

            from urduhack.normalization import romanize
            return romanize(text)

        if lang == "pa":
            return transliterate_punjabi(text)

        return text
    except Exception as e:
        print(f"[transliterate_text fallback] {e}")
        return text

    # try:
    #     # Use M2M100 model-based translation to English for KO/JA and fallback
    #     # Hindi → use indic-transliteration
    #     if lang == "hi":
    #         from indic_transliteration.sanscript import transliterate, DEVANAGARI, ITRANS
    #         try:
    #             return transliterate(text, DEVANAGARI, ITRANS).lower
    #         except Exception as e:
    #             print("[Hindi transliteration error]", e)
    #             return text

    #     # Korean/Japanese → fallback to M2M100 (translate-to-English)
    #     if lang in ("ko", "ja"):
    #         return to_en(text, lang)

    #     # Default fallback for languages with romanizable scripts
    #     return text

def process_user_input(text: str, selected_lang: str) -> tuple[str, str]:
    """
    Returns:
      roman_text   -> transliterated/romanized for DB search
      ui_text      -> original user text (preserve style for response)
    """
    text = (text or "").strip()
    if not text or selected_lang in ("auto", "en"):
        return text, text

    try:
        roman = transliterate_text(text, selected_lang)
        return roman, text   # roman used for DB, text kept for display
    except Exception as e:
        print(f"[transliteration fallback] {e}")
        return text, text

# --- add in src/translator.py ---
def detect_lang_by_script(text: str) -> str:
    t = (text or "").strip()
    if any('\u0900' <= ch <= '\u097F' for ch in t): return "hi"  # Devanagari
    if any('\u3040' <= ch <= '\u30FF' for ch in t): return "ja"  # Japanese
    if any('\uAC00' <= ch <= '\uD7AF' for ch in t): return "ko"  # Korean
    if any('\u0400' <= ch <= '\u04FF' for ch in t): return "ru"  # Cyrillic
    if any('\u4E00' <= ch <= '\u9FFF' for ch in t): return "zh"  # Chinese Han
    if any('\u0E00' <= ch <= '\u0E7F' for ch in t): return "th"  # Thai

    turkish_chars = set("ğĞıİşŞçÇöÖüÜ")
    if any(ch in turkish_chars for ch in t): return "tr"

    vietnamese_chars = set("ĂÂÊÔƠƯăâêôơưĐđ")
    if any(ch in vietnamese_chars for ch in t): return "vi"

    return "en"



translate_text = translate_any

