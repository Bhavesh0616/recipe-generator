# healthcheck.py
from __future__ import annotations
from typing import List
from recipe import _llm, SYS  # reuse same model + system prompt
from trans import safe_lang_code, mt_enabled, translate_text_m2m_cached

# A tiny map so the model gets the intent right.
# You can add more over time without hardcoding detailed rules.
_LABELS = {
    "diabetes": "Diabetes (focus: lower added sugar / slower glycemic impact)",
    "hypertension": "Hypertension (focus: lower sodium)",
    "low_sodium": "Low Sodium",
    "vegetarian": "Vegetarian (no meat/fish, allow dairy/eggs)",
    "vegan": "Vegan (no animal products)",
    "gluten_free": "Gluten-Free (no wheat, barley, rye)",
    "dairy_free": "Dairy-Free",
    "keto": "Keto/Very Low Carb",
    "paleo": "Paleo",
    "heart_health": "Heart-Healthy (lower sat fat, more fiber)",
    "renal": "Renal-Friendly (watch potassium, phosphorus, sodium)",
    "weight_loss": "Weight Loss",
    "high_protein": "High Protein",
    "low_fat": "Low Fat",
    "low_carb": "Low Carb",
    "pregnancy": "Pregnancy-Safe",
    "children": "Kid-Friendly",
    "senior": "Senior-Friendly",
}

def _render_filters(filters: List[str]) -> str:
    if not filters: 
        return "None (general healthfulness)."
    out = []
    for f in filters:
        out.append(f"- { _LABELS.get(f, f) }")
    return "\n".join(out)

def health_check_text(ingredients: List[str], steps: List[str], filters: List[str], display_lang: str) -> str:
    """
    Build a concise per-condition set of swaps/warnings for the current recipe.
    Returns English by default; will translate to display_lang if not English and MT is available.
    """
    display_lang = safe_lang_code(display_lang)
    ing_block = "- " + "\n- ".join(ingredients or [])
    step_block = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps or []))
    filt_block = _render_filters(filters or [])

    # Ask the GGUF model to give replacements. Keep it strict & structured.
    user = (
        "You are a culinary diet assistant. Analyze the recipe and propose specific ingredient SWAPS "
        "and brief reasons PER health condition. Keep it practical and concise.\n\n"
        "Recipe – Ingredients:\n"
        f"{ing_block}\n\n"
        "Recipe – Steps:\n"
        f"{step_block}\n\n"
        "Health conditions requested:\n"
        f"{filt_block}\n\n"
        "Return STRICTLY in this structure (no preface):\n"
        "Health Check:\n"
        "- For <Condition 1>:\n"
        "  • Swap: <ingredient A> → <ingredient B or technique>\n"
        "    Reason: <short reason>\n"
        "  • Swap: ... (2–4 bullets max)\n"
        "- For <Condition 2>:\n"
        "  • Swap: ...\n"
        "Notes:\n"
        "- If a condition is not relevant to this recipe, say 'Looks OK – no changes'.\n"
        "- Prefer concrete swaps. For diabetes, call out sugar swaps (e.g., stevia/erythritol, less sugar, or fruit puree). "
        "For hypertension, flag soy sauce/salt and offer low-sodium options. For gluten-free, suggest GF tamari/pasta/flours. "
        "If the user could consider jaggery or honey, clarify they still add sugar and portion control matters.\n"
    )

    out = _llm().create_chat_completion(
        messages=[{"role": "system", "content": SYS},
                  {"role": "user", "content": user}],
        temperature=0.25, max_tokens=600, top_p=0.95, repeat_penalty=1.1
    )
    text_en = out["choices"][0]["message"]["content"].strip()

    # Optional: translate the full block for the UI language
    if display_lang != "en" and mt_enabled():
        try:
            text_loc = translate_text_m2m_cached(text_en, display_lang, src_code="en")
            return text_loc
        except Exception:
            pass
    return text_en
