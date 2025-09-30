# nutrition.py
from __future__ import annotations
from typing import List
from recipe import _llm, SYS  # reuse same model + system

def nutrition_from_ingredients(ingredients: List[str]) -> str:
    if not ingredients:
        return "No ingredients provided."
    user = (
        "Estimate nutrition per serving for the following ingredient list. "
        "Return a compact text block with calories, macros, and 3-5 micronutrients where possible. "
        "If uncertain, give reasonable estimates with '~'.\n\n"
        "Ingredients:\n- " + "\n- ".join(ingredients)
    )
    out = _llm().create_chat_completion(
        messages=[{"role":"system","content": SYS},
                  {"role":"user","content": user}],
        temperature=0.2, max_tokens=300, top_p=0.95, repeat_penalty=1.1
    )
    return out["choices"][0]["message"]["content"].strip()
