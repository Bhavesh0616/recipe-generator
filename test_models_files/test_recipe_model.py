# f:\foodchat\mistral_recipe_gen.py
import os, json
from llama_cpp import Llama
import sqlite3

# Keep caches off C:
os.environ["HF_HOME"] = r"F:\foodchat"

MODEL_PATH = "gpt2-large"

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,        # context window; increase if your build supports larger
    n_threads=8,       # set to your CPU cores
    n_gpu_layers=0     # CPU-only; change if you compiled with CUDA/Metal
)

SYSTEM_PROMPT = """You are ChefMistral, an expert global recipe generator.
Return strictly VALID JSON with keys:
- title (string)
- cuisine (string)
- servings (integer)
- ingredients (array of strings: 'amount unit item' or just 'item')
- instructions (array of step strings)
- notes (string)
Be concise, authentic to the cuisine, and realistic with measures.
"""

def generate_recipe(query, cuisine=None, servings=None, context=None):
    parts = []
    if cuisine:  parts.append(f"Cuisine: {cuisine}")
    if servings: parts.append(f"Servings: {servings}")
    parts.append(f"Request: {query}")
    if context:  parts.append("Context:\n" + context[:2000])  # optional retrieved refs
    user = "\n".join(parts)

    out = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user},
        ],
        temperature=0.7,
        top_p=0.9,
        max_tokens=800,
        repeat_penalty=1.1,
    )
    text = out["choices"][0]["message"]["content"].strip()

    # Optional: try to extract/parse JSON safely
    try:
        # if model wrapped JSON in code fences
        if "```" in text:
            text = text.split("```")[1]
            if text.lower().startswith("json"):
                text = text.split("\n", 1)[1]
        return json.loads(text)
    except Exception:
        return text  # fall back to raw text if not valid JSON

if __name__ == "__main__":
    print("=== Example 1: Indian dessert ===")
    print(generate_recipe(
        "Make a dessert with gajar (carrot), badam (almonds), ghee, and kaju (cashews).",
        cuisine="Indian", servings=4
    ))

    print("\n=== Example 2: Italian pasta ===")
    print(generate_recipe(
        "Simple weeknight pasta with tomato, basil, and olive oil.",
        cuisine="Italian", servings=2
    ))

    print("\n=== Example 3: Japanese soup ===")
    print(generate_recipe(
        "Light miso soup with tofu and wakame.",
        cuisine="Japanese", servings=3
    ))
