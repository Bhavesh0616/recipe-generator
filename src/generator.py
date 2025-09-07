import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")



from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import List

from src.paths import assets_dir  # single source of truth

GPT_MODEL_DIR = str(assets_dir() / "gpt2")  # packaged under data/gpt2 when frozen

_chat_pipe = None

def get_chat_pipeline():
    global _chat_pipe
    if _chat_pipe is None:
        tok = AutoTokenizer.from_pretrained(GPT_MODEL_DIR, local_files_only=True)
        mdl = AutoModelForCausalLM.from_pretrained(GPT_MODEL_DIR, local_files_only=True)
        _chat_pipe = pipeline("text-generation", model=mdl, tokenizer=tok, device=-1)
    return _chat_pipe

def generate_recipes(prompt: str) -> List[str]:
    try:
        pipe = get_chat_pipeline()
        out = pipe(prompt, max_new_tokens=100, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        return [out[0]["generated_text"]]
    except Exception as e:
        print(f"[generator] error: {e}")
        return ["I couldn't generate a recipe. Please try again with different ingredients."]


def extract_keywords(text: str) -> List[str]:
    try:
        print(f"🔍 Extracting keywords using GPT for input: {text}")
        prompt = f"Extract ingredients or keywords only from this text:\n{text}\nKeywords:"
        output = generate_recipes(prompt)[0]
        keywords = [w.strip().strip(",") for w in output.lower().split() if len(w.strip()) > 2]
        return list(set(keywords))
    except Exception as e:
        print(f"[extract_keywords] fail: {e}")
        return []

def extract_recipe_title(text: str) -> str:
    try:
        print(f"📘 Extracting recipe title from: {text}")
        prompt = f"What is the recipe being asked about in this question?\n{text}\nAnswer with only the recipe name:"
        output = generate_recipes(prompt)[0]
        lines = output.strip().split("\n")
        for line in lines:
            if line and not line.lower().startswith(("question", "answer")):
                return line.strip().strip(".")
        return output.strip()
    except Exception as e:
        print(f"[extract_recipe_title] fail: {e}")
        return ""
    

import re

def extract_keywords_simple(text: str):
    stop = {
        "a","about","above","after","again","against","all","am","an","and","any","are","aren","as","at",
        "be","because","been","before","being","below","between","both","but","by",
        "can","could",
        "did","do","does","doing","don","down","during",
        "each",
        "few","for","from","further",
        "had","has","have","having","he","her","here","hers","herself","him","himself","his","how",
        "i","if","in","into","is","it","its","itself",
        "just",
        "ll","m","ma","me","more","most","my","myself",
        "no","nor","not","now",
        "of","off","on","once","only","or","other","our","ours","ourselves","out","over","own",
        "re",
        "s","same","she","should","so","some","such",
        "t","than","that","the","their","theirs","them","themselves","then","there","these","they","this","those","through","to","too",
        "under","until","up",
        "very",
        "was","we","were","what","when","where","which","while","who","whom","why","will","with","won","would",
        "y","you","your","yours","yourself","yourselves",
        # domain-specific fillers you don’t want
        "make","recipe","recipes","today","please","suggest","like","dish","food","cook","cooking"
    }

    words = re.findall(r"[a-zA-Z]+", (text or "").lower())
    return [w for w in words if w not in stop]
