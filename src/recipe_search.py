import ast
from difflib import get_close_matches

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.paths import assets_dir, data_file  # ✅ use namespaced imports only

CSV_PATH = data_file("full_dataset.csv")

DF = None
VEC = None
X = None

def _safe_eval_list(x):
    try:
        v = ast.literal_eval(x) if isinstance(x, str) else (x or [])
        return [str(i) for i in (v if isinstance(v, (list, tuple)) else [])]
    except Exception:
        return []


from src.paths import assets_dir

def init_search_engine(csv_path=None):
    global DF, VEC, X
    csv_path = csv_path or (assets_dir() / "full_dataset.csv")
    DF = pd.read_csv(csv_path)

    ner_tokens = DF.get("NER", "").fillna("").apply(_safe_eval_list).apply(lambda xs: " ".join(xs))
    ingr_text  = DF.get("ingredients", "").fillna("").astype(str)
    titles     = DF.get("title", "").fillna("").astype(str)

    # Index title + full ingredients + NER (lowercased)
    DF["ingredient_str"] = (
        titles.str.lower() + " " +
        ingr_text.str.lower() + " " +
        ner_tokens.str.lower()
    )

    # Bigrams help on phrases; min_df trims ultra-rare noise
    VEC = TfidfVectorizer(
    ngram_range=(1, 2),      # ✅ Keep your original ngram setting
    min_df=5,                # ✅ Ignore terms that appear in fewer than 5 recipes
    max_df=0.9,              # ✅ Drop very common words (in more than 90% of recipes)
    max_features=5000,       # ✅ Limit vocab size to top 5000 terms
    stop_words="english"     # ✅ Remove common English stopwords
    )
    X = VEC.fit_transform(DF["ingredient_str"])

def normalize_ingredients(query_ingredients):
    synonym_map = {
        "gajar": "carrots",
        "badam": "almonds",
        "kaju": "cashews"
    }
    return [synonym_map.get(ing.lower(), ing.lower()) for ing in query_ingredients]

def search_recipes(query_ingredients, top_k=5):
    global DF, VEC, X
    if DF is None:
        init_search_engine()

    tokens = [str(t).lower().strip() for t in (query_ingredients or []) if str(t).strip()]
    query_str = " ".join(tokens)

    # Exact title shortcut stays
    exact_match = DF[DF["title"].str.lower() == query_str.lower()]
    if not exact_match.empty:
        return exact_match.head(top_k)[["title", "ingredients", "directions", "link"]].drop_duplicates(subset=["title"])

    if not query_str:
        return DF.head(top_k)[["title", "ingredients", "directions", "link"]].drop_duplicates(subset=["title"])

    qv = VEC.transform([query_str])
    sims = cosine_similarity(qv, X, dense_output=False).toarray().flatten()

    # Take a wider pool then dedupe titles
    idx = sims.argsort()[-max(top_k*3, 10):][::-1]
    out = DF.iloc[idx][["title", "ingredients", "directions", "link"]]
    return out.dropna(subset=["title"]).drop_duplicates(subset=["title"]).head(top_k)

def match_punjabi_recipe(transliterated: str, top_k: int = 3):
    from difflib import get_close_matches
    title_list = DF['title'].dropna().astype(str).str.lower().tolist()
    matches = get_close_matches(transliterated.lower(), title_list, n=top_k, cutoff=0.5)
    return matches

from difflib import get_close_matches

def get_recipe_by_title(title_query: str, top_k=5):
    global DF
    if DF is None:
        init_search_engine()

    titles = DF['title'].dropna().tolist()
    lowered_titles = [str(t).lower() for t in titles]

    matches = get_close_matches(title_query.strip().lower(), lowered_titles, n=top_k, cutoff=0.6)

    results = []
    for match in matches:
        row = DF[DF['title'].str.lower() == match]
        if not row.empty:
            r = row.iloc[0]
            results.append({
                "title": r["title"],
                "ingredients": r["ingredients"].split("\n"),
                "steps": r["directions"].split("\n"),
                "link": r.get("link", "")
            })

    return results if results else None