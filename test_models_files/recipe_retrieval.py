import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load CSV data
df = pd.read_csv("full_dataset.csv")

# Combine ingredient keywords into a single string per recipe
df['ingredient_str'] = df['NER'].apply(lambda x: ' '.join(eval(x)))

# Vectorize ingredient strings
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['ingredient_str'])

def search_recipes(query_ingredients, top_k=20):
    query_str = ' '.join(query_ingredients)
    query_vec = vectorizer.transform([query_str])
    sim_scores = cosine_similarity(query_vec, X).flatten()
    top_indices = sim_scores.argsort()[-top_k:][::-1]
    return df.iloc[top_indices][['title', 'ingredients', 'directions', 'link']]

synonym_map = {
    "gajar": "carrots",
    "badam": "almonds",
    "kaju": "cashews"
}

def normalize_ingredients(query_ingredients):
    return [synonym_map.get(ing.lower(), ing.lower()) for ing in query_ingredients]

query = ['kaju', 'badam', 'gajar']
normalized_query = normalize_ingredients(query)
results = search_recipes(normalized_query)
print(results)
