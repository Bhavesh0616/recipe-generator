# test_korean.py

import re
import pandas as pd
from difflib import get_close_matches
from hangul_romanize import Transliter
from hangul_romanize.rule import academic

# === Load recipe titles from your dataset ===
CSV_PATH = r"f:/foodchat/full_dataset.csv"  # 🔁 Change this path to your real CSV
DF = pd.read_csv(CSV_PATH)
ALL_TITLES = DF['title'].astype(str).tolist()

# === Korean → Romanized ===
def korean_to_romanized(text: str) -> str:
    transliter = Transliter(academic)
    romanized = transliter.translit(text)
    return romanized

# === Get top-N closest matches from CSV titles ===
def find_closest_titles(query: str, all_titles, n=5, cutoff=0.6):
    return get_close_matches(query.lower(), [t.lower() for t in all_titles], n=n, cutoff=cutoff)

# === Test ===
if __name__ == "__main__":
    korean_words = [
        "김밥",     # gimbap
        "비빔밥",   # bibimbap
        "불고기",   # bulgogi
        "떡볶이",   # tteokbokki
        "갈비"      # galbi
    ]

    for word in korean_words:
        romanized = korean_to_romanized(word)
        closest = find_closest_titles(romanized, ALL_TITLES)
        print(f"\n🇰🇷 {word} → 🇺🇸 {romanized}")
        print("Top matches from CSV:")
        for i, match in enumerate(closest, 1):
            print(f"{i}. {match}")
