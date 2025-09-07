import pandas as pd
from difflib import get_close_matches

punjabi_to_roman_map = {
    # Vowels
    'ਅ': 'a', 'ਆ': 'aa', 'ਇ': 'i', 'ਈ': 'ee', 'ਉ': 'u', 'ਊ': 'oo', 'ੲ': 'i', 'ਊ': 'oo', 'ਏ': 'e', 'ਐ': 'ai', 'ਓ': 'o', 'ਔ': 'au',
    # Consonants
    'ਕ': 'k', 'ਖ': 'kh', 'ਗ': 'g', 'ਘ': 'gh', 'ਙ': 'ng',
    'ਚ': 'ch', 'ਛ': 'chh', 'ਜ': 'j', 'ਝ': 'jh', 'ਞ': 'ny',
    'ਟ': 't', 'ਠ': 'th', 'ਡ': 'd', 'ਢ': 'dh', 'ਣ': 'n',
    'ਤ': 't', 'ਥ': 'th', 'ਦ': 'd', 'ਧ': 'dh', 'ਨ': 'n',
    'ਪ': 'p', 'ਫ': 'ph', 'ਬ': 'b', 'ਭ': 'bh', 'ਮ': 'm',
    'ਯ': 'y', 'ਰ': 'r', 'ਲ': 'l', 'ਵ': 'v',
    'ਸ਼': 'sh', 'ਸ': 's', 'ਹ': 'h', 'ਲ਼': 'l',
    # Dependent vowel signs
    'ਾ': 'a', 'ਿ': 'i', 'ੀ': 'ee', 'ੁ': 'u', 'ੂ': 'oo',
    'ੇ': 'e', 'ੈ': 'ai', 'ੋ': 'o', 'ੌ': 'au',
    # Nasalization and special marks
    'ਂ': 'n', 'ਃ': 'h', 'ੱ': '', '': '',
    'ੜ': 'r',
    # Space and punctuation
    ' ': ' ', '।': '.', ',': ','
}

def transliterate_punjabi(text):
    result = []
    skip_next = False
    for i, ch in enumerate(text):
        if skip_next:
            skip_next = False
            continue

        # Combine dependent vowel sign if present
        if i + 1 < len(text) and text[i + 1] in ['ਾ', 'ਿ', 'ੀ', 'ੁ', 'ੂ', 'ੇ', 'ੈ', 'ੋ', 'ੌ']:
            base = punjabi_to_roman_map.get(ch, ch)
            vowel = punjabi_to_roman_map.get(text[i + 1], '')
            result.append(base + vowel)
            skip_next = True
        else:
            result.append(punjabi_to_roman_map.get(ch, ch))
    return ''.join(result)

def find_best_match(query, csv_path="full_dataset.csv"):
    df = pd.read_csv(csv_path)
    titles = df['title'].dropna().astype(str).tolist()

    # Lowercase query and titles for comparison
    query_lower = query.lower()
    titles_lower = [title.lower() for title in titles]

    # Get best matches
    matches = get_close_matches(query_lower, titles_lower, n=3, cutoff=0.5)

    # Map back to original titles
    best_matches = [titles[titles_lower.index(m)] for m in matches]

    return best_matches

if __name__ == "__main__":
    samples = [
        "ਆਲੂ ਪਰਾਂਠਾ", "ਸ਼ਾਨਦਾਰ Eggs", "ਮੱਖੀ ਚਿਕਨ", "ਦਾਲ ਮੱਖਣੀ", "ਪਨੀਰ ਬਟਰ ਮਸਾਲਾ"
    ]

    for s in samples:
        romanized = transliterate_punjabi(s)
        matches = find_best_match(romanized)
        print(f"Original: {s}")
        print(f"Romanized: {romanized}")
        print(f"Best Matches: {matches}\n")
