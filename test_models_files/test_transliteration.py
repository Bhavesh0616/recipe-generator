from indic_transliteration.sanscript import transliterate

# Convert phonetic Hindi written in English (ITRANS style) into Devanagari
input_text = "शिमला मिर्च और चावल"
output = transliterate(input_text, "itrans", "devanagari")
print(output)  # → शिमला मिर्च और चावल
