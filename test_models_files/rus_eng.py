# test_ru_translate.py

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Set path to your Facebook model directory
model_dir = r"F:\foodchat\translation_models"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, local_files_only=True)

# Function to translate English to Russian
def translate_to_russian(text):
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    with torch.no_grad():
        output_ids = model.generate(input_ids, num_beams=4, max_new_tokens=100)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Function to simulate transliteration fallback (just returns same string)
def transliterate_fallback(text, lang):
    if lang == "ru":
        return text  # In real logic, we’d have Cyrillic check
    return text  # placeholder

# === TEST CASES ===
english_input = "What can I cook with carrot and pasta?"
russian_title = "пельмени"

# Run translation
translated = translate_to_russian(english_input)
transliterated = transliterate_fallback(russian_title, "ru")

# Print results
print("📤 English:", english_input)
print("🌐 Translated to Russian:", translated)
print("✍️ Transliterated Russian Title:", transliterated)
