from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# Path to your local model directory
MODEL_PATH = r"F:\foodchat\translation_models"

# Load model & tokenizer
tokenizer = M2M100Tokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = M2M100ForConditionalGeneration.from_pretrained(MODEL_PATH, local_files_only=True)

# Define source and target languages
tokenizer.src_lang = "en"
target_lang = "ja"  # Japanese

# Example English recipe sentence
english_text = "Boil the rice in water for 10 minutes, then add salt and vinegar."

# Tokenize input
encoded = tokenizer(english_text, return_tensors="pt")

# Generate translated output
generated_tokens = model.generate(
    **encoded,
    forced_bos_token_id=tokenizer.get_lang_id(target_lang)
)

# Decode and print translation
japanese_translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
print("English:", english_text)
print("Japanese:", japanese_translation)
