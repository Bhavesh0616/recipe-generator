from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch

# === Set your model path ===
MODEL_PATH = "F:/foodchat/translation_models"
SRC_LANG = "en"
TGT_LANG = "fr"

# === Load the model and tokenizer ===
print("🔄 Loading model...")
tokenizer = M2M100Tokenizer.from_pretrained(MODEL_PATH)
model = M2M100ForConditionalGeneration.from_pretrained(MODEL_PATH)

# === Set source language ===
tokenizer.src_lang = SRC_LANG

# === Text to translate ===
ingredients = """- 2 1/2 lb. broiler-fryer, quartered, 2 Tbsp. butter or margarine, 8 small white onions, peeled, 1 clove garlic, crushed, 1 tsp. salt, 1/8 tsp. pepper, 1 c. canned condensed chicken broth, undiluted, chopped parsley, 6 slices bacon, diced, 8 whole mushrooms, 2/3 c. sliced green onion, 2 1/2 Tbsp. flour, 1/4 tsp. dried thyme leaves, 2 c. Burgundy, 8 small new potatoes scrubbed, 3 whole chicken breasts, boned, halved and skinned, salt, pepper and garlic, 1/4 c. flour, 1/4 c. butter, 1 Tbsp. chopped shallots or onions, 1/2 c. dry white Bordeaux wine, 1/2 tsp. dried tarragon, 3/4 c. chicken broth, 1/4 c. heavy cream."""

steps = """1. Sprinkle chicken with salt, pepper and garlic., Dredge chicken in flour., Reserve remaining flour. Saute chicken in 3 tablespoons of butter to brown., Transfer to a 13 x 9-inch pan."""

# === Define a translate function ===
def translate(text: str, target_lang: str) -> str:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.get_lang_id(target_lang),
            max_new_tokens=512
        )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

# === Translate ===
print("\n🌍 Translating Ingredients...")
translated_ingredients = translate(ingredients, TGT_LANG)
print("\n🍽️ French Ingredients:\n", translated_ingredients)

print("\n🌍 Translating Steps...")
translated_steps = translate(steps, TGT_LANG)
print("\n📋 French Steps:\n", translated_steps)
