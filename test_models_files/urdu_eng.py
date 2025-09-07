from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

MODEL_PATH = r"F:\foodchat\translation_models"

def translate_text(text, src_lang, tgt_lang, model, tokenizer):
    tokenizer.src_lang = src_lang
    encoded = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id(tgt_lang), max_new_tokens=128)
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

def transliterate_urdu_to_roman(text):
    urdu_to_roman_map = {
        'ا': 'a', 'آ': 'aa', 'ب': 'b', 'پ': 'p', 'ت': 't', 'ٹ': 't', 'ث': 's', 'ج': 'j', 'چ': 'ch',
        'ح': 'h', 'خ': 'kh', 'د': 'd', 'ڈ': 'd', 'ر': 'r', 'ڑ': 'r', 'ز': 'z', 'ژ': 'zh',
        'س': 's', 'ش': 'sh', 'ص': 's', 'ض': 'z', 'ط': 't', 'ظ': 'z', 'ع': 'a', 'غ': 'gh',
        'ف': 'f', 'ق': 'q', 'ک': 'k', 'گ': 'g', 'ل': 'l', 'م': 'm', 'ن': 'n', 'ں': 'n',
        'و': 'oo', 'ہ': 'h', 'ھ': 'h', 'ء': "'", 'ی': 'ee', 'ے': 'e', 'ں': 'n',
        '‍': '', 'ٔ': '', 'ٰ': 'a', '۔': '.', '،': ',', ' ': ' '
    }
    return ''.join(urdu_to_roman_map.get(ch, ch) for ch in text)

if __name__ == "__main__":
    model = M2M100ForConditionalGeneration.from_pretrained(MODEL_PATH, local_files_only=True)
    tokenizer = M2M100Tokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

    urdu_text = "چکن گوشٹ"

    # Urdu (ur) -> Hindi (hi)
    hindi_text = translate_text(urdu_text, "ur", "hi", model, tokenizer)
    print("Hindi Text:", hindi_text)

    # Hindi (hi) -> English (en)
    english_text = translate_text(hindi_text, "hi", "en", model, tokenizer)
    print("English Text:", english_text)

    # Hindi -> Urdu (ur)
    urdu_translated = translate_text(hindi_text, "hi", "ur", model, tokenizer)
    print("Urdu Translated Text:", urdu_translated)

    # Urdu transliterated to Roman Urdu (English script)
    roman_urdu = transliterate_urdu_to_roman(urdu_translated)
    print("Romanized Urdu:", roman_urdu)
