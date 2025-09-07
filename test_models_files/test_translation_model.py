from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

def translate(text, src_lang="en", tgt_lang="hi"):
    # Load tokenizer + model (will use your local cache if already downloaded)
    tok = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    mdl = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")

    # set source language
    tok.src_lang = src_lang
    encoded = tok(text, return_tensors="pt")

    # generate translation
    gen_tokens = mdl.generate(**encoded, forced_bos_token_id=tok.get_lang_id(tgt_lang), max_new_tokens=200)
    return tok.batch_decode(gen_tokens, skip_special_tokens=True)[0]

if __name__ == "__main__":
    # Test cases
    english_text = "Capsicum, tomato, cabbage and rice"
    hindi_text = translate(english_text, src_lang="en", tgt_lang="hi")
    french_text = translate(english_text, src_lang="en", tgt_lang="fr")

    print("Original (EN):", english_text)
    print("Hindi:", hindi_text)
    print("French:", french_text)
