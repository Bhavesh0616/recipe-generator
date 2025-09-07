from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration

model_name = "facebook/m2m100_418M"
local_dir = r"F:\foodchat\translation_models"

# Download and save tokenizer
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(local_dir)

# Download and save model
model = M2M100ForConditionalGeneration.from_pretrained(model_name)
model.save_pretrained(local_dir)

print(f"Model and tokenizer for {model_name} downloaded and saved to {local_dir}")
