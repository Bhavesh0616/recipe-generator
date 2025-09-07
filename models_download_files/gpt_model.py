from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "gpt2"
save_dir = r"F:/foodchat/gpt2"  # Update path if needed

# Load tokenizer and model directly from Hugging Face
tok = AutoTokenizer.from_pretrained(model_id)
mdl = AutoModelForCausalLM.from_pretrained(model_id)

# Save tokenizer and model locally
tok.save_pretrained(save_dir)
mdl.save_pretrained(save_dir)
