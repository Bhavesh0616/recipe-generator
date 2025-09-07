from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2-large"
local_dir = r"F:/gpt2-large"  # Change path as desired

# Download and cache the tokenizer and model locally in specified directory
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=local_dir)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=local_dir)

print(f"{model_name} model and tokenizer have been downloaded to {local_dir}")
