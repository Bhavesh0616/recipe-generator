from huggingface_hub import snapshot_download

# Choose ONE of the following model IDs:
# 1. Gemma-1.1-1B (instruction-tuned)
model_id = "google/gemma-1.1-1b-it"

# 2. Gemma-1.1-2B (under 5GB when quantized)
# model_id = "google/gemma-1.1-2b-it"

# 3. Gemma-1.1-7B (needs GPU + more RAM)
# model_id = "google/gemma-1.1-7b-it"

# Target directory to save the model
save_dir = "./gemma_model"

# Download model snapshot (weights, tokenizer, config)
snapshot_download(
    repo_id=model_id,
    local_dir=save_dir,
    local_dir_use_symlinks=False,
    resume_download=True  # useful for interrupted downloads
)

print(f"✅ Model downloaded to: {save_dir}")
