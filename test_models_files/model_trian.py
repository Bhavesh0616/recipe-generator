import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
import torch

# === Load and prepare dataset ===
df = pd.read_csv("full_dataset.csv")

# Rename columns if needed
df = df.rename(columns={
    "ingredients": "input_text",
    "directions": "target_text"
})

# Create Hugging Face dataset
dataset = Dataset.from_pandas(df[["input_text", "target_text"]])

# === Load tokenizer from base model ===
tokenizer = AutoTokenizer.from_pretrained("flax-community/t5-recipe-generation")

# Preprocessing function
def preprocess_function(examples):
    inputs = tokenizer(examples["input_text"], truncation=True, padding="max_length", max_length=128)
    targets = tokenizer(examples["target_text"], truncation=True, padding="max_length", max_length=256)
    inputs["labels"] = targets["input_ids"]
    return inputs

# Tokenize dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# === Load model from local path ===
model = AutoModelForSeq2SeqLM.from_pretrained("models/recipe_generator")

# === Training arguments ===
training_args = Seq2SeqTrainingArguments(
    output_dir="./models/recipe_generator_finetuned",
    evaluation_strategy="steps",
    save_strategy="steps",
    logging_steps=50,
    save_steps=200,
    save_total_limit=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    learning_rate=5e-5,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
)

# === Data collator ===
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# === Initialize Trainer ===
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset.select(range(100)),  # optional small eval set
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# === Start fine-tuning ===
trainer.train()

# === Save fine-tuned model ===
trainer.save_model("models/recipe_generator_finetuned")
tokenizer.save_pretrained("models/recipe_generator_finetuned")
