import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

MODEL_PATH = "./distilgpt2_model"

def generate_nutrition(ingredients_list):
    # Load tokenizer and model from local dir
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

    # Use text-generation pipeline
    generator = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )

    prompt = (
        "Given the ingredients below, provide the nutrition facts per serving in JSON format. "
        "Include calories, protein_g, fat_g, carbs_g, fiber_g with numeric values only.\n\n"
        "Ingredients:\n" + "\n".join(f"- {item}" for item in ingredients_list) +
        "\n\nExample output:\n{\"calories\": 200, \"protein_g\": 10, \"fat_g\": 5, \"carbs_g\": 30, \"fiber_g\": 4}\n\n"
        "Nutrition facts JSON:"
)



    outputs = generator(
        prompt,
        max_new_tokens=150,
        do_sample=False,
        temperature=0.3,
        top_p=0.9
    )


    result = outputs[0]["generated_text"]
    print("Prompt + generated nutrition text:\n")
    print(result)

if __name__ == "__main__":
    test_ingredients = ["egg", "bread", "butter"]
    generate_nutrition(test_ingredients)
