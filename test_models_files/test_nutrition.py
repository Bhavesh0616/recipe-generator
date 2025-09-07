from transformers import pipeline, set_seed

# Path to your local GPT2 model directory
MODEL_DIR = "F:\foodchat\gpt2"

def generate_health_advice(recipe_title, ingredients, steps, health_condition="Diabetes"):
    # Initialize GPT2 pipeline for text generation
    gen = pipeline("text-generation", model=MODEL_DIR, tokenizer=MODEL_DIR)

    # Seed for reproducibility
    set_seed(42)

    # Prepare prompt for GPT2
    prompt = f"""
You are a helpful nutrition assistant.
Given the following recipe details:

Title: {recipe_title}

Ingredients:
{chr(10).join(['- ' + i for i in ingredients])}

Preparation Steps:
{chr(10).join(['- ' + s for s in steps])}

Please provide advice specifically for a patient with {health_condition}.
List ingredients or steps that should be avoided or limited due to the condition, along with brief reasons.
"""
    # Generate the advice text from GPT2
    output = gen(
        prompt,
        max_length=200,
        num_return_sequences=1,
        do_sample=False
    )

    advice_text = output[0]["generated_text"][len(prompt):].strip()
    return advice_text

if __name__ == "__main__":
    # Example hardcoded recipe details
    title = "Sweet and Sour Pork"
    ingredients = [
        "1 cup sugar",
        "1/2 cup white vinegar",
        "2 tablespoons soy sauce",
        "Pork slices",
        "Vegetable oil",
        "Pineapple chunks",
        "Bell peppers",
        "Cornstarch"
    ]
    steps = [
        "Mix sugar, vinegar, soy sauce, and cornstarch to prepare sauce.",
        "Fry pork slices until golden brown.",
        "Add bell peppers and pineapple chunks.",
        "Pour sauce over and cook until thickened."
    ]

    health_advice = generate_health_advice(title, ingredients, steps, health_condition="Diabetes")
    print("Health Advice for Diabetic Patient:")
    print(health_advice)
