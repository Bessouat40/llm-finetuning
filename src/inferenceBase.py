import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Détection de l'appareil (CPU ou MPS pour Mac M1/M2)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Charger le modèle de base
model_name = "unsloth/Phi-3.5-mini-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.backends.mps.is_available() else torch.float32,
).to(device)

# Fonction d'inférence
def generate_response(prompt, max_length=200):
    """
    Génère une réponse en utilisant le modèle de base.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=0.7,  # Modère la diversité
        top_k=50,         # Limite aux 50 tokens les plus probables
        top_p=0.95,       # Filtrage par probabilité cumulée
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True    # Active le sampling
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Tester avec un prompt
prompt = ("You are a sports coach specialized in nutrition, and you answer the questions people ask you."
"Question: What are the benefits of fasting training for athletes?")
response = generate_response(prompt)
print("Prompt:", prompt)
print("Réponse:", response)
