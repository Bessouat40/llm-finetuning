import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Détecter si MPS est disponible (pour Mac M1/M2)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Chemins vers le modèle et le tokenizer
base_model_name = "unsloth/Phi-3.5-mini-instruct"
fine_tuned_model_dir = "lora_model"  # Chemin vers le modèle fine-tuné LoRA

# Charger le tokenizer
tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_dir, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token  # S'assurer que le token de padding est défini

# Charger le modèle de base
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16 if torch.backends.mps.is_available() else torch.float32,
).to(device)

# Charger les poids LoRA
model = PeftModel.from_pretrained(base_model, fine_tuned_model_dir)

# Passer le modèle sur le bon dispositif
model = model.to(device)

# Fonction d'inférence
def generate_response(prompt, max_length=200):
    """
    Génère une réponse à partir d'un prompt donné.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.5,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

prompt = (
    "Tu es un coach sportif spécialisé en nutrition. "
    "Réponds à la question suivante de manière concise et précise.\n\n"
    "Question : Code une fonction python\n"
    "Réponse :"
)
response = generate_response(prompt)
print("Prompt:", prompt)
print("Réponse:", response)