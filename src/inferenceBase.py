import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

model_name = "unsloth/Phi-3.5-mini-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.backends.mps.is_available() else torch.float32,
).to(device)

def generate_response(prompt, max_length=200):
    """
    Génère une réponse en utilisant le modèle de base.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True  
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

prompt = ("You are a sports coach specialized in nutrition, and you answer the questions people ask you."
"Question: What are the benefits of fasting training for athletes?")
response = generate_response(prompt)
print("Prompt:", prompt)
print("Réponse:", response)
