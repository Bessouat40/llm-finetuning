from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Chemin vers le modèle de base et le modèle LoRA
base_model_name = "unsloth/Phi-3.5-mini-instruct"
lora_model_dir = "lora_model"
output_model_dir = "fused_model"

# Charger le modèle de base
model = AutoModelForCausalLM.from_pretrained(base_model_name)

# Charger le modèle LoRA
model = PeftModel.from_pretrained(model, lora_model_dir)

# Fusionner les poids LoRA dans le modèle principal
model = model.merge_and_unload()

# Sauvegarder le modèle fusionné
model.save_pretrained(output_model_dir)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.save_pretrained(output_model_dir)

print(f"Modèle fusionné sauvegardé dans : {output_model_dir}")
