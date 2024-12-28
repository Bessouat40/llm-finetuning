from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_name = "unsloth/Phi-3.5-mini-instruct"
lora_model_dir = "lora_model"
output_model_dir = "fused_model"

model = AutoModelForCausalLM.from_pretrained(base_model_name)

model = PeftModel.from_pretrained(model, lora_model_dir)

model = model.merge_and_unload()

model.save_pretrained(output_model_dir)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.save_pretrained(output_model_dir)

print(f"Modèle fusionné sauvegardé dans : {output_model_dir}")
