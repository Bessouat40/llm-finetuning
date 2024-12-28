import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer
import json

# Détecter si MPS est disponible sur Mac M1
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
torch.backends.mps.allow_tf32 = False

model_name = "microsoft/Phi-3.5-mini-instruct"

# Chargement du dataset JSON
dataset = load_dataset("json", data_files="./data/trainData_HuggingFace.json")["train"]

def preprocess_function(examples):
    texts = []
    for q, a in zip(examples["question"], examples["answer"]):
        if q and a:
            formatted_text = (
                "<|user|>\n"
                f"{q.strip()}<|end|>\n"
                "<|assistant|>\n"
                f"{a.strip()}<|end|>"
            )
            texts.append({"text": formatted_text})
    return texts

processed_data = preprocess_function(dataset)
processed_dataset = Dataset.from_list(processed_data)

# Add data validation
print("Total dataset size:", len(dataset))
for data in processed_data[:3]:
    print('data : ', data)

output_file = "./nutrition_finetuning.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(processed_data, f, indent=4, ensure_ascii=False)

print(f"Data saved to {output_file}")

# Charger le tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token  # S'assurer d'avoir un token de padding

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map=None
).to(device)

# Configuration LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear"
)

for name, param in model.named_parameters():
    if "lora" in name:
        torch.nn.init.normal_(param, mean=0.0, std=0.02)

training_args = TrainingArguments(
    output_dir="outputs",
    bf16=True,  # Adjust based on your hardware
    fp16=False,   # Enable for MPS
    per_device_train_batch_size=2,  # Increase if possible
    gradient_accumulation_steps=2,
    num_train_epochs=1,  # Try more epochs
    max_steps=-1,
    optim="adamw_torch",
    lr_scheduler_type="linear",
    learning_rate=1e-5,  # Increased learning rate
    logging_steps=10,
    save_steps=100,
    report_to="none",
    max_grad_norm=0.1,
    seed=42,
    gradient_checkpointing=False,
    warmup_ratio=0.1,
)

# Création du trainer SFT
trainer = SFTTrainer(
    model=model,
    args=training_args,
    peft_config=lora_config,
    train_dataset=processed_dataset,
    dataset_text_field="text",
    tokenizer=tokenizer,
    max_seq_length=512,
    packing=False
)

# Before training
print("Initial Model State:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}: mean={param.data.mean().item()}, std={param.data.std().item()}")

# Lancement de l'entraînement
with torch.autograd.detect_anomaly():
    trainer.train()
# trainer.train()

# Sauvegarde du modèle LoRA et du tokenizer
trainer.save_model("lora_model")
tokenizer.save_pretrained("lora_model")
