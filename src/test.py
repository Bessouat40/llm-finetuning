import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

torch.random.manual_seed(0)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3.5-mini-instruct", 
    device_map="mps", 
    torch_dtype="auto", 
    trust_remote_code=True, 
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")

prompt = (
    "<|system|>"
    "Tu es un coach sportif spécialisé en nutrition. "
    "Réponds à la question suivante de manière concise et précise.\n\n"
    "<|user|>"
    "Comment gérer ma nutrition pendant un trail de 20km et 2000m de dénivelé positif ?\n"
)

messages = [
    {"role": "user", "content": prompt},
]

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

output = pipe(messages, **generation_args)
print(output[0]['generated_text'])
