# **Phi3.5 Fine Tuning**

## **Automate Fine-Tuning and Loading Inside Ollama**

This repository provides a script to streamline the process of fine-tuning a Phi3.5 model and loading it directly into Ollama for testing.

### Steps

1. Modify the variables at the top of the `mlx_pipeline.sh` script to suit your dataset, model, and paths.
2. Ensure you have the required dataset files: `train.jsonl`, `valid.jsonl`, and `test.jsonl` in the correct format.
3. Customize the system prompt for your Ollama model in the `Modelfile` file to adjust its behavior.
4. Run the script to train the model, convert it to GGUF format, and load it into Ollama:

```bash
sh mlx_pipeline.sh
```

---

The following steps and scripts are part of a personal workflow that can be reused but may require adaptation to specific use cases or projects.

---

## **Dataset Preparation**

### 1. **Dataset Creation**

Parse PDFs and generate questions/answers for training:

```bash
python3 src/preprocessing.py
```

### 2. **Dataset Cleaning**

Clean the generated JSON dataset:

```bash
python3 src/clean_json.py
```

### 3. **Dataset Concatenation**

Concatenate multiple datasets into a unified format:

```bash
python3 src/concatenation.py
```

### 4. **Dataset Formatting**

Format the dataset to meet Phi3.5 requirements:

```bash
python3 src/format_to_phi3.5.py
```

---

## **Manual Fine-Tuning Pipeline**

If you prefer a step-by-step manual approach, you can use the following commands:

### 1. **Fine-Tuning**

Fine-tune your model using the provided training script:

```bash
python3 src/train.py
```

### 2. **Model Fusion**

Fuse the LoRA adapters with the base model:

```bash
python3 src/fuseModels.py
```

### 3. **Model Conversion to GGUF Format**

Convert the fused model into the GGUF format using `llama.cpp`:

```bash
python3 llama.cpp/convert_hf_to_gguf.py <fused model> --outfile <model name .gguf> --outtype <quantization>
```

---

## **Using MLX for Fine-Tuning**

Alternatively, you can fine-tune and convert your model using MLX:

```bash
python lora.py --model <model name> \
               --train \
               --iters 50
python fuse.py --model <model name>
python3 convert_hf_to_gguf.py <lora fused model> --outfile <model name .gguf> --outtype <quantization>
```

---

## **Key Notes**

- Ensure the paths to your datasets, models, and scripts are correctly configured in the respective files or script variables.
- Use the automated pipeline for a seamless experience, or the manual approach for more granular control.
- Customize your `Modelfile` to adapt the system prompt for your model's behavior in Ollama.

This pipeline simplifies the process of preparing, fine-tuning, and deploying Phi3.5 models while maintaining flexibility for customization.
