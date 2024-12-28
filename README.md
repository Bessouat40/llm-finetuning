# Phi3.5 fine tuning

## Dataset Creation

PDF parsing and questions/answers creation :

```bash
python3 preprocessing.py
```

## Dataset Cleaning

```bash
python3 clean_json.py
```

## Dataset concatenation

```bash
python3 concatenation.py
```

## Dataset formatting to phi3.5 requirements

```bash
python3 format_to_phi3.5.py
```

## Fine tuning

```bash
python3 train.py
```

## Models Fusion

```bash
python3 fuseModels.py
```

## Model conversion to gguf format

Convert to gguf format using llama.cpp :

```bash
python3 convert_hf_to_gguf.py /Users/labess40/dev/phi-fine-tunning/fused_model --outfile /Users/labess40/dev/phi-fine-tunning/phi3.5_nutrition.gguf --outtype q8_0
```

python3 convert_hf_to_gguf.py /Users/labess40/dev/mlx-examples/lora/lora_fused_model --outfile /Users/labess40/dev/mlx-examples/lora/Mistral_7B_v01_nutrition.gguf --outtype q4_0

## Using MLX

```bash
python lora.py --model <model name> \
               --train \
               --iters 50
python fuse.py --model <model name>
python3 convert_hf_to_gguf.py /Users/labess40/dev/mlx-examples/lora/lora_fused_model --outfile /Users/labess40/dev/mlx-examples/lora/Mistral_7B_v01_nutrition.gguf --outtype q8_0
```
