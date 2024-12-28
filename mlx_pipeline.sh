#!/bin/bash
set -e

MLX_PATH="/Users/labess40/dev/mlx-examples/llms/mlx_lm"
LLAMA_CPP_PATH="/Users/labess40/dev/llama.cpp"
MODEL_NAME="Phi-3-mini-4k-instruct"
DATA_PATH="/Users/labess40/dev/mlx-examples/lora/data"
LORA_FUSE_MODEL_PATH="/Users/labess40/dev/phi-fine-tunning/lora_fused_model"
GGUF_PATH="/Users/labess40/dev/phi-fine-tunning/gguf"
GGUF_MODEL="Phi-3-mini-4k-instruct_nutrition.gguf"
ITERS=50

MODELFILE_PATH="/Users/labess40/dev/phi-fine-tunning/Modelfile"
OLLAMA_MODEL_NAME="phi-nutrition"

export PYTHONPATH="$(dirname "$MLX_PATH"):$PYTHONPATH"

echo "Step 1: Training the model with LoRA..."
python3 -m mlx_lm.lora --model "$MODEL_NAME" \
                       --data "$DATA_PATH"\
                       --train \
                       --iters "$ITERS"

echo "Step 2: Fusing the LoRA model..."
python3 -m mlx_lm.fuse --model "$MODEL_NAME" \
                       --save-path "$LORA_FUSE_MODEL_PATH"

echo "Step 3: Converting the model to GGUF format using llama.cpp..."
python3 "$LLAMA_CPP_PATH/convert_hf_to_gguf.py" "$LORA_FUSE_MODEL_PATH" \
                                               --outfile "$GGUF_PATH/$GGUF_MODEL" \
                                               --outtype q8_0

echo "Step 4: Updating the Modelfile..."
sed -i.bak "s|^FROM .*|FROM $GGUF_PATH/$GGUF_MODEL|" "$MODELFILE_PATH"
echo "Updated Modelfile:"
cat "$MODELFILE_PATH"

echo "Step 5: Creating the model in Ollama..."
ollama create "$OLLAMA_MODEL_NAME" -f "$MODELFILE_PATH"

echo "All steps completed successfully!"
