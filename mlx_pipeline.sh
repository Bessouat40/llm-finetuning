#!/bin/bash
set -e

# You need to clone https://github.com/ml-explore/mlx-examples/tree/main repo
# You need to clone https://github.com/ggerganov/llama.cpp.git repo
MLX_PATH= # Path to mlx-examples/llms/mlx_lm
LLAMA_CPP_PATH= # Path to llama.cpp
MODEL_NAME= # Model you want to train locally
DATA_PATH= # Path to your data folder (need to have train.jsonl, valid.jsonl, test.jsonl inside)
LORA_FUSE_MODEL_PATH= # Path to save the fused model
GGUF_PATH= # Path to save the GGUF model
GGUF_MODEL= # Name of the GGUF model
ITERS= # Number of iterations for training

MODELFILE_PATH= # Path to the modelfile used to create the model in Ollama
OLLAMA_MODEL_NAME= # Name of the model in Ollama

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
