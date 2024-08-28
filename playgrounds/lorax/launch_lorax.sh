#!/bin/bash

# Define variables
MODEL="NousResearch/Hermes-2-Theta-Llama-3-8B"
VOLUME="$PWD/data"
HUGGING_FACE_HUB_TOKEN="your-hf-token"

# Export the HuggingFace token
export HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN

# Run the Docker container with the HuggingFace token
docker run --gpus all --shm-size 1g -p 8080:80 -v $VOLUME:/data \
    -e HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN \
    ghcr.io/predibase/lorax:main --model-id $MODEL \
    --adapter-memory-fraction 0.4
    

    # ## optional args
    # --quantize eetq
    # --quantize hqq-2bit # 2,3, 4 available
    # --quantize awq
    
