#/bin/bash

curl -fsSL https://ollama.com/install.sh | sh

# Install playwright
pip install playwright
playwright install

# Install the rest of the deps
pip install -r requiremnets.txt

ollama pull llama3.1

