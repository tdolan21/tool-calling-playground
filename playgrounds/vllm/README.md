# vLLM Playground

This playground includes all options available for vLLM requests except for tool calling. the infrastructure is there but tool calling is still to experimental with vLLM.

This will be updated in the future once tool calling is available beyond the one branch where its implemented with mistral. It is now on the roadmap and should be implemented soon. 

To serve a model: 

```bash
vllm serve NousResearch/Meta-Llama-3-8B-Instruct --dtype auto --api-key token-abc123
```

Then to start the playground:

```bash
streamlit run playground.py
```