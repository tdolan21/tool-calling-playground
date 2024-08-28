# LoRAX Playground

I have a comprehensive write up about LoRAX, what it is, how you use it and why you use it available on [huggingface](https://huggingface.co/blog/macadeliccc/deploy-hundreds-of-models-on-one-gpu)

It explains how this application works, how to deploy LoRAX and test it for production use. 

## Prerequisites

+ Docker
+ Nvidia Container Toolkit

To start a local instance of LoRAX, you can use this command: 

```
./launch_lorax.sh
```

Then you can run: 

```python
streamlit run playground.py
```
