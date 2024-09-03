# Ollama Playground with tool calling

Open source playground similar to OpenGPT-4o project. This repo serves as a comprehensive tool calling example.

Before installing, this project will take ~16GB of VRAM to run everything. The litserve server is quite efficient. The ollama models and litserve models do not reserve the GPU space so the requests can be made in quick succession with low overhead compute cost. 

Llama3.1: 7.4GB peak VRAM Usage


```bash
git clone https://github.com/tdolan21/tool-calling-playground
cd playgrounds/ollama
```
For Linux (No Flux Server):

```bash
chmod +x install_linux.sh
./install_linux.sh
streamlit run playground.py
```
If you want to launch the application with access to flux, then you can use this command:
```bash
chmod +x run.sh
./run.sh
```

For Windows:

Make sure you have ollama ≤ v0.3.6

```bash
pip install ollama streamlit playwright
playwright install
```

## Add a new tool 

To add a new tool, define it in tools/your_tool.py:

```python
import math

class CalculatorTool:
    def __init__(self):
        self.name = "calculator"
        self.description = "Performs basic mathematical operations and some advanced functions."
        self.parameters = {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate."
                }
            },
            "required": ["expression"]
        }
        self.instructions = """Use this calculator tool to perform mathematical operations. 
The tool can handle basic arithmetic (addition, subtraction, multiplication, division) as well as 
square root (sqrt) and power (pow) operations. Provide the expression as a string, and the tool will evaluate it."""

    async def execute(self, expression: str) -> dict:
        try:
            # Use Python's eval function to calculate the result
            # This is potentially dangerous in a production environment and should be used with caution
            result = eval(expression, {"__builtins__": None}, {"sqrt": math.sqrt, "pow": math.pow})
            return {"result": result}
        except Exception as e:
            return {"error": str(e)}
```

If you want the tool to utilize multistep behavior and have the model use the function call response to create a new answer, then the instructions for that process must be defined in the tool class. If no instructions are provided then the model will give the function call response. 

Then to give the model use of the tool and have it populate in the UI, just add a line to the tool manager:
```python
if 'tool_manager' not in st.session_state:
        st.session_state.tool_manager = ToolManager()
        # Initialize tools here
        st.session_state.tool_manager.register_tool("web_scraper", WebScraper())
```
in `playground.py` like this: 

```python
        st.session_state.tool_manager.register_tool("calculator", Calculator())
```

This allows essentially any function to be used by a language model. Once a function is registered the instructions will be editable in UI if you initiate the function. 

Functions are all designed to be dedicated so if you initiate the function it will try to use it. I have had much less success with the model choosing to use the functions reliably and staying in a standard chat simply on its own knowledge.