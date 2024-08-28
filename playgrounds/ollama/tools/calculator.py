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