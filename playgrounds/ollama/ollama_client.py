from typing import List, Dict, Optional
import ollama

class OllamaClient:
    def __init__(self, model: str):
        self.client = ollama.AsyncClient()
        self.model = model

    async def chat(self, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None, options: Optional[Dict] = None) -> Dict[str, any]:
        return await self.client.chat(model=self.model, messages=messages, tools=tools, options=options)