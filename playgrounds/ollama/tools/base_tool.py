from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseTool(ABC):
    def __init__(self, name: str, description: str, instructions: str = ""):
        self.name = name
        self.description = description
        self.instructions = instructions

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        pass

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'function',
            'function': {
                'name': self.name,
                'description': self.description,
                'parameters': self.get_parameters(),
            },
        }

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        pass