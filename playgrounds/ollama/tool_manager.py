from typing import List, Dict, Any

class ToolManager:
    def __init__(self):
        self.tools = {}
        self.tool_switches = {}

    def register_tool(self, name: str, tool_instance):
        self.tools[name] = tool_instance
        self.tool_switches[name] = False
        
    def get_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                'type': 'function',
                'function': {
                    'name': name,
                    'description': getattr(tool, 'description', ''),
                    'parameters': getattr(tool, 'parameters', {}),
                }
            }
            for name, tool in self.tools.items() if self.tool_switches[name]
        ]

    def get_tool(self, name: str):
        return self.tools.get(name)
    
    def get_tool_names(self) -> List[str]:
        return list(self.tools.keys())

    def get_tool_instructions(self, name: str) -> str:
        return self.tool_instructions.get(name, "")

    def set_tool_instructions(self, name: str, instructions: str):
        if name in self.tools:
            self.tool_instructions[name] = instructions
            if hasattr(self.tools[name], 'instructions'):
                self.tools[name].instructions = instructions

    def get_all_tool_instructions(self) -> str:
        return "\n\n".join([
            f"{name} Instructions:\n{instructions}"
            for name, instructions in self.tool_instructions.items()
            if self.tool_switches[name]
        ])

    def set_tool_switch(self, name: str, state: bool):
        if name in self.tool_switches:
            self.tool_switches[name] = state

    def get_tool_switch(self, name: str) -> bool:
        return self.tool_switches.get(name, False)

    def get_active_tools(self) -> List[Dict[str, Any]]:
        return self.get_tools()

    async def execute_tool(self, tool_name: str, **kwargs) -> Any:
        if tool_name not in self.tools or not self.tool_switches[tool_name]:
            raise ValueError(f"Tool '{tool_name}' not found or not active")
        tool = self.tools[tool_name]
        if hasattr(tool, 'query_page_content'):
            return await tool.query_page_content(**kwargs)
        elif hasattr(tool, 'execute'):
            return await tool.execute(**kwargs)
        else:
            raise ValueError(f"Tool '{tool_name}' does not have a valid execution method")

    def get_tool_description(self, name: str) -> str:
        if name in self.tools:
            return getattr(self.tools[name], 'description', '')
        return ''

    def get_tool_parameters(self, name: str) -> Dict[str, Any]:
        if name in self.tools:
            return getattr(self.tools[name], 'parameters', {})
        return {}