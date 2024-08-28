import streamlit as st
import asyncio
import json
import ollama 
from tool_manager import ToolManager
from ollama_client import OllamaClient
from utils import get_ollama_options

from tools.web_scraper import WebScraper
from tools.calculator import CalculatorTool

DEFAULT_SYSTEM_MESSAGE = """You are an AI assistant specialized in processing queries and using tools when necessary. 
Always follow the specific instructions provided for each tool when you use them."""

async def process_query(query: str, model: str, tool_manager: ToolManager, options: dict, system_message: str, tool_instructions: dict, chat_history: list):
    ollama_client = OllamaClient(model)
    
    working_history = chat_history.copy()
    
    if not working_history or working_history[0]['role'] != 'system':
        working_history.insert(0, {'role': 'system', 'content': system_message})
    else:
        working_history[0]['content'] = system_message
    
    working_history.append({'role': 'user', 'content': query})
    
    active_tools = tool_manager.get_active_tools()
    
    response = await ollama_client.chat(working_history, active_tools, options)
    working_history.append(response['message'])
    
    if active_tools and response['message'].get('tool_calls'):
        for tool_call in response['message']['tool_calls']:
            tool_name = tool_call['function']['name']
            tool_args = tool_call['function']['arguments']
            
            st.sidebar.write(f"Using tool: {tool_name}")
            st.sidebar.write(f"Tool arguments: {tool_args}")
            
            tool_result = await tool_manager.execute_tool(tool_name, **tool_args)
            
            instructions = tool_instructions.get(tool_name, "Process the tool results and provide a response.")

            formatted_data = json.dumps(tool_result.get('extracted_data', tool_result), indent=2)
            
            working_history.append({
                'role': 'function',
                'name': tool_name,
                'content': formatted_data,
            })
            
            additional_instruction = {
                'role': 'user',
                'content': f"""Here is the data from the {tool_name}:

{formatted_data}

Instructions for processing this data:
{instructions}

"""            }
            
            working_history.append(additional_instruction)
        
        final_response = await ollama_client.chat(working_history, options=options)
        return final_response['message']['content']
    
    return response['message']['content']


async def main():
    st.set_page_config(page_title="Ollama Multi-Tool Demo", layout="wide", page_icon="https://ollama.com/public/icon-64x64.png")
    
    st.sidebar.title("Ollama Multi-Tool Demo")
    
    st.sidebar.header("Configuration")
    model = st.sidebar.selectbox("Select Model", ["llama3.1", "llama3.1:8b-instruct-fp16", "llama3.1:70b", "mistral"], index=0)
    
    ollama_options = get_ollama_options()

    with st.sidebar.expander("Ollama Controls"):
        pull = st.button(f"ollama pull {model}")
        st.sidebar.info("**If you have not pulled the model from Ollama, you can do so in the Ollama controls expander.**")
        if pull:
            ollama.pull(model)
    
    if 'tool_manager' not in st.session_state:
        st.session_state.tool_manager = ToolManager()
        # Initialize tools here
        st.session_state.tool_manager.register_tool("web_scraper", WebScraper())
        st.session_state.tool_manager.register_tool("calculator", CalculatorTool())

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'tool_instructions' not in st.session_state:
        st.session_state.tool_instructions = {}
    
    if 'tool_switches' not in st.session_state:
        st.session_state.tool_switches = {name: False for name in st.session_state.tool_manager.get_tool_names()}
    
    st.sidebar.header("Tool Configuration")
    
    # Tool toggles
    for tool_name in st.session_state.tool_manager.get_tool_names():
        use_tool = st.sidebar.toggle(
            f"Enable {tool_name.replace('_', ' ').title()}",
            value=st.session_state.tool_switches.get(tool_name, False),
            key=f"toggle_{tool_name}"
        )
        st.session_state.tool_switches[tool_name] = use_tool
        st.session_state.tool_manager.set_tool_switch(tool_name, use_tool)
    
    # System message editor
    st.sidebar.subheader("System Message")
    system_message = st.sidebar.text_area("System Message", DEFAULT_SYSTEM_MESSAGE, height=150)
    
    # Tool instructions editors
    st.sidebar.subheader("Tool Instructions")
    for tool_name in st.session_state.tool_manager.get_tool_names():
        if st.session_state.tool_switches.get(tool_name, False):
            tool = st.session_state.tool_manager.get_tool(tool_name)
            if tool and hasattr(tool, 'instructions'):
                default_instructions = tool.instructions
            else:
                default_instructions = "Process the tool results and provide a response."
            
            st.session_state.tool_instructions[tool_name] = st.sidebar.text_area(
                f"{tool_name.replace('_', ' ').title()} Instructions",
                st.session_state.tool_instructions.get(tool_name, default_instructions),
                key=f"instructions_{tool_name}",
                height=150
            )

    if prompt := st.chat_input("Ask a question or request a task"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Processing...")
            
            result = await process_query(prompt, model, st.session_state.tool_manager, ollama_options, system_message, st.session_state.tool_instructions, st.session_state.chat_history)
            
            try:
                json_result = json.loads(result)
                formatted_result = json.dumps(json_result, indent=2)
                message_placeholder.code(formatted_result, language="json")
            except json.JSONDecodeError:
                message_placeholder.markdown(result)
            
            st.session_state.chat_history.append({"role": "assistant", "content": result})

if __name__ == "__main__":
    asyncio.run(main())