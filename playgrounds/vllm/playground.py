import streamlit as st
from openai import OpenAI
import json
from typing import List, Optional, Dict
import io
import fitz  # PyMuPDF
import pandas as pd
from pydantic import BaseModel, Field, field_validator
import time 

# Initialize OpenAI client
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",
)

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    max_tokens: int = Field(256, ge=1, le=2048)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    n: int = Field(1, ge=1, le=5)
    stream: bool = False
    stop: Optional[List[str]] = None
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    logit_bias: Dict[str, float] = {}
    user: Optional[str] = None
    extra_body: Dict = Field(default_factory=dict)

    @field_validator('stop', mode='before')
    def split_stop_sequences(cls, v):
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v

    @field_validator('logit_bias', mode='before')
    def parse_logit_bias(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON for Logit Bias")
        return v

    @field_validator('extra_body', mode='before')
    def validate_extra_body(cls, v):
        if 'stop_token_ids' in v and isinstance(v['stop_token_ids'], str):
            try:
                v['stop_token_ids'] = [int(id.strip()) for id in v['stop_token_ids'].split(",") if id.strip()]
            except ValueError:
                raise ValueError("Invalid input for Stop Token IDs. Please enter comma-separated integers.")
        return v

def extract_text_from_file(file):
    file_extension = file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        pdf_document = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page in pdf_document:
            text += page.get_text()
        return text
    elif file_extension in ['xlsx', 'xls']:
        df = pd.read_excel(file)
        return df.to_string()
    else:
        st.error(f"Unsupported file format: {file_extension}")
        return None

def get_unique_key():
    st.session_state.message_counter += 1
    return f"message_{time.time()}_{st.session_state.message_counter}"


st.set_page_config(page_title="VLLM Playground", page_icon=".\vllm-scripts\assets\vllm-icon.png")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "metrics" not in st.session_state:
    st.session_state.metrics = []
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0
if "documents" not in st.session_state:
    st.session_state.documents = []
if 'message_counter' not in st.session_state:
    st.session_state.message_counter = 0
    
st.sidebar.image("./assets/vllm-logo-text-light.png")

# Sidebar for model selection, parameters, and document upload
st.sidebar.header("Model Configuration")
model = st.sidebar.text_input(
    label="Model",
    value="neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8"
)

# System message
system_message = st.sidebar.text_area("System Message", "You are a helpful assistant.")

# Document upload
st.sidebar.header("Document Upload (RAG)")
uploaded_files = st.sidebar.file_uploader("Upload PDFs or Spreadsheets for RAG", accept_multiple_files=True, type=['pdf', 'xlsx', 'xls'])

if uploaded_files:
    for file in uploaded_files:
        text_content = extract_text_from_file(file)
        if text_content:
            st.session_state.documents.append({
                "title": file.name,
                "text": text_content
            })
    st.sidebar.success(f"{len(uploaded_files)} document(s) uploaded successfully!")

# Display uploaded documents
if st.session_state.documents:
    st.sidebar.subheader("Uploaded Documents")
    for doc in st.session_state.documents:
        st.sidebar.text(doc['title'])

# Parameters
with st.sidebar.expander("Parameters", expanded=True):
    max_tokens = st.slider("Max Tokens", 1, 2048, 256)
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7)
    top_p = st.slider("Top P", 0.0, 1.0, 1.0)
    n = st.number_input("Number of Completions", 1, 5, 1)
    stream = st.checkbox("Stream", value=False)
    stop = st.text_input("Stop Sequences (comma-separated)")
    presence_penalty = st.slider("Presence Penalty", -2.0, 2.0, 0.0)
    frequency_penalty = st.slider("Frequency Penalty", -2.0, 2.0, 0.0)
    logit_bias = st.text_area("Logit Bias (JSON format)", "{}")
    user = st.text_input("User Identifier")

# Extra parameters in an expander
with st.sidebar.expander("Extra Parameters"):
    best_of = st.number_input("Best Of", 1, 5, 1)
    use_beam_search = st.checkbox("Use Beam Search", value=False)
    top_k = st.number_input("Top K", -1, 100, -1)
    min_p = st.slider("Min P", 0.0, 1.0, 0.0)
    repetition_penalty = st.slider("Repetition Penalty", 1.0, 2.0, 1.0)
    length_penalty = st.slider("Length Penalty", 0.1, 2.0, 1.0)
    early_stopping = st.checkbox("Early Stopping", value=False)
    stop_token_ids = st.text_input("Stop Token IDs (comma-separated integers)")
    include_stop_str_in_output = st.checkbox("Include Stop String in Output", value=False)
    ignore_eos = st.checkbox("Ignore EOS", value=False)
    min_tokens = st.number_input("Min Tokens", 0, 100, 0)
    skip_special_tokens = st.checkbox("Skip Special Tokens", value=True)
    spaces_between_special_tokens = st.checkbox("Spaces Between Special Tokens", value=True)
    truncate_prompt_tokens = st.number_input("Truncate Prompt Tokens", 1, 2048, 2048)
    
with st.sidebar.expander("Guided Generation Settings"):
    guided_option = st.radio("Select Guided Option", ["None", "JSON", "Regex", "Choice", "Grammar", "Decoding Backend"])
    
    guided_json = None
    guided_regex = None
    guided_choice = None
    guided_grammar = None
    guided_decoding_backend = None
 
    if guided_option == "JSON":
        st.info("Enter a JSON Schema to format the output")
        guided_grammar = st.text_area("Enter JSON Schema")
    elif guided_option == "Choice":
        st.info("Each choice is one of the possible outputs")
        num_choices = st.number_input("Number of Choices", 2, 10, 2)
        guided_choice = [st.text_input(f"Choice {i+1}") for i in range(num_choices)]
    elif guided_option == "Grammar":
        st.info("Learn more about [GBNF Grammar](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md)")
        guided_grammar = st.text_area("Enter Context Free Grammar")
    elif guided_option == "Regex":
        st.info("Enter a regex pattern to format output")
        guided_regex = st.text_input("Enter Regex Pattern")
    elif guided_option == "Decoding Backend":
        st.info("In vLLM, the guided decoding backend must only be set to one of these values")
        guided_decoding_backend = st.radio("Select Decoding Backend", ["outlines", "lm-format-enforcer"])

from streamlit_chat import message

for msg in st.session_state.messages:
    message(msg["content"], is_user=(msg["role"] == "user"), key=get_unique_key())

# Chat input
if prompt := st.chat_input("What is your message?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    message(prompt, is_user=True, key=get_unique_key())

    
    
    full_response = ""
    response_placeholder = st.empty()

        # Prepare messages for the API call
    messages = [{"role": "system", "content": system_message}] + [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

        # Create ChatCompletionRequest object
    try:
        extra_body = {
                "best_of": best_of,
                "use_beam_search": use_beam_search,
                "top_k": top_k,
                "min_p": min_p,
                "repetition_penalty": repetition_penalty,
                "length_penalty": length_penalty,
                "early_stopping": early_stopping,
                "stop_token_ids": stop_token_ids.split(",") if stop_token_ids else None,
                "include_stop_str_in_output": include_stop_str_in_output,
                "ignore_eos": ignore_eos,
                "min_tokens": min_tokens,
                "skip_special_tokens": skip_special_tokens,
                "spaces_between_special_tokens": spaces_between_special_tokens,
                "truncate_prompt_tokens": truncate_prompt_tokens,
                "documents": st.session_state.documents if st.session_state.documents else None
            }

            # Add guided parameters if selected
        if guided_option == "Regex" and guided_regex:
                extra_body["guided_regex"] = guided_regex
        elif guided_option == "Choice" and guided_choice:
                extra_body["guided_choice"] = [choice for choice in guided_choice if choice]
        elif guided_option == "Grammar" and guided_grammar:
                extra_body["guided_grammar"] = guided_grammar
        elif guided_option == "Decoding Backend" and guided_decoding_backend:
                extra_body["guided_decoding_backend"] = guided_decoding_backend
        elif guided_option == "JSON" and guided_json:
                extra_body["guided_json"] = guided_json

        request = ChatCompletionRequest(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                n=n,
                stream=stream,
                stop=stop,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                logit_bias=logit_bias,
                user=user,
                extra_body=extra_body
            )
    except ValueError as e:
        st.error(f"Invalid input: {str(e)}")
        st.stop()

    try:
        completion = client.chat.completions.create(**request.model_dump())

        if stream:
            # Use Streamlit's native chat_message for streaming
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                for chunk in completion:
                    if chunk.choices[0].delta.content is not None:
                        full_response += chunk.choices[0].delta.content
                        
                
                message(full_response, key=get_unique_key())
                message_placeholder.markdown(full_response)

        else:
            full_response = completion.choices[0].message.content
            message(full_response, key=get_unique_key())
            usage = completion.usage
            # Add assistant's response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            # Update metrics
            st.session_state.metrics.append({
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens
            })
            st.session_state.total_tokens += usage.total_tokens

            # Display updated metrics
            st.sidebar.header("Production Metrics")
            last_metric = st.session_state.metrics[-1]
            st.sidebar.info(f"""
            **Last Request:**
            Prompt Tokens: {last_metric['prompt_tokens']}
            Completion Tokens: {last_metric['completion_tokens']}
            Total Tokens: {last_metric['total_tokens']}

            **Application Total:**
            Total Tokens Served: {st.session_state.total_tokens}
            """)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Display raw API response
if st.sidebar.checkbox("Show Raw API Response"):
    st.subheader("Raw API Response")
    st.json(completion.model_dump())