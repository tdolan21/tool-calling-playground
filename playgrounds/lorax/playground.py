import streamlit as st
from lorax import Client
import os
from PIL import Image
import requests

def load_system_prompt(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def generate_response(client, prompt, **kwargs):
    for response in client.generate_stream(prompt, **kwargs):
        if response.token.special:
            continue  # Skip special tokens
        
        token_text = response.token.text
        if "<|im_end|>" in token_text:
            # Yield any text before the end token, then stop
            yield token_text.split("<|im_end|>")[0]
            break
        else:
            yield token_text

def fetch_metrics(endpoint_url):
    try:
        response = requests.get(f"{endpoint_url}/metrics")
        if response.status_code == 200:
            metrics = response.text
            decode_success = 0
            prefill_success = 0
            for line in metrics.split('\n'):
                if line.startswith('lorax_batch_inference_success{method="decode"}'):
                    decode_success = int(line.split()[-1])
                elif line.startswith('lorax_batch_inference_success{method="prefill"}'):
                    prefill_success = int(line.split()[-1])
            return decode_success, prefill_success
    except:
        pass
    return None, None

def main():
    st.set_page_config(page_title="Lorax Chat Demo", page_icon="🦁", layout="wide")

    # header photos
    image_path = "assets/lorax_guy.png"
    if os.path.exists(image_path):
        image = Image.open(image_path)
        st.sidebar.image(image, use_column_width=True)
    else:
        st.sidebar.warning("Header image not found. Please check the path: " + image_path)

    # sidebar config
    st.sidebar.title("Lorax Chat Demo")
    st.sidebar.header("Configuration")
    endpoint_url = st.sidebar.text_input("Endpoint URL", value="http://127.0.0.1:8080")
    adapter_source = st.sidebar.text_input("Adapter Source", value="hub")
    adapter_id = st.sidebar.text_input("Adapter ID", value="")
    api_token = st.sidebar.text_input("API Token", value="", type="password")
    st.sidebar.divider()
    system_prompt = st.sidebar.text_area("System Prompt", value="You are a helpful AI assistant", height=3)
    max_new_tokens = st.sidebar.number_input("Max New Tokens", value=315, min_value=1, max_value=1024)
    
    with st.sidebar.expander("Advanced Settings"):
        temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
        top_p = st.sidebar.slider("Top-p", 0.0, 1.0, 0.95)
        top_k = st.sidebar.slider("Top-k", 1, 10, 10)
        typical_p = st.sidebar.slider("Typical-p", 0.0, 1.0, 0.95)

    # Add template selection dropdown
    template_options = {
        "Base Model (Completion)":"""{ctx}""",

        "ChatML": """
            <|im_start|>system
            {system}<|im_end|>
            <|im_start|>user
            {ctx}<|im_end|>
            <|im_start|>assistant
            """,
       
    }

    selected_template = st.sidebar.selectbox("Select Template", list(template_options.keys()))

    # Initialize session state for chat history (only keeping the last message)
    if "last_message" not in st.session_state:
        st.session_state.last_message = None

    # Display last message if it exists
    if st.session_state.last_message:
        with st.chat_message(st.session_state.last_message["role"]):
            st.markdown(st.session_state.last_message["content"])

    # Chat input
    if prompt := st.chat_input("What's your question?"):
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            client = Client(endpoint_url)
            template = template_options[selected_template]
            full_prompt = template.format(ctx=prompt, system=system_prompt)
            response_container = st.empty()
            full_response = ""

            # Prepare kwargs for generate_response
            kwargs = {
                "adapter_source": adapter_source,
                "api_token": api_token,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "typical_p": typical_p,
                "stop_sequences": ["<|im_end|>"]
            }

            # Only add adapter_id if it's provided
            if adapter_id:
                kwargs["adapter_id"] = adapter_id

            for response_chunk in generate_response(client, full_prompt, **kwargs):
                full_response += response_chunk
                response_container.markdown(full_response + "▌")
            response_container.markdown(full_response)

        # Update the last message in session state
        st.session_state.last_message = {"role": "assistant", "content": full_response}

    # Fetch and display metrics
    decode_success, prefill_success = fetch_metrics(endpoint_url)
    if decode_success is not None and prefill_success is not None:
        metrics_info = f"""
        Inference Metrics:
        - Decode Success: {decode_success}
        - Prefill Success: {prefill_success}
        """
        st.sidebar.info(metrics_info)
    else:
        st.sidebar.warning("Unable to fetch metrics. Please check the endpoint URL.")

if __name__ == "__main__":
    main()