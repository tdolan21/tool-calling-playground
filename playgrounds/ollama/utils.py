import streamlit as st

def get_ollama_options():
    options = {}
    
    with st.sidebar.expander("Parameters", expanded=False):
        num_predict = st.number_input("Num Predict", value=100, min_value=1, key="num_predict")
        if st.session_state.num_predict != 100:
            options["num_predict"] = num_predict

        top_k = st.number_input("Top K", value=20, min_value=1, key="top_k")
        if st.session_state.top_k != 20:
            options["top_k"] = top_k

        top_p = st.slider("Top P", 0.0, 1.0, value=0.9, key="top_p")
        if st.session_state.top_p != 0.9:
            options["top_p"] = top_p

        temperature = st.slider("Temperature", 0.0, 2.0, value=0.8, key="temperature")
        if st.session_state.temperature != 0.8:
            options["temperature"] = temperature

    with st.sidebar.expander("Advanced Options", expanded=False):
        num_ctx = st.number_input("Num Ctx", value=1024, min_value=1, key="num_ctx")
        if st.session_state.num_ctx != 1024:
            options["num_ctx"] = num_ctx
        
        num_batch = st.number_input("Num Batch", value=2, min_value=1, key="num_batch")
        if st.session_state.num_batch != 2:
            options["num_batch"] = num_batch

        min_p = st.slider("Min P", 0.0, 1.0, value=0.0, key="min_p")
        if st.session_state.min_p != 0.0:
            options["min_p"] = min_p

        tfs_z = st.slider("TFS Z", 0.0, 1.0, value=0.5, key="tfs_z")
        if st.session_state.tfs_z != 0.5:
            options["tfs_z"] = tfs_z

        typical_p = st.slider("Typical P", 0.0, 1.0, value=0.7, key="typical_p")
        if st.session_state.typical_p != 0.7:
            options["typical_p"] = typical_p

        repeat_last_n = st.number_input("Repeat Last N", value=33, min_value=1, key="repeat_last_n")
        if st.session_state.repeat_last_n != 33:
            options["repeat_last_n"] = repeat_last_n

        repeat_penalty = st.slider("Repeat Penalty", 0.0, 2.0, value=1.2, key="repeat_penalty")
        if st.session_state.repeat_penalty != 1.2:
            options["repeat_penalty"] = repeat_penalty

        presence_penalty = st.slider("Presence Penalty", 0.0, 2.0, value=1.5, key="presence_penalty")
        if st.session_state.presence_penalty != 1.5:
            options["presence_penalty"] = presence_penalty

        frequency_penalty = st.slider("Frequency Penalty", 0.0, 2.0, value=1.0, key="frequency_penalty")
        if st.session_state.frequency_penalty != 1.0:
            options["frequency_penalty"] = frequency_penalty

        mirostat = st.selectbox("Mirostat", [0, 1, 2], index=1, key="mirostat")
        if st.session_state.mirostat != 1:
            options["mirostat"] = mirostat

        mirostat_tau = st.slider("Mirostat Tau", 0.0, 2.0, value=0.8, key="mirostat_tau")
        if st.session_state.mirostat_tau != 0.8:
            options["mirostat_tau"] = mirostat_tau

        mirostat_eta = st.slider("Mirostat Eta", 0.0, 2.0, value=0.6, key="mirostat_eta")
        if st.session_state.mirostat_eta != 0.6:
            options["mirostat_eta"] = mirostat_eta

        penalize_newline = st.checkbox("Penalize Newline", value=True, key="penalize_newline")
        if not st.session_state.penalize_newline:
            options["penalize_newline"] = penalize_newline

        stop_words = st.text_area("Stop Words (comma-separated)", value='', key="stop_words")
        if st.session_state.stop_words != '\n,user:':
            options["stop"] = [word.strip() for word in stop_words.split(',') if word.strip()]

        numa = st.checkbox("NUMA", value=False, key="numa")
        if st.session_state.numa:
            options["numa"] = numa

        num_gpu = st.number_input("Num GPU", value=1, min_value=0, key="num_gpu")
        if st.session_state.num_gpu != 1:
            options["num_gpu"] = num_gpu

        main_gpu = st.number_input("Main GPU", value=0, min_value=0, key="main_gpu")
        if st.session_state.main_gpu != 0:
            options["main_gpu"] = main_gpu

        low_vram = st.checkbox("Low VRAM", value=False, key="low_vram")
        if st.session_state.low_vram:
            options["low_vram"] = low_vram

        f16_kv = st.checkbox("F16 KV", value=True, key="f16_kv")
        if not st.session_state.f16_kv:
            options["f16_kv"] = f16_kv

        vocab_only = st.checkbox("Vocab Only", value=False, key="vocab_only")
        if st.session_state.vocab_only:
            options["vocab_only"] = vocab_only

        use_mmap = st.checkbox("Use MMAP", value=True, key="use_mmap")
        if not st.session_state.use_mmap:
            options["use_mmap"] = use_mmap

        use_mlock = st.checkbox("Use MLOCK", value=False, key="use_mlock")
        if st.session_state.use_mlock:
            options["use_mlock"] = use_mlock

        num_thread = st.number_input("Num Thread", value=8, min_value=1, key="num_thread")
        if st.session_state.num_thread != 8:
            options["num_thread"] = num_thread

        
    return options
