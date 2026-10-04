import streamlit as st
import requests
import time

st.set_page_config(page_title="RAGOps Assistant", page_icon="🤖", layout="wide")

API_URL = "http://127.0.0.1:8000/api/v1"

st.title("🤖 RAGOps Enterprise Knowledge Assistant")
st.markdown("Ask questions grounded in our verified organizational documents.")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for configuration and system status
with st.sidebar:
    st.header("⚙️ Configuration")
    retriever_option = st.selectbox(
        "Select Retriever",
        ["semantic", "hybrid", "bm25", "tfidf"],
        index=0
    )
    top_k = st.slider("Top K Sources", min_value=1, max_value=10, value=5)
    department = st.text_input("Department Filter (Optional)", placeholder="e.g. HR, Engineering")

    st.divider()
    
    st.header("📊 System Health")
    try:
        health_resp = requests.get(f"{API_URL}/health", timeout=2)
        if health_resp.status_code == 200:
            health_data = health_resp.json()
            st.success(f"Status: {health_data.get('status', 'Unknown')}")
            st.metric("Vectorstore Chunks", health_data.get('vectorstore_chunks_count', 0))
        else:
            st.error("Backend health endpoint returned error.")
    except Exception:
        st.error("Backend not reachable. Please ensure the API is running.")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("View Sources"):
                for idx, src in enumerate(message["sources"]):
                    meta = src.get("metadata", {}) if isinstance(src, dict) else {}
                    title = meta.get("title") or src.get("document_id") or meta.get("source", "Document")
                    score = src.get("score") or meta.get("score", 0.0)
                    chunk_text = src.get("text") or src.get("page_content") or ""
                    st.markdown(f"**Source {idx+1}:** {title} *(Score: {score:.4f})*")
                    if chunk_text:
                        preview = chunk_text.strip()
                        if len(preview) > 600:
                            preview = preview[:600].rstrip() + "..."
                        st.text(preview)

# Accept user input
if prompt := st.chat_input("Ask a question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking... ⏳")
        
        payload = {
            "question": prompt,
            "retriever": retriever_option,
            "top_k": top_k
        }
        if department:
            payload["department"] = department
            
        try:
            start_time = time.time()
            response = requests.post(f"{API_URL}/query", json=payload)
            latency = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                answer = data["answer"]
                sources = data.get("sources", [])
                
                # Show answer
                message_placeholder.markdown(answer)
                
                # Metrics / Metadata
                st.caption(f"⏱️ Total Latency: {data.get('total_latency_ms', 0)} ms | 🔍 Retriever: {data.get('retriever_used', retriever_option)}")
                
                if data.get("is_fallback"):
                    st.warning("⚠️ The model couldn't find a grounded answer and used the fallback protocol.")
                
                # Sources expander
                if sources:
                    with st.expander("View Sources"):
                        for idx, src in enumerate(sources):
                            meta = src.get("metadata", {}) if isinstance(src, dict) else {}
                            title = meta.get("title") or src.get("document_id") or meta.get("source", "Document")
                            score = src.get("score") or meta.get("score", 0.0)
                            chunk_text = src.get("text") or src.get("page_content") or ""
                            st.markdown(f"**Source {idx+1}:** {title} *(Score: {score:.4f})*")
                            if chunk_text:
                                preview = chunk_text.strip()
                                if len(preview) > 600:
                                    preview = preview[:600].rstrip() + "..."
                                st.text(preview)
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
                
            elif response.status_code == 503:
                err_data = response.json() if "application/json" in response.headers.get("content-type", "") else {}
                detail = err_data.get("detail", {}) if isinstance(err_data.get("detail"), dict) else {}
                err_msg = err_data.get("error") or detail.get("error") or err_data.get("detail") or response.text
                provider = err_data.get("provider") or detail.get("provider") or "llm"

                if str(provider).lower() == "ollama":
                    resolve_hint = "Ensure Ollama is running (`ollama serve`) and the model is pulled (`ollama pull llama3.2:latest`)."
                elif str(provider).lower() == "gemini":
                    resolve_hint = "Ensure a valid `GEMINI_API_KEY` is set in `.env` with available quota at Google AI Studio."
                elif str(provider).lower() == "openai":
                    resolve_hint = "Ensure a valid `OPENAI_API_KEY` is set in `.env`."
                else:
                    resolve_hint = f"Ensure `{str(provider).upper()}` credentials and model are configured in `.env`."

                alert_text = (
                    f"⚠️ **{str(provider).upper()} Service Unavailable (HTTP 503)**\n\n"
                    f"**Details:** {err_msg}\n\n"
                    f"💡 *To resolve:* {resolve_hint}"
                )
                message_placeholder.warning(alert_text)
                st.session_state.messages.append({"role": "assistant", "content": alert_text})
            else:
                error_msg = f"Error {response.status_code}: {response.text}"
                message_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                
        except requests.exceptions.RequestException as e:
            error_msg = f"API Request failed: {str(e)}"
            message_placeholder.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

