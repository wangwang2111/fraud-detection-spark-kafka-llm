import streamlit as st
import openai
from openai import OpenAI
from datetime import datetime

# === Setup: Connect to LM Studio's local server ===
client = OpenAI(
    base_url="http://192.168.56.1:1234/v1",  # 👈 This is critical
    api_key="not-needed",  # Can be any dummy string; LM Studio doesn't check
)

MODEL_NAME = "deepseek-r1-0528-qwen3-8b"  # Change to your loaded model name in LM Studio

st.set_page_config(page_title="DeepSeek Chat", layout="centered")

# === Sidebar info ===
with st.sidebar:
    st.title("🤖 DeepSeek Chat (Local)")
    st.markdown("This app connects to your local DeepSeek model via LM Studio.")
    st.markdown("Make sure LM Studio is running at `http://192.168.56.1:1234/v1`.")
    st.markdown("---")
    temperature = st.slider("Response Creativity (Temperature)", 0.0, 1.5, 0.7, 0.05)
    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit and OpenAI-compatible API")

# === Chat title ===
st.title("💬 Chat with DeepSeek (Local)")

# === Initialize message history ===
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]

# === Display chat history ===
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# === Input box ===
user_input = st.chat_input("Ask me anything...")
if user_input:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Query the local DeepSeek model via LM Studio
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=st.session_state.messages,
                    temperature=temperature
                )
                reply = response.choices[0].message.content
            except Exception as e:
                reply = f"⚠️ Error: {e}"

            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
