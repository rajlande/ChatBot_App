import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set Streamlit page config
st.set_page_config(page_title="DialoGPT Chatbot", layout="centered")

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    return tokenizer, model

tokenizer, model = load_model()

# Initialize session state
if "chat_history_ids" not in st.session_state:
    st.session_state.chat_history_ids = None
if "generated" not in st.session_state:
    st.session_state.generated = []
if "past" not in st.session_state:
    st.session_state.past = []

# Title and description
st.title("💬 DialoGPT Chatbot")
st.write("This is a conversational AI powered by DialoGPT. Start chatting below:")

# Input form
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("You:", "")
    submitted = st.form_submit_button("Send")

# Handle user input
if submitted and user_input:
    # Encode input and build chat history
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    
    if st.session_state.chat_history_ids is not None:
        bot_input_ids = torch.cat([st.session_state.chat_history_ids, input_ids], dim=-1)
    else:
        bot_input_ids = input_ids

    # Generate response
    with st.spinner("DialoGPT is typing..."):
        st.session_state.chat_history_ids = model.generate(
            bot_input_ids,
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,  # Added randomness for better conversation
            top_k=50,
            top_p=0.95
        )

    # Decode response
    response = tokenizer.decode(
        st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0],
        skip_special_tokens=True
    )

    # Update chat history
    st.session_state.past.append(user_input)
    st.session_state.generated.append(response)

# Display conversation
if st.session_state.generated:
    for user_msg, bot_msg in zip(st.session_state.past, st.session_state.generated):
        st.markdown(f"**🧑 You:** {user_msg}")
        st.markdown(f"**🤖 Bot:** {bot_msg}")

# Reset button
if st.button("🔁 Reset Chat"):
    st.session_state.chat_history_ids = None
    st.session_state.generated = []
    st.session_state.past = []
    st.experimental_rerun()
