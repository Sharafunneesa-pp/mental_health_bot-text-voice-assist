import streamlit as st
from chat import chat_gen
import subprocess
import os

# Function to initialize the chat generator
@st.cache_resource
def initialize():
    chat = chat_gen()
    return chat

st.session_state.chat = initialize()

# Define Streamlit UI
st.title('👨‍⚕️ Echocare')
st.subheader('" 🫂Your Mental Health Companion 🫂"')
st.info("Hello, I'm Echo. I'm here to support your mental health as you navigate social media challenges.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_chat" not in st.session_state:
    st.session_state.current_chat = []

# Function to execute voice assistant script
def execute_voice_assistant():
    # Display initial listening message
    listening_text = st.empty()
    listening_text.info("I'm echo 🎙️ Listening...")

    # Use subprocess to trigger voice assistant script
    subprocess.run(['python', 'assistant_voice.py'])

# Sidebar for conversation history
st.sidebar.title('Conversation History🗣️')

# Button to clear chat in the center, positioned at the top
if st.sidebar.button('New Chat'):
    st.session_state.current_chat = []

# Display conversation history below the "Clear Chat" button
for message in st.session_state.messages:
    with st.sidebar.chat_message(f"{message['role']}"):
        st.sidebar.markdown(message["content"])

# Main area for current chat interaction and audio button
# Horizontal separator

# Audio button
if st.button("🎤", key="speak_button"):
    execute_voice_assistant()

# Display chat messages from current chat on app rerun
for message in st.session_state.current_chat:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.markdown("---") 

# Chat input and display
prompt = st.chat_input("Type here to chat 😍")
if prompt:
    # Display user message in main area
    st.chat_message("user").markdown(prompt)
    # Add user message to current chat
    st.session_state.current_chat.append({"role": "user", "content": prompt})

    response = st.session_state.chat.ask_query(prompt)
    # Display assistant response in main area
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to current chat
    st.session_state.current_chat.append({"role": "assistant", "content": response})

    # Also add messages to conversation history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": response})

# Additional information or footer
st.markdown("<div style='height: 200px;'></div>", unsafe_allow_html=True) 


