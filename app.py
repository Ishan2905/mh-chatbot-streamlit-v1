import streamlit as st
import os
import json
import base64
from firebase_admin import credentials, firestore, initialize_app
from dotenv import load_dotenv
import openai
from anthropic import Anthropic
import firebase_admin

# Load environment variables
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
anthropic = Anthropic(api_key=anthropic_api_key)

# --- Firebase Initialization ---
firebase_creds_b64 = os.getenv("FIREBASE_CREDENTIALS")
firebase_creds_json = json.loads(base64.b64decode(firebase_creds_b64).decode("utf-8"))
cred = credentials.Certificate(firebase_creds_json)

# Add check for existing Firebase app
if not firebase_admin._apps:
    initialize_app(cred)
db = firestore.client()

# --- Firestore Helper Functions ---
def get_conversation(user_id):
    """Retrieve conversation history for a given user from Firestore."""
    doc_ref = db.collection("conversations").document(user_id)
    doc = doc_ref.get()
    conversation = doc.to_dict().get("conversation", []) if doc.exists else []
    print(f"\nConversation: {conversation}\n")
    return conversation

def update_conversation_messages(user_id, new_messages):
    """Append new message objects to the user's conversation history in Firestore."""
    doc_ref = db.collection("conversations").document(user_id)
    conversation = []
    doc = doc_ref.get()
    if doc.exists:
        conversation = doc.to_dict().get("conversation", [])
    conversation.extend(new_messages)
    print(f"\nUpdated Conversation: {conversation}\n")
    doc_ref.set({"conversation": conversation}, merge=True)

# --- LLM Response Function ---
def get_llm_response(conversation_history, incoming_msg, provider="anthropic"):
    """
    Use either OpenAI or Anthropic to generate a response.
    Loads the system prompt from prompts.json.
    """
    with open("prompts.json", "r", encoding="utf-8") as f:
        prompts = json.load(f)
    system_prompt = prompts["system_prompt"]

    if provider == "openai":
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": incoming_msg})
        print(f"\nMessages sent to LLM: {messages}\n")
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages,
            temperature=1
        )
        print(f"\nLLM Response: {response}\n")
        return response.choices[0].message.content.strip()
        
    elif provider == "anthropic":
        messages = []
        for msg in conversation_history:
            # Prepare messages in the format expected by Anthropic
            if msg["role"] == "user":
                messages.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                messages.append({"role": "assistant", "content": msg["content"]})
        messages.append({"role": "user", "content": incoming_msg})
        print(f"\nMessages sent to LLM: {messages}\n")
        response = anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=messages,
            system=system_prompt
        )
        print(f"\nLLM Response: {response}\n")
        return response.content[0].text.strip()
        
    else:
        raise ValueError(f"Unsupported provider: {provider}")

# --- Streamlit App ---
def main():
    st.title("Tuco - Your AI Therapist")

    # Ask for a user identifier (in this case, a name)
    if "user_id" not in st.session_state or not st.session_state.user_id:
        st.session_state.user_id = st.text_input(
            "Hi! I'm your AI therapist. I'm here to understand your problems, help you explore solutions and build on your strengths. I would love to know more about you, what should I address you as?"
        )
        if not st.session_state.user_id:
            st.stop()
        else:
            # Load any previous conversation from Firestore
            st.session_state.conversation = get_conversation(st.session_state.user_id)
            print(f"\nFetched conversations: {st.session_state.conversation}")

    st.markdown(f"Welcome, {st.session_state.user_id}! Please tell me a little about yourself.")

    # Display conversation history with a divider between messages
    for message in st.session_state.conversation:
        if message["role"] == "user":
            st.markdown(f"**{st.session_state.user_id}:** {message['content']}")
        else:
            st.markdown(f"**Tuco:** {message['content']}")
        st.markdown("---")  # Divider between messages

    # Create typing indicator placeholder before the input field
    typing_placeholder = st.empty()

    # Create a container for input and button
    input_container = st.container()
    
    # Create two columns for the input field and send button
    col1, col2 = input_container.columns([6, 1])

    # Input field for new message using text_area instead of text_input
    if "user_input_key" not in st.session_state:
        st.session_state.user_input_key = 0

    with col1:
        # Add a help text to explain the shortcuts
        st.caption("Press Cmd/Ctrl + Enter to send,  Enter for new line")
        user_input = st.text_area(
            "Your message:",
            key=f"user_input_{st.session_state.user_input_key}",
            height=100,
            on_change=None,  # Remove any existing on_change handler
        )

    with col2:
        send_button = st.button("Send", use_container_width=True)

    # Handle message sending (either through button click or Enter key)
    if (send_button or (user_input and user_input.strip() and "\n" not in user_input)) and user_input.strip():
        # 1. Display user message immediately: Append to conversation and update Firestore
        user_message = {"role": "user", "content": user_input.strip()}
        st.session_state.conversation.append(user_message)
        update_conversation_messages(st.session_state.user_id, [user_message])
        
        # Increment the key to force a new input box
        st.session_state.user_input_key += 1
        
        st.experimental_rerun()  # Rerun to show the user message instantly

    # If a new user message was added but response not yet generated:
    # Check if the last message is from the user (i.e. awaiting response).
    if st.session_state.conversation and st.session_state.conversation[-1]["role"] == "user":
        # Show typing indicator while waiting for response.
        typing_placeholder.text("Tuco is thinking...")
        
        # Get response from LLM (simulate streaming by waiting and then updating)
        response_text = get_llm_response(st.session_state.conversation[:-1], st.session_state.conversation[-1]["content"])
        
        # Clear typing indicator
        typing_placeholder.empty()
        
        assistant_message = {"role": "assistant", "content": response_text}
        st.session_state.conversation.append(assistant_message)
        update_conversation_messages(st.session_state.user_id, [assistant_message])
        
        st.experimental_rerun()

if __name__ == "__main__":
    main()

