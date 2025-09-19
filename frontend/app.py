import streamlit as st
import requests
import uuid

# --- CONFIGURATION ---
BACKEND_URL = "http://127.0.0.1:8080"

# --- API COMMUNICATION FUNCTIONS ---
def ingest_files_api(session_id, files):
    """Sends files to the backend for processing."""
    file_data = [("files", (file.name, file.getvalue(), file.type)) for file in files]
    try:
        response = requests.post(
            f"{BACKEND_URL}/ingest",
            files=file_data,
            data={"session_id": session_id}
        )
        response.raise_for_status() # Raises an exception for 4XX/5XX errors
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to backend: {e}")
        return None

def chat_api(session_id, message, chat_history):
    """Gets a chat response from the backend."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/chat",
            json={
                "session_id": session_id,
                "message": message,
                "chat_history": chat_history
            }
        )
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to backend: {e}")
        return "Sorry, I'm having trouble connecting to my brain. Please try again later."

# --- SESSION STATE INITIALIZATION ---
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_file_names" not in st.session_state:
    st.session_state.processed_file_names = []

# --- STREAMLIT APP INTERFACE ---
st.set_page_config(page_title="AI Research Assistant", page_icon="🔬")
st.title("🔬 AI Research Assistant")

with st.sidebar:
    st.header("Your Documents")
    uploaded_files = st.file_uploader(
        "Upload files for this session:", accept_multiple_files=True, type=['pdf', 'docx', 'pptx']
    )
    st.info(f"Session ID: {st.session_state.session_id}")

# --- DOCUMENT PROCESSING LOGIC ---
if uploaded_files:
    current_file_names = sorted([f.name for f in uploaded_files])
    if current_file_names != st.session_state.processed_file_names:
        with st.spinner("Processing documents..."):
            result = ingest_files_api(st.session_state.session_id, uploaded_files)
            if result and result["success"]:
                st.session_state.processed_file_names = current_file_names
                st.success(result["message"])
            elif result:
                st.error(result["message"])

# --- CHAT UI LOGIC ---
st.info("Ask me about your documents or anything from the web!")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like to research?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        api_history = st.session_state.messages[:-1]
        ai_response = chat_api(st.session_state.session_id, prompt, api_history)
        
        message_placeholder.markdown(ai_response)
    
    st.session_state.messages.append({"role": "assistant", "content": ai_response})