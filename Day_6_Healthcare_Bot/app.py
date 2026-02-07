import streamlit as st
from chatbot import setup_chain

# 1. Page Config (Must be the first command)
st.set_page_config(page_title="Dr. AI Healthcare Bot", page_icon="🩺")
st.title("🩺 Dr. AI - Personal Health Assistant")

# 2. Session State (Memory)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. Initialize the Chain (Only once)
if "chain" not in st.session_state:
    try:
        st.session_state.chain = setup_chain()
    except Exception as e:
        st.error(f"Error setting up AI: {e}")
        st.stop()

# 4. Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 5. User Input Handler
user_input = st.chat_input("Describe your symptoms...")

if user_input:
    # Display User Message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Format History for the Prompt
    chat_history_text = "\n".join(
        f"{m['role']}: {m['content']}" 
        for m in st.session_state.messages
    )

    # Generate AI Response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing symptoms..."):
            try:
                # Invoke the chain
                response = st.session_state.chain.invoke({
                    "question": user_input,
                    "chat_history": chat_history_text
                })
                st.markdown(response)
                
                # Save Response
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                st.error(f"An error occurred: {e}")