import os
import time
from http.client import responses
from groq import Groq
from httpx import stream
from langchain_groq import ChatGroq
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

def display_chat(height="400px"):
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    if "model_name" not in st.session_state:
        st.session_state["model_name"] = "llama-3.3-70b-versatile"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # display: flex;
    # flex-direction: column; /* Ensures new messages appear at the bottom */
    # Apply CSS for a scrollable chat container
    # st.markdown(
    #     f"""
    #     <style>
    #     .chat-container {{
    #         height: {height}; /* Fixed height */
    #         overflow-y: auto;
    #         border: 1px solid #ccc;
    #         padding: 10px;
    #         border-radius: 8px;
    #     }}
    #     </style>
    #     """,
    #     unsafe_allow_html=True,
    # )

    # # Create a scrollable chat container with an empty placeholder
    # chat_container = st.container(height=400, border=True)
    # with chat_container:
    #     chat_display = st.empty()

    # # Function to update chat display
    # def update_chat():
    #     chat_history = ""
    #     for message in st.session_state.messages:
    #         chat_history += f"**{message['role'].capitalize()}**: {message['content']}\n\n"
    #     chat_display.markdown(f'<div class="chat-container">{chat_history}</div>', unsafe_allow_html=True)

    # # Display existing chat messages
    # update_chat()

    # # User input
    # if prompt := st.chat_input("Ask a question?"):
    #     st.session_state.messages.append({"role": "user", "content": prompt})
    #     update_chat()  # Update chat after user input

    #     with st.spinner("Thinking..."):
    #         time.sleep(1)
    #         try:
    #             response = client.chat.completions.create(
    #                 model=st.session_state["model_name"],
    #                 messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
    #                 stream=False,
    #             ).choices[0].message.content

    #             st.session_state.messages.append({"role": "assistant", "content": response})
    #             update_chat()  # Update chat after assistant response

    #         except Exception as e:
    #             error_message = f"Oops! Too many users. Error: {e}"
    #             st.session_state.messages.append({"role": "assistant", "content": error_message})
    #             update_chat()  # Update chat with error message
    
    # Create a scrollable chat container with an empty placeholder
    chat_container = st.container(height=400, border=True)
    # with chat_container:
    #     chat_display = st.empty()

    # Function to update chat display
    def update_chat():
        # print(st.session_state.messages)
        for message in st.session_state.messages:
            chat_message =  chat_container.chat_message(message["role"], avatar="static/user1.png" if message["role"]=="user" else "static/bot.png")
            chat_message.markdown(message["content"])

    # Display existing chat messages
    update_chat()

    # User input
    if prompt := st.chat_input("Ask a question?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        # update_chat()  # Update chat after user input
        chat_message = chat_container.chat_message("user", avatar="static/user1.png")
        chat_message.markdown(prompt)

        with st.spinner("Thinking..."):
            time.sleep(1)
            try:
                response = client.chat.completions.create(
                    model=st.session_state["model_name"],
                    messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                    stream=False,
                ).choices[0].message.content

                st.session_state.messages.append({"role": "assistant", "content": response})
                # update_chat()  # Update chat after assistant response
                chat_message = chat_container.chat_message("assistant", avatar="static/bot.png")
                chat_message.markdown(response)

            except Exception as e:
                error_message = f"Oops! Too many users. Error: {e}"
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                update_chat()  # Update chat with error message


def hide_default_sidebar():
    st.markdown(
        """
        <style>
        [data-testid="stSidebarNav"], [data-testid="stSidebar"] {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
