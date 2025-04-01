import streamlit as st

# Function to display chat text window and handle chat input
def display_chat(display_area):
    user_input = st.text_area("Chat with us", placeholder="Type your message here...", height=68)
    if st.button("Send"):
        if user_input:
            answer = f"Answer to your question: {user_input}"
            display_area.text_area("Responses", value=answer, height=100, placeholder="Responses will appear here...", key="response_area")

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