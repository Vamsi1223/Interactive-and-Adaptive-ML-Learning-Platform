import streamlit as st
from utils import display_chat
from utils import hide_default_sidebar

# Hide default Streamlit sidebar
hide_default_sidebar()

st.title("Machine Learning Course")
st.write("Select a topic below to start learning:")

topics = [
    "Introduction to Machine Learning",
    "Supervised Learning",
    "Unsupervised Learning",
    "Neural Networks",
    "Natural Language Processing",
    "Computer Vision"
]

# Initialize session state for topic if not set
if "selected_topic" not in st.session_state:
    st.session_state.selected_topic = None

# Display topics as buttons instead of links
for t in topics:
    if st.button(t):
        st.session_state.selected_topic = t  # Store the selected topic
        st.switch_page("pages/topic.py")  # Navigate to topic page

if st.button("Back to Main Page"):
    st.session_state.selected_topic = None  # Clear selected topic
    st.switch_page("app.py")  # Navigate back to main page

# Display area for answers
display_area = st.empty()
display_area.text_area("Responses", placeholder="Responses will appear here...", height=200, key="response_area", disabled=True)

# Display chat text window
display_chat(display_area)