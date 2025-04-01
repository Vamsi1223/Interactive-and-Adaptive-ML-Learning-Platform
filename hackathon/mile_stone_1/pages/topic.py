import streamlit as st
from utils import display_chat
from utils import hide_default_sidebar

# Hide default Streamlit sidebar
hide_default_sidebar()

# Get selected topic from session state
if "selected_topic" in st.session_state and st.session_state.selected_topic:
    topic = st.session_state.selected_topic
else:
    topic = "Unknown Topic"

st.title(topic)
st.write(f"Welcome to the {topic} page!")

# Back button to Machine Learning Course
if st.button("Back to Machine Learning Course"):
    st.switch_page("pages/ml_course.py")

# Include chat component
# Display area for answers
display_area = st.empty()
display_area.text_area("Responses", placeholder="Responses will appear here...", height=200, key="response_area", disabled=True)

# Display chat text window
display_chat(display_area)
