import streamlit as st
import os
import time

from utils import display_chat
from topics_data import topics

# Main application
def main():
    # --- Page Configuration ---
    st.set_page_config(layout='wide', page_title='AskWise', page_icon=':blue_book:')
    hide_sidebar_style = """
       <style>
           [data-testid="stSidebarNav"] {
               display: none;
           }
       </style>
   """

    st.markdown(hide_sidebar_style, unsafe_allow_html=True)
    # --- Left Sidebar (Navigation) ---
    with st.sidebar:
        st.image("static/ask_wise_logo.png", width=170)
        st.markdown("---")
        st.header("Courses")

        # Define the courses
        courses = ["Machine Learning", "Deep Learning", "LLM"]

        # Initialize the default selected course in session_state
        if "selected_course" not in st.session_state:
            st.session_state.selected_course = courses[0]  # Default selection

        # Create a button/tile for each course in a horizontal layout
        for i, course in enumerate(courses):
            if st.button(course):
                st.session_state.selected_course = course  # Update selected course

    col1, col2 = st.columns([4, 2])

    with col1:
        st.subheader(f"{st.session_state.selected_course} Topics")
        courses_topics = {
            "Machine Learning": topics,
            "Deep Learning": ["Courses yet to be added"],
            "LLM": ["Courses yet to be added"]
        }

        if "selected_topic" not in st.session_state:
            st.session_state.selected_topic = None

        if st.session_state.selected_course in courses_topics:
            for topic in courses_topics[st.session_state.selected_course]:
                if topic != "Courses yet to be added":
                    if st.button(topic):  # Dynamic topic selection
                        st.session_state.selected_topic = topic
                        st.switch_page("pages/topic.py")
                else:
                    st.write(topic)

    with col2:
        st.markdown("<h5> Ask me anything </h5>", unsafe_allow_html=True)
        display_chat()


# Run the Streamlit app
if __name__ == "__main__":
    main()
