import streamlit as st
from utils import display_chat, hide_default_sidebar
from topics_data import topic_outlines  # Import the dynamic topic outlines
from multiagents import lang_graph

# Get selected topic from session state
topic = st.session_state.get("selected_topic", "Unknown Topic")

# Sidebar for topic outline
with st.sidebar:
  st.title(f"{topic} Outline")

  # Display the common structure for the selected topic
  if topic in topic_outlines:
      for section in topic_outlines[topic]:
          if st.button(section):
              st.session_state["selected_section"] = section

# Main content area
st.title(topic)

col1, col2 = st.columns([4, 2])

# Thread
thread = {"configurable": {"thread_id": "1"}}

with col1:
  selected_section = st.session_state.get("selected_section", None)
  page_name = st.session_state.get("selected_section", None)

  if selected_section:
     st.subheader(selected_section)
     if page_name == "Theoretical Learning":
        input = {"topic": topic, "page_name": page_name}
        # Run the graph until the first interruption
        for response in lang_graph.stream(input, thread):
          for key, value in response.items():
              if key == "Content Generation":
                theory_content = value["content"][-1]
        st.write(theory_content)
      
     elif page_name == "Practical Learning":
        input = {"topic": topic, "page_name": page_name}
        # Run the graph until the first interruption
        for response in lang_graph.stream(input, thread):
          for key, value in response.items():
              if key == "Content Generation":
                practical_content = value["content"][-1]
        st.write(practical_content)

     elif page_name == "Theory Quiz":
        input = {"topic": topic, "page_name": page_name}
        # Run the graph until the first interruption
        for response in lang_graph.stream(input, thread):
          for key, value in response.items():
              if key == "Question Generation":
                question = value["question"][-1]
                options = value["options"]
                correct_answer = value["correct_answer"][-1]

        options = options.split("|")
        success_msg = "Correct answer"
        error_msg = "Incorrect answer"
        question_placeholder = st.empty()
        options_placeholder = st.empty()
        results_placeholder = st.empty()
        question_placeholder.markdown(f"**Question: ** {question}")
        # track the user selection
        options_placeholder.radio("", options, index=1, key=f"{page_name} Question")
        selected_option = st.selectbox("Select the correct answer:", options)
        st.write(f"📘 Content for {selected_section} will be dynamically generated here.")

     elif page_name == "Practical Quiz":
        st.write(f"📘 Content for {selected_section} will be dynamically generated here.")

     elif page_name == "Real-Time Hands-on":
        st.write(f"📘 Content for {selected_section} will be dynamically generated here.")

     elif page_name == "Mini Project":
        st.write(f"📘 Content for {selected_section} will be dynamically generated here.")

# Chat container
with col2:
  st.markdown("<h5> Ask me anything </h5>", unsafe_allow_html=True)
  display_chat()

st.markdown("</div>", unsafe_allow_html=True)


