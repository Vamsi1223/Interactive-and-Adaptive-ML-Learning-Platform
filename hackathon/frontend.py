import streamlit as st
import base64

# Function to encode image to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Function to display chat text window and handle chat input
def display_chat(display_area):
    user_input = st.text_area("Chat with us", placeholder="Type your message here...", height=68)
    if st.button("Send"):
        if user_input:
            answer = f"Answer to your question: {user_input}"
            display_area.text_area("Responses", value=answer, height=100, placeholder="Responses will appear here...", key="response_area")

# Paths to images
image_path = "C:/Users/divya.agarwal/AppData/Roaming/JetBrains/PyCharmCE2024.2/scratches/tredence_logo.png"
image_path2 = "C:/Users/divya.agarwal/AppData/Roaming/JetBrains/PyCharmCE2024.2/scratches/machine_learning.jpg"

# Encode the image to base64
encoded_image = get_base64_image(image_path2)

# Custom CSS to push contents up
st.markdown(
    """
    <style>
    .main-content {
        margin-top: -200px;
    }
    .tile-container {
        display: flex;
        justify-content: flex-start;
        align-items: center;
        margin-top: 20px;
    }
    .tile {
        width: 250px;
        height: auto;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease-in-out;
    }
    .tile:hover {
        transform: scale(1.05);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Get query parameters
query_params = st.query_params
page = query_params.get("page", "main")
topic = query_params.get("topic", None)

with st.container():
    st.markdown('<div class="main-content">', unsafe_allow_html=True)

    if page == "main":
        st.image(image_path, width=300)
        st.subheader('LEARN SEAMLESSLY')

        # Clickable image for navigation
        st.markdown(
            f"""
            <div class="tile-container">
                <a href="?page=ml_course" target="_self">
                    <img class="tile" src="data:image/png;base64,{encoded_image}" alt="Tile Image">
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )

    elif page == "ml_course" and topic:
        st.title(f"{topic.replace('_', ' ')}")
        st.write(f"Welcome to the {topic.replace('_', ' ')} page!")

        # Back button
        if st.button("Back to Machine Learning Course"):
            st.query_params.clear()
            st.query_params.update(page="ml_course")
            st.rerun()

    elif page == "ml_course":
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

        for t in topics:
            st.markdown(f"- [**{t}**](?page=ml_course&topic={t.replace(' ', '_')})")

        # Back button
        if st.button("Back to Main Page"):
            st.query_params.update(page="main")
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# Display area for answers
display_area = st.empty()
display_area.text_area("Responses", placeholder="Responses will appear here...", height=200, key="response_area", disabled=True)

# Display chat text window
display_chat(display_area)