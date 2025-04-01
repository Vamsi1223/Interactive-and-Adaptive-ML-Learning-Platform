import streamlit as st
import base64

from utils import display_chat

from utils import hide_default_sidebar

# Hide default Streamlit sidebar
hide_default_sidebar()

# Function to encode image to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Paths to images
image_path = "assets/tredence_logo.png"
image_path2 = "assets/machine_learning.jpg"

# Encode the image
encoded_image = get_base64_image(image_path2)

# Main Page
st.image(image_path, width=300)
st.subheader('LEARN SEAMLESSLY')

# Clickable image for ML course
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

# Handle navigation
query_params = st.query_params
if query_params.get("page") == "ml_course":
    st.switch_page("pages/ml_course.py")

# Display area for answers
display_area = st.empty()
display_area.text_area("Responses", placeholder="Responses will appear here...", height=200, key="response_area", disabled=True)

# Display chat text window
display_chat(display_area)
