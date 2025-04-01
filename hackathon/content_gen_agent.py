import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langgraph.graph import StateGraph
from typing import TypedDict
import os
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

# Load environment variables
load_dotenv()

st.set_page_config(page_title="ML Learning Resource Recommender", layout="centered")

st.title("🔍 AI-Powered ML Resource Recommendation")
st.write("Get personalized learning resources based on your skill level and topic.")

# Azure AI Search credentials
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX")

if not AZURE_SEARCH_ENDPOINT or not AZURE_SEARCH_KEY or not AZURE_SEARCH_INDEX:
    st.error("Please set up Azure AI Search credentials in the environment variables.")
    st.stop()

# Initialize Azure AI Search client
search_client = SearchClient(endpoint=AZURE_SEARCH_ENDPOINT, 
                             index_name=AZURE_SEARCH_INDEX, 
                             credential=AzureKeyCredential(AZURE_SEARCH_KEY))

# Define LLM
llm = ChatOpenAI(model_name="gpt-4", temperature=0.0)

# Define state for LangGraph
class ResourceRecommendationState(TypedDict):
    skill: str
    expertise: str
    topic: str
    recommendations: list

# LangGraph workflow
graph = StateGraph(ResourceRecommendationState)

def fetch_resources(state: ResourceRecommendationState):
    query = f"{state['topic']} {state['expertise']} {state['skill']}"
    results = search_client.search(search_text=query, top=5)
    
    recommendations = [doc['content'] for doc in results]
    state["recommendations"] = recommendations
    return state

def format_response(state: ResourceRecommendationState):
    formatted_response = "\n".join([f"📌 {rec}" for rec in state["recommendations"]])
    state["formatted_response"] = formatted_response
    return state

graph.add_node("fetch_resources", fetch_resources)
graph.add_node("format_response", format_response)

graph.add_edge("fetch_resources", "format_response")

graph.set_entry_point("fetch_resources")
graph.set_finish_point("format_response")

workflow = graph.compile()

# Streamlit UI
skill = st.selectbox("📖 Select Your Skill:", ["Machine Learning", "Deep Learning", "NLP", "Computer Vision"])
expertise = st.selectbox("🎯 Select Your Expertise Level:", ["Beginner", "Intermediate", "Advanced"])
topic = st.text_input("📝 Enter a Specific Topic:")

if st.button("🔍 Get Recommendations"):
    if skill and expertise and topic:
        with st.spinner("Fetching resources..."):
            state = {"skill": skill, "expertise": expertise, "topic": topic}
            result = workflow.invoke(state)
        
        st.success("🎯 Here are your recommended resources:")
        st.markdown(result["formatted_response"])
    else:
        st.warning("⚠️ Please fill in all fields.")

st.markdown("---")
st.markdown("🤖 Built with **LangGraph, Azure AI Search, and Streamlit**")
