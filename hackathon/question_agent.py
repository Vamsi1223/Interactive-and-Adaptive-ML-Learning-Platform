import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langgraph.graph import StateGraph
from typing import TypedDict
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

st.set_page_config(page_title="Question Generation Agent", layout="centered")

st.title("Question Generation Agent")
st.write("Generate ML-related questions based on topic, difficulty, and type.")

# API Key
api_key = st.sidebar.text_input("🔑 Enter Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))
if not api_key:
    st.error("Please enter your Groq API key in the sidebar.")
    st.stop()

# Define LLM model
llm = ChatGroq(groq_api_key=api_key, model_name="deepseek-r1-distill-llama-70b", temperature=0.7)

# Define Question Generation State
class QuestionGenState(TypedDict):
    topic: str
    difficulty: str
    question_type: str
    generated_question: str

# LangGraph for Question Generation
graph = StateGraph(QuestionGenState)

def generate_question(state: QuestionGenState):
    system_prompt = """
    You are an AI question generator for ML learning. Given a topic, difficulty level, and question type, generate an appropriate question.
    Types:
    - Conceptual: Theory-based questions.
    - Coding: Code implementation-related questions.
    - Application: Real-world ML application-based questions.
    """
    
    user_prompt = f"""
    Topic: {state['topic']}
    Difficulty: {state['difficulty']}
    Question Type: {state['question_type']}
    """
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    
    state["generated_question"] = response.content
    return state

graph.add_node("generate_question", generate_question)
graph.set_entry_point("generate_question")
graph.set_finish_point("generate_question")

workflow = graph.compile()

# Streamlit UI
topic = st.text_input("📌 Enter ML Topic:")
difficulty = st.selectbox("🎚️ Select Difficulty:", ["Beginner", "Intermediate", "Advanced"])
question_type = st.selectbox("📘 Select Question Type:", ["Conceptual", "Coding", "Application"])

if st.button("🔎 Generate Question"):
    if topic.strip():
        with st.spinner("Generating question..."):
            state = {"topic": topic, "difficulty": difficulty, "question_type": question_type}
            result = workflow.invoke(state)
            
            st.success("✅ Question Generated!")
            st.write(f"**Question:** {result['generated_question']}")
    else:
        st.warning("⚠️ Please enter a topic.")

st.markdown("---")
st.markdown("👨‍💻 Built with **LangChain, LangGraph, and Streamlit**")
