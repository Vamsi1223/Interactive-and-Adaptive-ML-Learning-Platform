import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langgraph.graph import StateGraph
from langchain.memory import ConversationBufferMemory
from typing import TypedDict
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

st.set_page_config(page_title="ML Learning Supervisor", layout="centered")

st.title("ML Learning Supervisor Agent")
st.write("Manages your learning progress and directs you to the next step.")

api_key = st.sidebar.text_input("🔑 Enter Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))
if not api_key:
    st.error("Please enter your Groq API key in the sidebar.")
    st.stop()

# Define LLM Model
llm = ChatGroq(groq_api_key=api_key, model_name="deepseek-r1-distill-llama-70b", streaming=True, temperature=0.2)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define Learning Progress State
class LearningState(TypedDict):
    user_id: str
    topic: str
    quiz_completed: bool
    practical_completed: bool
    real_world_completed: bool
    next_step: str

# LangGraph for Supervisor Agent
graph = StateGraph(LearningState)

def check_progress(state: LearningState):
    """
    Determines the next step based on the user's progress.
    """
    system_prompt = """You are an AI tutor supervising a structured ML learning process. 
    The learning flow consists of:
    1. Theoretical Content (with MCQ & Open-ended Quiz)
    2. Practical Coding Exercise
    3. Real-World Application Question
    4. Mini-Projects (Beginner, Intermediate, Advanced)
    
    Given the user’s progress, determine the next step. Ensure they complete each stage before moving forward.
    Respond with: 'quiz', 'practical', 'real_world', or 'mini_project' as the next step.
    """
    
    progress_summary = f"""User Progress:
    - Topic: {state['topic']}
    - Quiz Completed: {state['quiz_completed']}
    - Practical Completed: {state['practical_completed']}
    - Real-World Question Completed: {state['real_world_completed']}
    """
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=progress_summary)
    ])
    
    state["next_step"] = response.content.strip().lower()
    return state

graph.add_node("check_progress", check_progress)
graph.set_entry_point("check_progress")
graph.set_finish_point("check_progress")

workflow = graph.compile()

# Streamlit UI
st.subheader("🔍 Track Your Learning Progress")

user_id = st.text_input("🆔 Enter User ID:", "user_123")
topic = st.text_input("📖 ML Topic:", "Decision Trees")
quiz_completed = st.checkbox("✅ Quiz Completed")
practical_completed = st.checkbox("✅ Practical Completed")
real_world_completed = st.checkbox("✅ Real-World Application Completed")

if st.button("🔄 Check Next Step"):
    with st.spinner("Analyzing progress..."):
        result = workflow.invoke({
            "user_id": user_id,
            "topic": topic,
            "quiz_completed": quiz_completed,
            "practical_completed": practical_completed,
            "real_world_completed": real_world_completed
        })

    st.success(f"📌 Next Step: **{result['next_step'].capitalize()}**")

st.markdown("---")
st.markdown("👨‍💻 Built with **LangChain, LangGraph, and Streamlit**")
