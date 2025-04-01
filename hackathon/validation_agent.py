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

st.set_page_config(page_title="ML Question Validation", layout="centered")

st.title("ML Question Validation & Feedback")
st.write("Submit your answer and get instant validation along with feedback.")

api_key = st.sidebar.text_input("🔑 Enter Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))
if not api_key:
    st.error("Please enter your Groq API key in the sidebar.")
    st.stop()

# Define LLM Model
llm = ChatGroq(groq_api_key=api_key, model_name="deepseek-r1-distill-llama-70b", streaming=True, temperature=0.2)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define ML Validation State
class MLValidationState(TypedDict):
    question: str
    user_answer: str
    llm_feedback: str
    is_correct: bool

# LangGraph for Validation & Feedback
graph = StateGraph(MLValidationState)

def validate_answer(state: MLValidationState):
    system_prompt = """You are an expert ML tutor. Evaluate the given answer based on correctness and clarity.
    Provide constructive feedback. If the answer is correct, confirm it; if not, guide the user toward the right answer.
    Respond with 'Correct' or 'Incorrect' followed by detailed feedback.
    """
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Question: {state['question']}\nAnswer: {state['user_answer']}")
    ])
    
    state["llm_feedback"] = response.content
    state["is_correct"] = "Correct" in response.content
    return state

graph.add_node("validate_answer", validate_answer)
graph.set_entry_point("validate_answer")
graph.set_finish_point("validate_answer")

workflow = graph.compile()

# Streamlit UI
question = st.text_area("📌 ML Question:", "What is the difference between bias and variance in ML?", height=100)
user_answer = st.text_area("📝 Your Answer:", height=150)

if st.button("🔍 Validate Answer"):
    if user_answer.strip():
        with st.spinner("Evaluating..."):
            result = workflow.invoke({"question": question, "user_answer": user_answer})
        
        if result["is_correct"]:
            st.success("✅ Correct Answer!")
        else:
            st.error("❌ Incorrect Answer!")
        
        with st.expander("📖 Detailed Feedback"):
            st.write(result["llm_feedback"])
    else:
        st.warning("⚠️ Please enter an answer.")

st.markdown("---")
st.markdown("👨‍💻 Built with **LangChain, LangGraph, and Streamlit**")
