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

st.set_page_config(page_title="Adaptive ML Learning Pipeline", layout="wide")

st.title("🧠 Adaptive & Interactive ML Learning Pipeline")
st.write("Integrating all agents for a seamless learning experience.")

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
    mini_project_completed: bool
    next_step: str
    current_question: str
    user_answer: str
    is_correct: bool
    feedback: str

# Initialize LangGraph
graph = StateGraph(LearningState)

### **🔹 Supervisor Agent**
def supervisor(state: LearningState):
    # If all steps are completed, stop execution
    if state["quiz_completed"] and state["practical_completed"] and state["real_world_completed"] and state["mini_project_completed"]:
        state["next_step"] = "done"
        return state  # This prevents looping

    system_prompt = """You are an AI tutor supervising an ML learning process. 
    Learning steps:
    1. Theoretical Content (Quiz)
    2. Practical Coding Exercise
    3. Real-World Application
    4. Mini-Projects

    Based on the user's progress, determine the next step: 'quiz', 'practical', 'real_world', or 'mini_project'.
    """

    progress_summary = f"""User Progress:
    - Topic: {state['topic']}
    - Quiz Completed: {state['quiz_completed']}
    - Practical Completed: {state['practical_completed']}
    - Real-World Application Completed: {state['real_world_completed']}
    - Mini-Project Completed: {state['mini_project_completed']}
    """

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=progress_summary)
    ])

    state["next_step"] = response.content.strip().lower()
    return state

### **🔹 Question Generation Agent**
def generate_question(state: LearningState):
    question_type = state["next_step"]
    
    prompt = f"Generate a {question_type} question on {state['topic']}."
    response = llm.invoke([SystemMessage(content=prompt)])

    state["current_question"] = response.content.strip()
    return state

### **🔹 Validation & Feedback Agent**
def validate_answer(state: LearningState):
    validation_prompt = f"Evaluate this answer: {state['user_answer']} for the question: {state['current_question']}. Provide feedback."
    response = llm.invoke([SystemMessage(content=validation_prompt)])

    if "correct" in response.content.lower():
        state["is_correct"] = True
    else:
        state["is_correct"] = False

    state["feedback"] = response.content.strip()
    return state

### **🔹 Updating Progress**
def update_progress(state: LearningState):
    if state["is_correct"]:
        if state["next_step"] == "quiz":
            state["quiz_completed"] = True
        elif state["next_step"] == "practical":
            state["practical_completed"] = True
        elif state["next_step"] == "real_world":
            state["real_world_completed"] = True
        elif state["next_step"] == "mini_project":
            state["mini_project_completed"] = True

    # If learning is done, mark next_step as "done"
    if state["quiz_completed"] and state["practical_completed"] and state["real_world_completed"] and state["mini_project_completed"]:
        state["next_step"] = "done"

    return state

# Define Graph Workflow
graph.add_node("supervisor", supervisor)
graph.add_node("generate_question", generate_question)
graph.add_node("validate_answer", validate_answer)
graph.add_node("update_progress", update_progress)

graph.add_edge("supervisor", "generate_question")
graph.add_edge("generate_question", "validate_answer")
graph.add_edge("validate_answer", "update_progress")
# graph.add_edge("update_progress", "supervisor")  # Loop back to supervisor

graph.set_entry_point("supervisor")
# graph.set_finish_point("supervisor")
def finish_node(state: LearningState):
    return state

graph.add_node("finish", finish_node)

graph.add_edge("update_progress", "supervisor", condition=lambda s: s["next_step"] != "done")
graph.add_edge("update_progress", "finish", condition=lambda s: s["next_step"] == "done")

graph.set_finish_point("finish")  # Ensure the workflow stops at "finish"


workflow = graph.compile()

# Streamlit UI
st.subheader("📚 Select Learning Topic")

user_id = st.text_input("🆔 User ID:", "user_123")
topic = st.text_input("📖 ML Topic:", "Decision Trees")

quiz_completed = st.checkbox("✅ Quiz Completed")
practical_completed = st.checkbox("✅ Practical Completed")
real_world_completed = st.checkbox("✅ Real-World Application Completed")
mini_project_completed = st.checkbox("✅ Mini-Project Completed")

if st.button("🎯 Start Learning"):
    with st.spinner("Initializing Learning Pipeline..."):
        result = workflow.invoke({
            "user_id": user_id,
            "topic": topic,
            "quiz_completed": quiz_completed,
            "practical_completed": practical_completed,
            "real_world_completed": real_world_completed,
            "mini_project_completed": mini_project_completed,
            "next_step": "",
            "current_question": "",
            "user_answer": "",
            "is_correct": False,
            "feedback": ""
        })

    st.success(f"📌 Next Step: **{result['next_step'].capitalize()}**")
    st.subheader("💡 Question")
    st.write(result["current_question"])

    user_answer = st.text_area("✍️ Enter Your Answer:")
    if st.button("✅ Submit Answer"):
        result["user_answer"] = user_answer
        result = workflow.invoke(result)
        st.write(f"🔍 **Feedback:** {result['feedback']}")

st.markdown("---")
st.markdown("🚀 **Powered by LangChain, LangGraph, and Streamlit**")
