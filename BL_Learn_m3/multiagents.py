from ast import List
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from typing import Sequence, TypedDict, Annotated, Literal, Optional
import operator
import sqlite3
import json
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = "./ml_resources_index"
model_name = "llama-3.3-70b-versatile"

# Load the FAISS index safely
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vector_db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)  # ✅ Fix applied
retriever = vector_db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.8})  # Convert FAISS to retriever, can be "mmr"

# Ensure FAISS contains documents
if not vector_db.docstore._dict:
    raise ValueError("Error: The FAISS index is empty! Ensure 'index.faiss' and 'index.pkl' exist.")

def _parse_llm_response(response_text):
    """Parse the LLM response and convert it to a proper dictionary"""
    try:            
        clean_text = response_text.strip('`').strip('json').strip('python').strip()
        if clean_text.startswith('{'):
            return json.loads(clean_text)
    except json.JSONDecodeError as e:
        print(f"Error parsing response: {e}")
        return None

content_generation_prompt ='''
Role:
You are an advanced Machine Learning Content Generation Agent. Your task is to generate high-quality educational content based on a given {topic} and {page_name}. The generated content should align precisely with the learning objective of the specified {page_name}.
Generate more diverse content realted to the 'page name' provided.

Content Generation Guidelines:

1. If {page_name} = "Theoretical Learning":

Generate clear, well-structured, and in-depth theoretical content.
The content should provide a strong conceptual understanding of the {topic}.
Use precise definitions, explanations, key principles, mathematical formulations (if applicable), and real-world relevance.
Maintain a formal, academic tone, making it accessible for learners at various levels.

2. If {page_name} = "Practical Learning":

Generate hands-on, application-driven content focused on implementing the {topic}.
Explain how to apply the concept in real-world scenarios using step-by-step guidance.
Include code snippets (if applicable), practical use cases, best practices, and troubleshooting insights.
Ensure the content is actionable, concise, and solution-oriented to facilitate hands-on learning.

Note: Return the content str as a string **enclosed in double quotes (" ") and must be in a single line or use explicit newline characters (\n)** wherever necessary. Do not multi-line strings using triple quotes (""" """).
Strictly Return only a **valid JSON objects** with the following structure as below, also do not include the markdown keywords like '`' or 'json':
{{
'content': str
}}
'''

question_generation_prompt = """
Role:
You are an AI-powered question generation agent specializing in Machine Learning. Your goal is to generate structured question based on user , 
tailored to the page name: {page_name} in the topic of {topic}. Questions should progressively increase in difficulty where applicable.

Question Generation Guidelines:
1. Theoretical & Practical Quizzes
If {page_name} = "Theory Quiz" or "Practical Quiz":

Generate a list of multiple-choice questions (MCQs) in increasing difficulty (easy → medium → hard).

"Theory Quiz" should focus on conceptual knowledge, while "Practical Quiz" should assess real-world application.

Each question must have four options, separated by |, with a clearly defined "correct_answer".

Example Output:
{{
    "question": "What is the main purpose of the activation function in neural networks?",
    "question_type": "mcq",
    "options": "Introduce non-linearity|Reduce model complexity|Increase training speed|Adjust learning rate",
    "correct_answer": "Introduce non-linearity"
}}

2. Real-Time Hands-on (Code Completion)
If {page_name} = "Real-Time Hands-on":

Generate a coding question requiring the user to fill in missing code lines related to {topic}.

Use "ADD_CODE_HERE" as a placeholder for missing parts.

Ensure missing lines focus on key learning points but still provide enough contextual hints.

The "correct_answer" should list only the missing lines, separated by |.

Set "options": "None".

Example Output:

{{
    "question": "Complete the missing parts of this logistic regression training script:\\n\\n\
    ```python\\n\
    from sklearn.linear_model import LogisticRegression\\n\
    from sklearn.model_selection import train_test_split\\n\
    from sklearn.metrics import accuracy_score\\n\\n\
    X_train, X_test, y_train, y_test = ADD_CODE_HERE\\n\\n\
    model = LogisticRegression()\\n\
    ADD_CODE_HERE\\n\
    y_pred = model.predict(X_test)\\n\
    print('Accuracy:', accuracy_score(y_test, y_pred))",
    "question_type": "code-snippet",
    "options": "None",
    "correct_answer": "train_test_split(X, y, test_size=0.2, random_state=42)|model.fit(X_train, y_train)"
}}

3. Mini Project (Comprehensive Task)
If {page_name} = "Mini Project":

Provide a detailed problem statement that clearly defines the project's goal.

Include relevant dataset links for the user to work with.

Provide references or helpful resources to guide the user.

The "correct_answer" and "options" field should be "None" since this is an open-ended project. The "question_type" must be "mini-project".

Example Output:

{{
    "question": "Develop a machine learning model to predict house prices based on features such as square footage, number of bedrooms, and location. Your task includes:\\n\
    - Data preprocessing (handling missing values, feature scaling).\\n\
    - Model selection (Linear Regression, Decision Tree, etc.).\\n\
    - Hyperparameter tuning and performance evaluation.\\n\
    - Visualizing important features that impact house prices.\\n\\n\
    **Dataset:** [House Prices Dataset](https://www.kaggle.com/datasets/housingsales)\\n\
    **Evaluation Metrics:** RMSE, R-squared\\n\
    **References:**\\n\
    - [Scikit-Learn Regression Guide](https://scikit-learn.org/stable/modules/linear_model.html)\\n\
    - [Feature Engineering for ML](https://towardsdatascience.com/feature-engineering)"
    "question_type": "mini-project",
    "options": "None",
    "correct_answer": "None"
}}

Output Format
You only return one question based on above instructions.
Strictly return only a **valid JSON objects** with the following structure, also do not include markdown keywords like '`' or 'json' in the response:
{{
    "question": str,
    "question_type": str,
    "options": str,
    "correct_answer": str
}}
"""

validation_feedback_prompt = """
Role:  
You are an AI-powered evaluation agent responsible for validating user responses and providing constructive feedback. Given the input:  
{{'question': {question}, "options": {options}, 'correct_answer': {correct_answer}, 'user_input': {user_input}, 'page_name': {page_name} }},  
you must verify whether the user’s answer is correct and generate appropriate feedback.

### **Evaluation Guidelines based on page name:**

1. **Theory Quiz:**  
   - The 'options' are | seperated options as it is a MCQ question.
   - Compare `user_input` with `correct_answer`.  
   - If they match, set `"is_correct": "Correct"` and provide a brief reinforcement in `"feedback"`.  
   - If incorrect, set `"is_correct": "Incorrect"` and provide a **guiding hint** in `"feedback"` without revealing the correct answer.  
   - **Hint Example:** Instead of saying *"The correct answer is X"*, say *"Think about how neural networks introduce non-linearity. Which function is responsible for this?"*  

2. **Practical Quiz:**  
    - The 'options' are | seperated options as it is a MCQ question.
   - Follow the same validation approach as "Theory Quiz".  
   - Ensure `"feedback"` includes a hint based on real-world application, helping the user understand the logic without revealing the answer.  

3. **Mini Project:**  
   - Analyze `user_input` in depth.
   - Provide constructive feedback in `"feedback"`, highlighting strengths and improvement areas.  
   - Offer **suggestions** rather than direct answers, guiding the user towards refining their project approach.  
   - Since it's open-ended, `"is_correct"` should **not** be used.  

4. **Real-Time Hands-on:**  
   - Check if all missing code parts in `user_input` match `correct_answer`.  
   - If correct, set `"is_correct": "Correct"` with an encouraging `"feedback"`.  
   - If incorrect, set `"is_correct": "Incorrect"` and provide **clues about the logic or syntax** without revealing the missing code directly.  
   - **Hint Example:** Instead of saying *"You missed `train_test_split(X, y, test_size=0.2, random_state=42)`"*, say *"Make sure you correctly split the dataset before training. Which function from `sklearn.model_selection` helps achieve this?"*  

**Output Format:** 
Strictly Return only a **valid JSON objects** with the following structure as below, also do not include the markdown keywords like '`' or 'json':
{{
    "is_correct": str, 
    "feedback":str
}}
"""


class AgentState(TypedDict):
    topic: str
    page_name: str
    content: Annotated[list, operator.add]
    pages_completed: Annotated[list, operator.add]
    question: Annotated[list, operator.add]
    question_type: str
    options: str
    correct_answer: Annotated[list, operator.add]
    user_input: Annotated[list, operator.add]
    feedback: str
    is_correct: str

sqlite_conn = sqlite3.connect("graph_persistence.db", check_same_thread=False)
memory = SqliteSaver(sqlite_conn)

def content_generation_agent(state: AgentState):
    """Fetches theoretical/practical content using a vector DB."""

    page_name = state.get("page_name", "Theoretical Learning")
    topic = state.get("topic", "Linear Regression")

    retrieved_documents = vector_db.similarity_search(topic, k=5)
    retrieved_content = "\n".join([doc.page_content for doc in retrieved_documents])
    content_prompt = content_generation_prompt.format(topic=topic, page_name=page_name, content = retrieved_content)

    # Define LLM Model
    llm = ChatGroq(model_name=model_name, streaming=False, temperature=0.7)

    response = llm.invoke([
        HumanMessage(content=content_prompt)
    ])
    response = response.content
    # print("Content generation before parsing:", response)

    response = _parse_llm_response(response)
    # print("Content generation response:", response)
    # print("Content generation response type:", type(response))
    
    return {"content": [response.get("content", "")]}

def question_generation_agent(state: AgentState):
    learned_content = state.get("content", "")[-1]
    topic = state.get("topic", "Linear Regression")
    page_name = state.get("page_name", "Theoretical Learning")

    question_prompt = question_generation_prompt.format(topic=topic, page_name=page_name, content=learned_content)
    # Define LLM model
    llm = ChatGroq( model_name=model_name, streaming=False, temperature=0.75)

    response = llm.invoke([
        HumanMessage(content=question_prompt)
    ])
    
    response = response.content
    response = _parse_llm_response(response)

    return {"question": [response.get('question')], 
            "question_type": response.get('question_type'), 
            "options": response.get('options'),
            "correct_answer": [response.get('correct_answer')]
            }

def validation_feedback_agent(state: AgentState):
    """Validates answers, provides hints, and tracks progress."""
    
    page_name = state.get('page_name')
    user_input = state.get('user_input')[-1]
    correct_answer = state.get('correct_answer')[-1]
    question = state.get('question')[-1]
    options = state.get('options')

    validation_prompt = validation_feedback_prompt.format(page_name=page_name, question=question, options=options, correct_answer=correct_answer, user_input=user_input)

    # Define LLM Model
    llm = ChatGroq(model_name=model_name, streaming=False, temperature=0.7)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    response = llm.invoke([
        HumanMessage(content=validation_prompt)
    ])

    response = response.content
    response = _parse_llm_response(response)

    return {
        "is_correct": response.get("is_correct", "Incorrect").strip(),
        "feedback": response.get("feedback", "No Feedback").strip()
    }
 
# Graph Definition
graph = StateGraph(AgentState)

# graph.add_node("Supervisor Agent", supervisor_agent)
graph.add_node("Content Generation", content_generation_agent)
graph.add_node("Question Generation", question_generation_agent)
graph.add_node("Validation & Feedback", validation_feedback_agent)

# graph.add_edge("Supervisor Agent", "content_generation")
# graph.add_edge("content_generation", "question_generation")
# graph.add_edge("question_generation", "validation_feedback")
# graph.add_edge("validation_feedback", "supervisor_agent")

def validation_feedback_routing(state: AgentState):
    is_correct = state.get("is_correct", "Incorrect").lower().strip()
    page_name = state.get("page_name", "Theoretical Learning")
    if is_correct == "correct":
        if page_name in ["Theoretical Learning", "Practical Learning"]:
            return "Content Generation"
        elif page_name in ["Theory Quiz", "Practical Quiz", "Real-Time Hands-on"]:
            return "Question Generation"
        elif page_name in ["Mini Project"]:
            return END
    elif is_correct == "incorrect":
        return "Validation & Feedback"


def supervisor_routing_function(state: AgentState):
    page_name = state.get("page_name", "Theoretical Learning")
    if page_name in ["Theoretical Learning", "Practical Learning"]:
        return "Content Generation"
    elif page_name in ["Theory Quiz", "Practical Quiz", "Real-Time Hands-on", "Mini Project"]:
        return "Question Generation"
    elif page_name in "":
        return "Validation & Feedback"

# graph.add_edge(START, "Supervisor Agent")
graph.add_conditional_edges(START, supervisor_routing_function)
graph.add_edge("Content Generation", "Question Generation")
graph.add_edge("Question Generation", "Validation & Feedback")
graph.add_conditional_edges("Validation & Feedback", validation_feedback_routing)

lang_graph = graph.compile(checkpointer=memory, interrupt_after=["Content Generation", "Question Generation", "Validation & Feedback"])