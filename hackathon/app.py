import time

import streamlit as st

# from openai import OpenAI

from groq import Groq

from langchain_groq import ChatGroq

import os

from dotenv import load_dotenv
 
load_dotenv()
 
# Set page configuration and initialize OpenAI API key

st.set_page_config(

    page_title="ML Learning Assistant", 

    page_icon="🧑‍🏫", layout="centered", 

    initial_sidebar_state="expanded", 

)
 
st.title("ChatGPT-like clone")

st.info("Learn Theory and Practice of Machine Learning", icon="📃")

client = Groq(api_key=os.environ["GROQ_API_KEY"])
 
if "model_name" not in st.session_state:

    st.session_state["model_name"] = "llama-3.3-70b-versatile" #"gpt-3.5-turbo"
 
if "messages" not in st.session_state:

    st.session_state.messages = []
 
for message in st.session_state.messages:

    with st.chat_message(message["role"], avatar="user1.png" if message["role"]=="user" else "bot.png"):

        st.markdown(message["content"])
 
if prompt := st.chat_input("Ask a question?"):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar="user1.png"):

        st.markdown(prompt)
 
    with st.chat_message("assistant", avatar="bot.png"):

        with st.spinner("Thinking..."):

            time.sleep(1)

            try:

                stream = client.chat.completions.create(

                    model=st.session_state["model_name"],

                    messages=[

                        {"role": m["role"], "content": m["content"]}

                        for m in st.session_state.messages

                    ],

                    stream=True,

                )
 
                def stream_data():

                    for chunks in stream:

                        response = chunks.choices[0].delta.content

                        yield response if response else ""
 
                response = st.write_stream(stream_data)

                # print("Response: ", response, "\nType: ", type(response))

                st.session_state.messages.append(

                    {"role": "assistant", "content": response}

                )

            except Exception as e:

                print("Error generating response: ", e)

                st.session_state.max_messages = len(st.session_state.messages)

                rate_limit_message = f"""

                    Oops! Sorry, I can't talk now. Too many people have used

                    this service recently. Error: {e}

                """

                st.session_state.messages.append(

                    {"role": "assistant", "content": rate_limit_message}

                )

                # st.rerun()

 