from typing import Set

import streamlit as st
from streamlit_chat import message
from qa.retrieval_qa import retrieval_qa

st.header("LLM Chatbot")

prompt = st.text_input("Prompt", placeholder="Enter your promt here...")

if ("user_prompt_history" not in st.session_state 
    and "chat_answers_history" not in st.session_state
):
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_answers_history"] = []

if prompt:
    with st.spinner("Generating response..."):
        generated_response = retrieval_qa(query=prompt)

    answer = generated_response['answer']

    st.session_state["user_prompt_history"].append(prompt)
    st.session_state["chat_answers_history"].append(answer)

if st.session_state["chat_answers_history"]:
    for answer, user_query in zip(st.session_state["chat_answers_history"], st.session_state["user_prompt_history"]):
        message(answer)
        message(user_query, is_user=True)