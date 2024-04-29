from langchain_helper import get_few_shot_db_chain

import streamlit as st

st.title("T-Shirt Shop: Database Q&A ")

question = st.text_input("Question: ")

if question:
    chain = get_few_shot_db_chain()
    answer = chain.invoke(question)
    st.header("Answer:")
    st.write(answer['result'])
