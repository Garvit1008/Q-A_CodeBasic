import streamlit as st
import streamlit.components.v1 as components
from langchain_helper import  create_vector_db, get_qa_chain
st.title("CodeBasics QA 🙋‍♂️")
btn = st.button("create knowledge")
if btn:
    pass
question = st.text_input("question: ")
if question:
    chain = get_qa_chain()
    response = chain(question)
    st.header("Answer: ")
    st.write(response["result"])