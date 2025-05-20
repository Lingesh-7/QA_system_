import streamlit as st
import langchain_helper

st.title("QA system for E-learning Company")


question=st.text_input("Question")
if question:
    # pass
    chain=langchain_helper.get_qa_chain()
    resposne=chain(question)

    st.header("Answer")
    st.write(resposne['result'])