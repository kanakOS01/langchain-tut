## integrate code with OpenAI API
import os
from constants import openai_key
from langchain_openai import ChatOpenAI
import streamlit as st

# streamlit code

st.title('LangChain using OpenAI API')
# taking user input
input_text = st.text_input('Search the topic you want')

## OPENAI LLMS
llm = ChatOpenAI(openai_api_key=openai_key)       # temperature defins how much control does the model has


if input_text:      # if user has given some input
    content = llm.invoke(input_text)
    st.write(content.content)