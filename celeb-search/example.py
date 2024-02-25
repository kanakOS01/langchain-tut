## integrate code with OpenAI API
import os
from constants import openai_key
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

# streamlit code

st.title('Celebrity Search Results')
celeb_input = st.text_input('Enter the name of any celebrity')

## OPENAI LLMS
llm = ChatOpenAI(openai_api_key=openai_key)      

## PROMPT TEMPLATES
prompt = ChatPromptTemplate.from_messages([
    ("system", "In 20 words tell me about the celebrity"),
    ("user", "{input}")
])

chain = prompt | llm

# celeb_input = input()
# content = chain.invoke({"input": {celeb_input}})
# print(content.content)

if celeb_input:
    content = chain.invoke({"input": {celeb_input}})
    st.write(content.content)