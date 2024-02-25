import os
from constants import openai_key
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st

os.environ['OPENAI_API_KEY'] = openai_key

st.title("Celeb Search")
user_input = st.text_input("Enter name of a celebrity")

## PROMPT
prompt = PromptTemplate(
    input_variables=['name'], 
    template="Tell me about the celebrity {name}"
)

## OPENAI LLM
llm = OpenAI(temperature=0.8)

chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

if user_input:
    content = chain.run(user_input)     # the name fiels in the prompt will take the value of user_input
    st.write(content)