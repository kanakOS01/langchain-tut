import os
from constants import openai_key
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory
import streamlit as st

os.environ['OPENAI_API_KEY'] = openai_key

st.title("Celeb Search")
user_input = st.text_input("Enter name of a celebrity")


## MEMORY -- to store the result in the llm chain
person_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
dob_memory = ConversationBufferMemory(input_key='person', memory_key='chat_history')
event_memory = ConversationBufferMemory(input_key='dob', memory_key='chat_history')


## OPENAI LLM
llm = OpenAI(temperature=0.8)


## PROMPT 1
prompt1 = PromptTemplate(
    input_variables=['name'], 
    template="Tell me about the celebrity {name} in 50 words"
)

chain1 = LLMChain(llm=llm, prompt=prompt1, verbose=True, output_key='person', memory=person_memory)


## PROMPT 2
prompt2 = PromptTemplate(
    input_variables=['person'], 
    template="When was {person} born"
)

chain2 = LLMChain(llm=llm, prompt=prompt2, verbose=True, output_key='dob', memory=dob_memory)


## PROMPT 3
prompt3 = PromptTemplate(
    input_variables=['dob'], 
    template="Mention 5 things that happened around {dob}"
)

chain3 = LLMChain(llm=llm, prompt=prompt3, verbose=True, output_key='events', memory=event_memory)

# in SimpleSequentialChain the chains are executed linearly, one after the other
# it only shows the final output
# parent_chain = SimpleSequentialChain(chains=[chain1, chain2], verbose=True)

# if we want to show all the output then we can use SequentialChain
parent_chain = SequentialChain(
    chains=[chain1, chain2,chain3], 
    input_variables=['name'], 
    output_variables=['person', 'dob', 'events'], 
    verbose=True
)


if user_input:
    # content = parent_chain.run(user_input)     # the name fiels in the prompt will take the value of user_input
    content = parent_chain({'name':{user_input}})
    st.write(content)

    with st.expander('Person name'):
        st.info(person_memory.buffer)

    with st.expander('Major events'):
        st.info(event_memory.buffer)