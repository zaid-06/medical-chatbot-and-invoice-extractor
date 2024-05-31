	
# # export GOOGLE_API_KEY=""/

from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain
# from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import os
# import google.generativeai as genai   
# import streamlit as st
from getpass import getpass
import os 


if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass("AIzaSyC_mSyMfY3zZZriaRrGKx1wiV0pspdWqw4")

# genai.configure(api_key= os.getenv("GOOGLE_API_KEY"))
llm = ChatGoogleGenerativeAI(model="gemini-pro")
# llm = genai.GenerativeModel('gemini-pro-vision')

import streamlit as st
from langchain.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

st.set_page_config(page_title="chat-bot")
st.header("Medical ChatBot")


embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Create the retrieval chain
# retriever = vectordb.as_retriever(search_kwargs={"k": 5})
persist_directory = "D:/AI-with-langchain/gemini/Models/MedicalModel"
retriever = Chroma(persist_directory=persist_directory, embedding_function=embeddings).as_retriever(search_kwargs={"k": 5})

template = """
You are a helpful AI assistant.
Answer based on the context provided. 
context: {context}
input: {input}
answer:
""" 

prompt = PromptTemplate.from_template(template)

combine_docs_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

input_text = st.text_input("Input prompt:")
submit_button = st.button("Run")

if submit_button:
    response = retrieval_chain.invoke({"input": input_text})
    st.subheader("The response is:")
    st.write(response["answer"])

