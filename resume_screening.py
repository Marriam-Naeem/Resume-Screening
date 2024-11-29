import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import Chroma 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")

embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.1,
    max_tokens=None,
    timeout=None,
    max_retries=5,)


def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    vector_store = Chroma.from_texts(texts= text_chunks, embedding=embeddings,persist_directory="vector_store")

def get_relevant_docs_basic(user_query):
    vectordb = Chroma(persist_directory="./vector_store",
                      embedding_function=embeddings)
    retriever = vectordb.as_retriever(score_threshold=0.5)
    relevant_docs = retriever.invoke(user_query)
    return relevant_docs

def get_relevant_docs_with_BM25(user_query):
    # Example of a vector search-based retriever (replace with actual implementation)
    return [{"page_content": "Sample content from vector search retriever"}]

def get_relevant_docs_with_multi_query(user_query):
    # Example of a custom retriever (replace with actual implementation)
    return [{"page_content": "Sample content from custom retriever"}]

def get_relevant_docs_with_ensemble(user_query):
    # Example of OpenAI retriever (replace with actual implementation)
    return [{"page_content": "Sample content from OpenAI retriever"}]

def make_rag_prompt(query, relevant_passage):
    relevant_passage = ''.join(relevant_passage)
    prompt = (
        f"You are a helpful assistant that evaluates resumes based on the provided job description. "
        f"Only use the information from the reference passage provided below. Do not include outside information or assumptions."
        f"\n\nYour task is as follows:"
        f"\n1. Analyze the given job description and compare it with the resumes provided in the context."
        f"\n2. Identify the resume that best matches the job description, and provide:"
        f"   - The metadata of the best-matching resume (e.g., candidate name, experience, skills, etc.)."
        f"   - A detailed explanation of why this resume is the best match for the job description."
        f"\n3. Rank the remaining resumes in order of relevance to the job description. For each, explain why it does not match perfectly and what areas are lacking compared to the best match."
        f"\n\nMaintain a professional and concise tone. Ensure your response is logical, clearly structured, and strictly based on the context provided."
        f"\n\nQUESTION: '{query}'\n"
        f"PASSAGE: '{relevant_passage}'\n\n"
        f"ANSWER:"
    )
    return prompt


def generate_response(user_prompt):
    answer = llm.invoke(user_prompt)
    return answer.content


def generate_answer(query, retriever_type):
    load_dotenv()
    relevant_text = get_relevant_docs_by_selection(retriever_type, query)
    text = " \n".join([doc.page_content for doc in relevant_text])
    prompt = make_rag_prompt(query, relevant_passage=text)
    print(prompt)
    answer = generate_response(prompt)
    return answer



def get_relevant_docs_by_selection(retriever_type, user_query):
    if retriever_type == "Basic Simliarity Search":
        return get_relevant_docs_basic(user_query)
    elif retriever_type == "BM25 Search":
        return get_relevant_docs_with_BM25(user_query)
    elif retriever_type == "MultiQuery Retriever":
        return get_relevant_docs_with_multi_query(user_query)
    elif retriever_type == "Ensemble Retriever":
        return get_relevant_docs_with_ensemble(user_query)
    else:
        return get_relevant_docs_basic(user_query)


def main():
    st.set_page_config("Resume Screening App", page_icon="📄", layout="wide")

    user_question = st.text_input("Enter your job description:")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        retriever_type = st.selectbox("Select Retriever Type", ["Chroma Retriever", "Custom Retriever", "OpenAI Retriever", "Vector Search"])
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

    if user_question:
        response = generate_answer(user_question, retriever_type)
        st.write("Answer: ", response)


if __name__== "__main__":
    main()