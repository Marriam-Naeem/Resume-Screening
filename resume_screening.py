import streamlit as st
from PyPDF2 import PdfReader
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
import pandas as pd
from langchain_community.document_loaders import CSVLoader 
from langchain.retrievers.multi_query import MultiQueryRetriever


load_dotenv()
os.getenv("GOOGLE_API_KEY")

embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.1,
    max_tokens=None,
    timeout=None,
    max_retries=5,)

# Function to process PDFs and save metadata and chunk data
def process_pdfs(pdf_docs):
    metadata = []
    data = []
    index = 1

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        text = "".join([page.extract_text() for page in pdf_reader.pages])

        # Save metadata
        metadata.append({"index": index, "file_name": pdf.name})

        # Split text into chunks
        chunk_size = 800
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            data.append({"index": index, "text": chunk})
        index += 1

    # Save to CSV
    metadata_df = pd.DataFrame(metadata)
    data_df = pd.DataFrame(data)
    metadata_df.to_csv("metadata.csv", index=False)
    data_df.to_csv("data.csv", index=False)


def get_vector_store():
    loader = CSVLoader(file_path='data.csv', source_column="index")
    data = loader.load()
    print(data)
    vectordb = Chroma.from_documents(
        documents=data, 
        embedding=embeddings, 
        persist_directory="vector_store")


def get_relevant_docs_basic(user_query):
    vectordb = Chroma(persist_directory="vector_store",
                      embedding_function=embeddings)
    retriever = vectordb.as_retriever(score_threshold=0.5)
    relevant_docs = retriever.invoke(user_query)
    return relevant_docs

def get_relevant_docs_with_BM25(user_query):
    vectordb = Chroma(persist_directory="vector_store",
                      embedding_function=embeddings)
    retriever = vectordb.as_retriever(score_threshold=0.5)
    relevant_docs = retriever.invoke(user_query)
    return relevant_docs

def get_relevant_docs_with_multi_query(user_query):
    vectordb = Chroma(persist_directory="./vector_store",
                      embedding_function=embeddings)
    retriever = MultiQueryRetriever.from_llm(
    retriever=vectordb.as_retriever(score_threshold=0.5), llm=llm)
    relevant_docs = retriever.invoke(user_query)
    return relevant_docs

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
        f"   - The metadata of the best-matching resume (e.g., candidate name, experience, skills etc.)."
        f"   - The index of the best-matching resume in the context."
        f"   - A detailed explanation of why this resume is the best match for the job description."
        f"\n3. Rank the remaining resumes in order of relevance to the job description. For each, explain why it does not match perfectly and what areas are lacking compared to the best match."
        f"\n Remember there is no repition the data, so if some information us repeated, consider that only once in result"
        f"\n Start your response with 'Here is the Resume that Best Matches the Job Description that you have provided:'"
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
    st.set_page_config("Resume Screening")
    st.header("Resume Screening")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        retriever_type = st.selectbox("Select Retriever Type", ["Basic Simliarity Search", "BM25 Search", "MultiQuery Retriever", "Ensemble Retriever"])
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                process_pdfs(pdf_docs)
                get_vector_store()
                st.success("Done")
            # Display metadata.csv
                st.write("The files you uploaded have been assigned the following indices:")
                metadata_df = pd.read_csv("metadata.csv")
                st.dataframe(metadata_df)
    

    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:

        if os.path.exists("metadata.csv"):
            st.write("The files you uploaded have been assigned the following indices:")
            metadata_df = pd.read_csv("metadata.csv")
            st.dataframe(metadata_df)
        response = generate_answer(user_question, retriever_type)
        st.write("Answer: ", response)
    


if __name__== "__main__":
    main()