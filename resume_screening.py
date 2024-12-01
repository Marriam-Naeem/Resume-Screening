import streamlit as st
from PyPDF2 import PdfReader
import os
from json import dumps, loads
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import pandas as pd
from langchain_community.document_loaders import CSVLoader 
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema import HumanMessage, SystemMessage, AIMessage


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
    vectordb = Chroma(persist_directory="vector_store",
                      embedding_function=embeddings)
    ids = vectordb.get().get("ids")
    for ids in ids:
        vectordb.delete(ids=ids)
        #print(f"Deleted {ids}")
    loader = CSVLoader(file_path='data.csv', source_column="index")
    data = loader.load()
    #print(data)
    vectordb = Chroma.from_documents(
        documents=data, 
        embedding=embeddings, 
        persist_directory="vector_store")
    vectordb.persist()



def generate_subquestions(question: str):
   
    # Define the system message for task instructions
    system_message = SystemMessage(content="""
    You are an expert in talent acquisition. Separate this job description into 3-4 more focused aspects for efficient resume retrieval. 
    Make sure every single relevant aspect of the query is covered in at least one query. You may choose to remove irrelevant information 
    that doesn't contribute to finding resumes such as the expected salary of the job, the ID of the job, the duration of the contract, etc.
    Only use the information provided in the initial query. Do not make up any requirements of your own. 
    Put each result in one line, separated by a linebreak.
    """)

    # Provide an example for the model
    oneshot_example = HumanMessage(content="""
      Generate 3 to 4 sub-queries based on this initial job description:

      Wordpress Developer
      We are looking to hire a skilled WordPress Developer to design and implement attractive and functional websites and Portals for our Business and Clients. You will be responsible for both back-end and front-end development including the implementation of WordPress themes and plugins as well as site integration and security updates.
      To ensure success as a WordPress Developer, you should have in-depth knowledge of front-end programming languages, a good eye for aesthetics, and strong content management skills. Ultimately, a top-class WordPress Developer can create attractive, user-friendly websites that perfectly meet the design and functionality specifications of the client.
      WordPress Developer Responsibilities:
      Meeting with clients to discuss website design and function.
      Designing and building the website front-end.
      Creating the website architecture.
      Designing and managing the website back-end including database and server integration.
      Generating WordPress themes and plugins.
      Conducting website performance tests.
      Troubleshooting content issues.
      Conducting WordPress training with the client.
      Monitoring the performance of the live website.
      WordPress Developer Requirements:
      Bachelors degree in Computer Science or a similar field.
      Proven work experience as a WordPress Developer.
      Knowledge of front-end technologies including CSS3, JavaScript, HTML5, and jQuery.
      Knowledge of code versioning tools including Git, Mercurial, and SVN.
      Experience working with debugging tools such as Chrome Inspector and Firebug.
      Good understanding of website architecture and aesthetics.
      Ability to project manage.
      Good communication skills.
      Contract length: 12 months
      Expected Start Date: 9/11/2020
      Job Types: Full-time, Contract
      Salary: 12,004.00 - 38,614.00 per month
      Schedule:
      Flexible shift
      Experience:
      Wordpress: 3 years (Required)
      web designing: 2 years (Required)
      total work: 3 years (Required)
      Education:
      Bachelor's (Preferred)
      Work Remotely:
      Yes
    """)
    oneshot_response = AIMessage(content="""
    1. WordPress Developer Skills:
       - WordPress, front-end technologies (CSS3, JavaScript, HTML5, jQuery), debugging tools (Chrome Inspector, Firebug), code versioning tools (Git, Mercurial, SVN).
       - Required experience: 3 years in WordPress, 2 years in web designing.

    2. WordPress Developer Responsibilities:
       - Meeting with clients for website design discussions.
       - Designing website front-end and architecture.
       - Managing website back-end including database and server integration.
       - Generating WordPress themes and plugins.
       - Conducting website performance tests and troubleshooting content issues.
       - Conducting WordPress training with clients and monitoring live website performance.

    3. WordPress Developer Other Requirements:
       - Education requirement: Bachelor's degree in Computer Science or similar field.
       - Proven work experience as a WordPress Developer.
       - Good understanding of website architecture and aesthetics.
       - Ability to project manage and strong communication skills.

    4. Skills and Qualifications:
       - Degree in Computer Science or related field.
       - Proven experience in WordPress development.
       - Proficiency in front-end technologies and debugging tools.
       - Familiarity with code versioning tools.
       - Strong communication and project management abilities.
    """)

    # User message with the job description
    user_message = HumanMessage(content=f"""
    Generate 3 to 4 sub-queries based on this initial job description: 
    {question}
    """)

    # Generate the response
    response = llm.invoke([system_message, oneshot_example, oneshot_response, user_message])

    # Parse the output into a list of subquestions
    subquestions = response.content.split("\n")
    return subquestions

def get_subquestion_docs(subquestions):
    relevant_docs = []
    for subquestion in subquestions:
        relevant_docs.append(get_relevant_docs_basic(subquestion))
    return relevant_docs


def reciprocal_rank_fusion(results, k=60):
    fused_scores = {}
    # Iterate through results for each subquery
    for docs in results:
        for rank, doc in enumerate(docs):
            # Convert Document object to a dictionary
            doc_dict = {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
            doc_str = dumps(doc_dict)  # Serialize the dictionary
            
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)
    
    # Rerank documents based on their scores
    reranked_results = [
        (loads(doc),score, f"This value {score} refers to the document's relevance based on its rank across multiple subqueries.")
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    
    return reranked_results


def get_relevant_docs_RAGFusion(user_query):
    subquestions = generate_subquestions(user_query)
    print(subquestions)
    relevant_docs = get_subquestion_docs(subquestions)
    print(relevant_docs)
    results = reciprocal_rank_fusion(relevant_docs)
    print(results)
    return results



def get_relevant_docs_basic(user_query):
    vectordb = Chroma(persist_directory="vector_store",
                      embedding_function=embeddings)
    retriever = vectordb.as_retriever(score_threshold=0.5)
    relevant_docs = retriever.invoke(user_query)
    return relevant_docs

def get_relevant_docs_with_BM25(user_query):

    loader = CSVLoader(file_path='data.csv', source_column="index")
    data = loader.load()

    bm25_retriever = BM25Retriever.from_documents(
        data, 
        k=5
    )
    relevant_docs = bm25_retriever.invoke(user_query)
    return relevant_docs

def get_relevant_docs_with_multi_query(user_query):
    vectordb = Chroma(persist_directory="./vector_store",
                      embedding_function=embeddings)
    retriever = MultiQueryRetriever.from_llm(retriever=vectordb.as_retriever(score_threshold=0.5), llm=llm)
    relevant_docs = retriever.invoke(user_query)
    return relevant_docs

def get_relevant_docs_with_ensemble(user_query):
    vectordb = Chroma(persist_directory="./vector_store",
                      embedding_function=embeddings)
    multi_query_retriever = MultiQueryRetriever.from_llm(retriever=vectordb.as_retriever(score_threshold=0.5), llm=llm)

    loader = CSVLoader(file_path='data.csv', source_column="index")
    data = loader.load()

    bm25_retriever = BM25Retriever.from_documents(
        data, 
        k=5
    )

    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, multi_query_retriever], weights=[0.5, 0.5])
    relevant_docs = ensemble_retriever.invoke(user_query)
    return relevant_docs


def make_rag_prompt(query, relevant_passage):
    prompt = (
        f"You are a helpful assistant that evaluates resumes based on the provided job description. "
        f"Only use the information from the reference passage provided below. Do not include outside information or assumptions."
        f"Please know that there is no repetition of data. If multiple documents share the same `source` in the metadata and `index` in the page content, It means that they are from the same resume"
        f"If the question is not related to the resume, politely respond that you are tuned to only answer questions that are related to resumes."
        f"\n\nYour task is as follows:"
        f"\n1. Analyze the given job description and compare it with the resumes provided in the context."
        f"\n2. Identify the resume that best matches the job description, and provide:"
        f"   - The Candidate's name. , index of the resume, experince, education, skills, and any other relevant information."
        f"   - The index of the best-matching resume in the context."
        f"   - A detailed explanation of why this resume is the best match for the job description."
        f"\n3. Rank the remaining resumes in order of relevance to the job description. For each, explain why it does not match perfectly and what areas are lacking compared to the best match."
        f"\n4. **Important**: The `source` in the meta data and `index` in the page content are considered the same. If multiple documents share the same `source` and `index`, they refer to the same candidate and should be treated as such. Ignore the `row` number in the metadata as it is not relevant for evaluating candidates."
        f"\n\nStart your response with 'Here is the Resume that Best Matches the Job Description that you have provided:'"
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
    # text = " \n".join([doc.page_content for doc in relevant_text])
    prompt = make_rag_prompt(query, relevant_passage=relevant_text)
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
    elif retriever_type == "RAG Fusion":
        return get_relevant_docs_RAGFusion(user_query)
    else:
        return get_relevant_docs_basic(user_query)


def main():
    st.set_page_config("Resume Screening")
    st.header("Resume Screening")

    with st.sidebar:
        st.title("Menu:")

        # File uploader widget
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, type=["pdf"])

        # Check if the submit button is clicked
        if st.button("Submit & Process"):
            # Check if files are uploaded
            if not pdf_docs:
                st.warning("Please upload at least one PDF file to proceed.")
            else:
                with st.spinner("Processing..."):
                    process_pdfs(pdf_docs)
                    get_vector_store()
                    st.success("Done")

        # Dropdown for retriever type selection
        st.write("Select a Retriever Type:")
        retriever_type = st.selectbox("Select Retriever Type", ["Basic Simliarity Search", "BM25 Search", "MultiQuery Retriever", "Ensemble Retriever", "RAG Fusion"])
        
    # Input for user question
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