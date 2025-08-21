# Resume-Screening

An intelligent resume screening system built using **Retrieval-Augmented Generation (RAG)**.  
The project explores multiple retrieval methods and integrates state-of-the-art LLMs to improve resume-to-job matching accuracy, reduce bias, and handle diverse resume formats effectively.

## Table of Contents
- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Research Objectives](#research-objectives)
- [Significance](#significance)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Implementation](#implementation)
- [Evaluation](#evaluation)
- [Results](#results)
- [Setup](#setup)
- [Future Work](#future-work)

## Introduction
Traditional resume screening is time-consuming, prone to bias, and inconsistent across recruiters.  
This project applies **RAG-based methods** to automate and improve resume screening by combining advanced document processing, multiple retrieval techniques, and LLMs for context-aware ranking.

## Problem Statement
Manual resume screening faces challenges such as:
- **Time Intensity** – HR professionals spend ~23% of their time screening resumes.  
- **Bias** – Both conscious and unconscious bias influence selection.  
- **Volume Management** – Thousands of applications per job make manual review impractical.  
- **Contextual Limitations** – Keyword-based systems fail to capture semantic meaning.  
- **Lack of Standardization** – Inconsistent criteria between recruiters.  

## Research Objectives
- Develop an **advanced RAG-based system** for accurate, context-aware resume screening.  
- Implement **semantic matching** beyond keyword search.  
- Evaluate multiple retrieval approaches (**BM25, MultiQuery, Ensemble, RAG Fusion**).  
- Reduce bias and improve fairness in candidate ranking.  

## Significance
This work enhances recruitment efficiency by:  
- Improving accuracy with multiple retrieval methods.  
- Enhancing contextual understanding with LLMs.  
- Handling diverse resume layouts and job descriptions.  
- Reducing bias and inconsistencies in candidate evaluation.  

## System Architecture

### Components
- **Document Processing Pipeline:**  
  - PDF parsing, text extraction (PyPDF2)  
  - Text splitting (RecursiveCharacterTextSplitter)  
  - Metadata & embeddings stored in ChromaDB  

- **Retrieval Methods:**  
  - Basic Similarity Search (cosine similarity)  
  - BM25  
  - MultiQuery Retriever  
  - Ensemble Retriever  
  - **RAG Fusion** (Reciprocal Rank Fusion)  

- **Language Model Integration:**  
  - Embeddings: Google Generative AI  
  - Response generation: **Gemini-1.5-flash**  
  - Prompting: one-shot prompting, chain-of-thought reasoning  

## Features
- Automated resume parsing and semantic retrieval.  
- Multiple retrievers with comparative evaluation.  
- Context-aware response generation using LLMs.  
- RAG Fusion for improved accuracy on complex queries.  
- Streamlit-based user interface for uploading resumes and querying.  

## Technologies Used
- **Python**  
- **Streamlit** (UI)  
- **PyPDF2** (resume parsing)  
- **LangChain**  
- **ChromaDB** (vector store)  
- **Google Generative AI** (embeddings + Gemini-1.5-flash model)  

## Implementation
1. PDF resumes parsed and converted into text.  
2. Text split into chunks (800 chars, 50 overlap).  
3. Embeddings stored in **ChromaDB**.  
4. Queries processed through different retrieval methods.  
5. Final ranking generated via **RAG Fusion**.  
6. Results displayed in Streamlit UI.  

## Evaluation
Evaluation used **RAGAS metrics**:  
- Context Precision & Recall  
- Faithfulness  
- Answer Relevancy  
- Answer Correctness  
- Semantic Similarity  

### Retrieval Comparison
- **RAG Fusion** → best overall accuracy.  
- **MultiQuery** → strong coverage for diverse queries.  
- **BM25, Ensemble, Basic Search** → lower accuracy, useful for baseline comparison.  

## Results
- **Relevant queries:** RAG Fusion achieved perfect context precision/recall (1.0).  
- **Somewhat relevant queries:** RAG Fusion outperformed others in faithfulness and answer relevancy.  
- **Irrelevant queries:** MultiQuery occasionally retrieved results; RAG Fusion handled best with faithful answers.  

## Setup

1. **Clone the repository:**

    ```bash
    git clone https://github.com/Marriam-Naeem/Resume-Screening.git
    cd Resume-Screening
    ```

2. **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    # Activate (Linux/Mac)
    source venv/bin/activate
    # Activate (Windows)
    venv\Scripts\activate
    ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Run the application:**

    ```bash
    streamlit run app.py
    ```

5. **(Optional) Configure paths:**
   Update `config.py` for custom model paths, vector store directory, or dataset location.

## Future Work

- Fine-tune LLMs for specific industry recruitment scenarios to improve accuracy.  
- Add real-time adaptability using user/recruiter feedback loops.  
- Extend system capabilities for multi-lingual resume and job description support.  
- Explore hybrid retrieval (static + dynamic) for scalability and efficiency.  

