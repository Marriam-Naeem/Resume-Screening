from datasets import Dataset
from dotenv import load_dotenv
from ragas import evaluate
import os 
from resume_screening import generate_response, generate_answer,get_relevant_docs_basic,get_relevant_docs_with_BM25,get_relevant_docs_with_multi_query,get_relevant_docs_with_ensemble,get_relevant_docs_RAGFusion




load_dotenv()
os.getenv("OPENAI_API_KEY")

retriever_type = "Basic Simliarity Search"

questions = [
    "Versatile Information Technology Professional with extensive experience in IT operations, team leadership, and strategic implementation of technology solutions. Skilled in cloud computing, cybersecurity frameworks, and IT governance. Proficient in driving digital transformation, managing budgets, and ensuring compliance with industry standards. Strong expertise in data analytics, artificial intelligence integration, and DevOps practices. Known for innovative problem-solving, cross-functional collaboration, and a focus on business continuity."]


ground_truths = [
    ["Here is the Resume that Best Matches the Job Description that you have provided: Candidate's Name: (Unavailable in dataset; associated with index 2) Index of the Resume: 2 Experience, Education, Skills, and Relevant Information: Experience: Information Technology Manager with professional accomplishments in managing IT teams and strategic projects. Skills: Likely includes IT operations and team management (specifics unavailable in the raw text). Education: Details not explicit in the text. File Name: data.csv Why this resume is the best match: The candidate's profile matches one keyword in the job description and highlights relevant IT leadership and managerial roles, although details are sparse. Ranking of Remaining Resumes: Index 5 Score: 1 (Equal to Index 2, but with less specificity to IT leadership roles). Why it's not the best match: While it mentions IT management and a career overview, it lacks explicit mention of strategic implementation or advanced skills like cloud computing or AI. Indexes 1, 3, 4 Score: 0 Why they are less relevant: Index 1: Focuses on an IT Technician role, not aligned with the seniority or strategic aspects of the job description. Index 3: Mentions RF Systems Engineering, unrelated to the IT scope of the job description. Index 4: Repeats the title of IT Manager but lacks content aligning with the required skills and experiences."]
]

# Simulate an empty list to hold answers and contexts
answers = []
context = []
references = []

# Simulate response
for query in questions:
    context, response = generate_answer(query,retriever_type)
    answers.append(response)
    context.append(context)
    references.append(ground_truths[questions.index(query)][0])
   

# Convert to dictionary format for creating the dataset
data = {
    "question": questions,
    "answer": answers,  # Ensure answers are strings
    "contexts": context,
    "ground_truths": ground_truths,
    "reference": references
}

# Convert the dictionary into a Dataset object
dataset = Dataset.from_dict(data)

# Print or explore the dataset
print(dataset)
#display rows


from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    answer_correctness,
    answer_similarity
)

result = evaluate(
    dataset = dataset,
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
        answer_correctness,
        answer_similarity
    ],
)

df = result.to_pandas()

print(df)

