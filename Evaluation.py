from datasets import Dataset
from dotenv import load_dotenv
from ragas import evaluate
import os 
from resume_screening import generate_response, generate_answer,get_relevant_docs_basic,get_relevant_docs_with_BM25,get_relevant_docs_with_multi_query,get_relevant_docs_with_ensemble,get_relevant_docs_RAGFusion




load_dotenv()
os.getenv("OPENAI_API_KEY")
os.getenv("GOOGLE_API_KEY")

retriever_type = "RAG Fusion"

questions = [
    "I need someone as a Versatile Information Technology Professional with extensive experience in IT operations, team leadership, and strategic implementation of technology solutions. Skilled in cloud computing, cybersecurity frameworks, and IT governance. Proficient in driving digital transformation, managing budgets, and ensuring compliance with industry standards. Strong expertise in data analytics, artificial intelligence integration, and DevOps practices. Known for innovative problem-solving, cross-functional collaboration, and a focus on business continuity.",
   "I need experienced Information Technology Manager with over a decade of expertise in IT operations, infrastructure management, and project delivery. Proficient in leveraging methodologies like Scrum and Waterfall to enhance team performance, ensure high availability systems, and drive business KPIs. Demonstrates strong skills in IT infrastructure design, enterprise platform management, and customer-focused solutions across global markets, including SaaS environments. Recognized for effective leadership in team development, resource optimization, and innovative problem-solving." ,
   "I need someone who is experienced in coordinating marketing teams and managing marketing projects to ensure efficiency and meet objectives. Skilled in problem-solving, adapting to new environments, and improving processes to support overall success. Focused on fostering collaboration and driving positive results."]
ground_truths = [
    ["""Based on the provided job description and resumes, Candidate 2 (index 2) is the best match for the role of Versatile Information Technology Professional.

1. Best Matching Candidate:

Candidate Name: The passage does not provide the candidate's name.
Resume Index: 2 (Source 1).

Experience:
Candidate 2 brings extensive experience as an Information Technology Manager, including:

Managing application databases, hardware systems, office technologies, and anti-spam/anti-virus servers.
Evaluating, recommending, implementing, and troubleshooting hardware and software.
Maintaining LAN/WAN infrastructure, connectivity, and security, along with creating LAN user documentation.
Performing disaster recovery planning and administering licenses/service contracts.
Designing and implementing network infrastructure enhancements for improved performance, security, and remote access.
Establishing a helpdesk system to prioritize and manage IT support requests.
Implementing automation solutions, such as scripting for data retrieval and asset tracking, resulting in improved efficiency and significant cost savings.

Education:

Master of Science in Leadership.
Master of Science in Information Systems Management (Project Management focus).
Bachelor of Science in Information Systems Management and Mechanical Engineering — all from Walden University.

Skills and Certifications:

Key Skills: Backup/Restore Operations, Strategic Planning, Budgeting, Team Building, Policy Development, Troubleshooting, Process Improvement, and Quality Assurance.
Technical Expertise: LAN/WAN infrastructure, disaster recovery, software standardization, and helpdesk support systems.
Certifications: MCP, CompTIA Network+, and CompTIA A+.

2. Explanation of Best Match
Candidate 2's resume aligns closely with the role’s requirements:

Extensive IT Management Experience: Their role as an IT Manager demonstrates expertise in leading complex projects, managing IT infrastructure, and optimizing workflows.
Strategic Skills: Proven ability in budgeting, policy development, and implementing technological solutions to enhance performance.
Technical Proficiency: Hands-on experience with LAN/WAN infrastructure, disaster recovery, and automation aligns with key responsibilities of the job.
Leadership: Direct involvement in team coordination, special projects, and user training reflects strong leadership capabilities.
Educational Foundation: Advanced degrees in Leadership and Information Systems Management highlight their preparedness for strategic and managerial aspects of IT.
While cloud computing and cybersecurity frameworks aren’t explicitly mentioned, their expertise in network security and disaster recovery suggests relevant transferable skills.

3. Ranking of Remaining Resumes:

The remaining resumes lack the breadth and depth of experience and education demonstrated by Candidate 2. Specific gaps include:

Candidate 1 (index 1): This candidate has experience as a Systems Administrator, showcasing skills in troubleshooting, server management, and systems analysis. However, their resume lacks the extensive experience in team leadership, strategic planning, budgeting, and the specific technologies mentioned in the job description (e.g., cloud computing, AI integration).

Candidate 3 (index 3): This candidate's experience focuses on data analysis and RF systems, which are only partially relevant to the job description. Their skills in data analysis are mentioned, but the role requires a broader IT skillset.

Candidate 4 (index 4): This candidate has experience as an IT Manager and IT Administrator, but the details provided are less comprehensive than Candidate 2's resume. The description lacks specifics on the scope of their responsibilities and the technologies they used.

Candidate 5 (index 5): This candidate's resume is a list of skills and accomplishments in various IT areas, but lacks structured experience descriptions and doesn't clearly demonstrate the required level of expertise in team leadership, strategic planning, or budget management.

In summary, while other candidates possess some relevant skills, Candidate 2's resume presents the most comprehensive and direct alignment with the requirements of the Versatile Information Technology Professional role."""],
    
    ["""Based on the provided job description and resumes, Candidate 4 (index 4) is the best match for the Information Technology Manager position.

1. Best Matching Candidate:

Candidate Name: The name is not explicitly provided in the passage.
Resume Index: 4 (source 3)
Experience:
- **IT Manager**:
  - Tenure: March 2013 – Present
  - Responsibilities: Managing a four-person IT team, resource allocation, enforcing deadlines, and overseeing SaaS customers across North America, Canada, and Australia. The candidate also focused on customer experience optimization, team building, budget management, and executing proof-of-concept projects.
  - Highlights: Experience includes managing global IT support, collaborating across international teams, and leveraging Waterfall and Scrum methodologies.
- **IT Administrator**:
  - Tenure: June 2011 – March 2013
  - Responsibilities: Designing and delivering mission-critical infrastructure, managing IT infrastructure across multiple locations (Chicago, Houston, Montreal, Sydney), disaster recovery, and VMware architecture deployment.
- **Other Roles**: Various IT positions showcasing a progressive career trajectory and technical expertise in network administration and IT consulting.

Education:
- Bachelor’s in Network and Communications Management (2009, DeVry University)
- Master’s in Business Information Technology (2018, DePaul University)

Skills:
- **Leadership**: Team management, project tracking, and performance criteria tracking.
- **Technical**: Active Directory, VMware, disaster recovery, LAN/WAN management, Microsoft Exchange, and Windows Server.
- **Methodologies**: Waterfall framework and Scrum for project delivery.
- **Other**: Budget management, staff development, and operations management.

Alignment with Job Description:
Candidate 4’s extensive managerial experience, technical skills, and demonstrated ability to deliver IT projects on a global scale strongly align with the job requirements. Their knowledge of Agile methodologies (Scrum) and Waterfall frameworks adds further relevance, especially for positions demanding modern project management expertise.

2. Ranking of Remaining Resumes:

- **Candidate 2 (index 2)**: This candidate holds a Master of Science degree in Information Systems Management and multiple certifications. However, they lack explicit experience in managing a team or large-scale IT infrastructure projects, as seen in Candidate 4's resume. While their expertise in Network Engineering and Disaster Recovery is notable, the scope and level of experience are not clearly defined.
  
- **Candidate 1 (index 1)**: This candidate's resume highlights technical skills such as troubleshooting, system administration, and backup management, but there is no mention of managerial experience or leadership roles, which are critical for the position.

- **Candidate 5 (index 5)**: While this candidate demonstrates a broad range of IT skills, including website development, hardware installation, and troubleshooting, there is no indication of managerial experience or proficiency in methodologies like Scrum or Waterfall, or in managing large-scale IT infrastructure projects.

- **Candidate 3 (index 3)**: This candidate’s experience is focused on technical project design, development, testing, and validation in the RF systems domain. However, there is no indication of managerial experience or the broader IT infrastructure management skills needed for the position.

In conclusion, Candidate 4's resume is the strongest match for the position, demonstrating the relevant managerial experience, technical expertise, and project delivery skills required. The other candidates lack the necessary experience and/or breadth of skills as outlined in the job description."""],
    
    ["""After reviewing the provided job description and resumes, none of the candidates exhibit strong experience in coordinating marketing teams or managing marketing projects. The job description emphasizes skills such as project management, team coordination, problem-solving, adaptability, and process improvement within a marketing context. Unfortunately, none of the resumes explicitly mention relevant marketing experience or align closely with the core requirements outlined in the job description. As a result, it is not possible to rank the candidates meaningfully based on relevance. To identify a more suitable match, it is recommended to source additional resumes from candidates with demonstrated marketing experience."""]
]

print(f"Number of questions: {len(questions)}")
print(f"Number of ground truths: {len(ground_truths)}")
 
# Simulate an empty list to hold answers and contexts
answers = []
context = []
references = []

# Simulate response
for query in questions:
    cxt, response = generate_answer(query,retriever_type)
    answers.append(response)
    context.append([entry[0]['page_content'] for entry in cxt])  # Extract page_content from each document
    references.append(ground_truths[questions.index(query)][0])


# Convert to dictionary format for creating the dataset
data = {
    "question": questions,
    "answer": answers,  # Ensure answers are strings
    "contexts": context,
    "ground_truths": ground_truths,
    "reference": references
}

print(data["contexts"])


# Convert the dictionary into a Dataset object
dataset = Dataset.from_dict(data)

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

df.to_csv("evaluation_result.csv", index=False)