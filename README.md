Resume Matching Engine
AI-Powered Resume Screening and Candidate Ranking

🔗 Live App: [Insert link to live app]
📂 GitHub Repo: [Insert link to GitHub repository]

---

Problem Statement

The Resume Matching Engine project aims to automate the resume screening process by developing an AI-powered engine that can match resumes with job descriptions and provide a ranked list of top candidates.

---

Dataset and Data Engineering

The project uses a dataset consisting of:
10 resume datasets from Indian university students
3 job descriptions from Korean technology companies

The dataset is preprocessed using the following steps:
Skill Normalization: Skills are normalized using a predefined alias mapping to ensure consistency across resumes.
Deduplication: Duplicate skills are removed from each resume.
Vocabulary Construction: A shared vocabulary is constructed from the normalized and deduplicated resume skills.
TF-IDF Vector Construction: TF-IDF vectors are computed for each resume using the constructed vocabulary.
Job Description Vector Construction: Binary vectors are constructed for each job description using the same vocabulary.

---

Model Performance

The model performance is evaluated using cosine similarity between resume TF-IDF vectors and job description binary vectors. The top 3 candidates are selected for each job description based on their similarity scores.

---

Example Output

The output of the Resume Matching Engine is a ranked list of top candidates for each job description, along with their similarity scores.

Example:
JD-1 - Kakao (ML Engineer):
Arjun Sharma (0.85)
Priya Nair (0.78)
Rahul Gupta (0.75)
JD-2 - Naver (Backend Engineer):
Sneha Patel (0.90)
Vikram Singh (0.85)
Ananya Krishnan (0.80)
JD-3 - Line (Frontend Engineer):
Karan Mehta (0.92)
Deepika Rao (0.88)
Aditya Kumar (0.85)

---

Code Structure

The code is organized into the following sections:
Skill Normalization: skill_normalization.py
Vocabulary Construction: vocabulary_construction.py
TF-IDF Vector Construction: tfidfvector_construction.py
Job Description Vector Construction: jobdescriptionvector_construction.py
Cosine Similarity Computation: cosinesimilaritycomputation.py
Main Entry Point: main.py

---

Requirements

The project requires the following libraries and frameworks:
Python 3.8+
Redrob AI library
NumPy
Pandas
Scikit-learn

---

Usage

To run the Resume Matching Engine, follow these steps:
Clone the repository: git clone https://github.com/your-username/resume-matching-engine.git
Install the required libraries: pip install -r requirements.txt
Run the main entry point: python main.py

---

Contributing

Contributions are welcome! To contribute to the project, please follow these steps:
Fork the repository: git fork https://github.com/Hiteshtyagi610/resume_parser
Create a new branch: git branch your-branch-name
Make changes and commit: git commit -m "Your commit message"
Push changes to your fork: git push origin your-branch-name
Create a pull request: git pull-request



---

Acknowledgments

The Resume Matching Engine uses the following libraries and frameworks:
Redrob AI library


---

Future Improvements

The following improvements are planned for the Resume Matching Engine:
Model Explainability: Implement model explainability using SHAP values or other techniques.
Real-Time Data Ingestion: Integrate real-time data ingestion to enable continuous updating of the model.
Advanced Anomaly Detection: Implement advanced anomaly detection techniques, such as Isolation Forest or One-Class SVM.
API Deployment: Deploy the Resume Matching Engine as a RESTful API using Flask or Django.

---

Author

Your Name
hiteshtyagi130305@gmail.com
+91-7428925864

---

Conclusion

The Resume Matching Engine is an AI-powered engine that automates the resume screening process by matching resumes with job descriptions and providing a ranked list of top candidates. The engine uses a combination of natural language processing and machine learning techniques to achieve high accuracy and efficiency. The project is open-source and welcomes contributions from the community.