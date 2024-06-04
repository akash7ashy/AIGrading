# AIGrading
#Project: Hybrid Rule-Based and AI Answer Grading

This project implements a web application for evaluating student answers using a combination of rule-based and AI techniques. It utilizes Streamlit to create a user-friendly interface for uploading answer sheets, reference keys, and question papers.

#Functionality

#Upload Files:

Users can upload answer sheets (txt format), reference answer keys (txt format), and question papers (txt format) using the designated file upload sections.
Answer Evaluation:

The application classifies each question based on its format:
Numerical: Questions containing numbers (e.g., "What is 2 + 2?").
MCQ (Multiple Choice Question): Questions with answer choices (e.g., "What is the capital of France? (a) Paris (b) London").
Open Ended: Questions requiring descriptive answers (e.g., "Describe the process of photosynthesis").
For numerical questions, the application checks if the answer matches the reference answer exactly.
For MCQ questions, it checks if the selected answer option matches the reference answer.
For open-ended questions, it employs a pre-trained BERT model (Bidirectional Encoder Representations from Transformers) to evaluate semantic similarity between the student's answer and the reference answer. Cosine similarity is used to quantify this semantic similarity. A threshold is set (default 0.5) to determine whether the answer aligns with the context of the reference answer.
Grading:

Based on the evaluation method for each question type, scores are assigned:
Numerical and MCQ: 100 points for a correct answer, 0 points for an incorrect answer.
Open Ended: The score is based on the cosine similarity between the answer and reference answer, multiplied by 100 (e.g., a similarity of 0.8 translates to a score of 80).
Additionally, a "Context Check" is included for open-ended questions. It indicates "Passed" if the similarity is above the threshold, suggesting the answer aligns with the reference answer's context, or "Failed" otherwise.
Results:

The application displays the evaluated answers along with their corresponding scores, context checks, and question types. This provides a detailed breakdown of the grading process.
Components

Streamlit: A Python library for creating web apps.
Transformers: A library for natural language processing (NLP) tasks, including the pre-trained BERT model used for semantic similarity evaluation.
Torch: A numerical computation library used for tensor operations (BERT model computations).
scikit-learn: A machine learning library used for cosine similarity calculations.
Instructions

Ensure you have Python installed along with the required libraries (Streamlit, Transformers, Torch, scikit-learn). You can install them using pip install streamlit transformers torch scikit-learn.
Place all the code files (including this readme file) in the same directory.
Open a terminal or command prompt in that directory.
Run the application using streamlit run main.py.
This will launch the Streamlit app in your web browser, typically at http://localhost:8501/.
Upload the necessary files (answer sheets, reference keys, question papers) and click 'Enter' after each upload.
The application will process the files, evaluate the answers, and display the results.
Note:

This is a basic implementation and can be further enhanced by incorporating features like answer visualization, feedback generation, and support for more question types.
The accuracy of the BERT model's evaluation for open-ended questions can be improved by fine-tuning it on a specific domain-related dataset.
