import streamlit as st
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import re

# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def evaluate_numerical(answer, correct_answer):
    try:
        answer_value = float(answer)
        correct_value = float(correct_answer)
        return answer_value == correct_value
    except ValueError:
        return False

def evaluate_mcq(answer, correct_answer):
    return answer.strip().lower() == correct_answer.strip().lower()


def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    return text

def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

def calculate_similarity(answer_embedding, reference_embedding):
    return cosine_similarity(answer_embedding, reference_embedding)[0][0]


def classify_question(question_text):
    if re.search(r'\d+', question_text):
        return 'numerical'
    elif re.search(r'\b(a|b|c|d)\b', question_text, re.IGNORECASE):
        return 'mcq'
    else:
        return 'open-ended'


def evaluate_answers(answers, reference_answers, question_types):
    results = []

    for i, (answer, reference, q_type) in enumerate(zip(answers, reference_answers, question_types)):
        if q_type == 'numerical':
            score = 100.0 if evaluate_numerical(answer, reference) else 0.0
            context_check = "Passed" if score == 100.0 else "Failed"
        elif q_type == 'mcq':
            score = 100.0 if evaluate_mcq(answer, reference) else 0.0
            context_check = "Passed" if score == 100.0 else "Failed"
        else:
            answer_embedding = get_embedding(preprocess_text(answer))
            reference_embedding = get_embedding(preprocess_text(reference))
            similarity = calculate_similarity(answer_embedding, reference_embedding)
            score = similarity * 100
            context_check = "Passed" if similarity > 0.5 else "Failed"

        results.append((answer, score, context_check, q_type))

    return results


def main():
    st.title("Hybrid Rule-Based and AI Answer Grading")

    uploaded_answers = st.file_uploader("Upload answer file (txt format)", type="txt")
    uploaded_references = st.file_uploader("Upload reference answers file (txt format)", type="txt")
    uploaded_questions = st.file_uploader("Upload questions file (txt format)", type="txt")

    if uploaded_answers is not None and uploaded_references is not None and uploaded_questions is not None:
        answers = uploaded_answers.read().decode("utf-8").splitlines()
        references = uploaded_references.read().decode("utf-8").splitlines()
        questions = uploaded_questions.read().decode("utf-8").splitlines()

        question_types = [classify_question(q) for q in questions]
        results = evaluate_answers(answers, references, question_types)

        for i, (answer, score, context_check, q_type) in enumerate(results):
            st.write(f"--- Answer {i+1} ({q_type}) ---")
            st.write(answer)
            st.write(f"Average Score: {score:.2f}")
            st.write(f"Context Check: {context_check}")

if __name__ == "__main__":
    main()