import streamlit as st
import json
from transformers import pipeline

# Load Medical Data
def load_medical_data():
    with open("medical_data.json", "r") as file:
        return json.load(file)

medical_data = load_medical_data()

# Load NLP Model
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Define Chatbot Function
def healthcare_chatbot(user_input):
    best_match = None
    for condition, advice in medical_data.items():
        if condition.lower() in user_input.lower():
            best_match = advice
            break

    if best_match:
        return best_match
    else:
        # Use AI if no match is found in JSON
        context = "\n".join([f"{k}: {v}" for k, v in medical_data.items()])
        result = qa_pipeline(question=user_input, context=context)
        return result["answer"]

# Streamlit UI
st.title("🩺 AI Health Assistant")
st.write("Ask about common medical symptoms!")

user_input = st.text_input("Enter your symptoms:")

if user_input:
    response = healthcare_chatbot(user_input)
    st.write("### 🤖 AI Response:")
    st.success(response)
