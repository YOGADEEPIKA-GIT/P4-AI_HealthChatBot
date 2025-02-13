import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="AI Health Assistant", page_icon="🩺", layout="centered")


# ✅ Load a More Suitable Model (Using GPT-2 for better responses)
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="gpt2")

text_gen_pipeline = load_model()

# ✅ Improved Medical Knowledge Base
HEALTH_CONTEXT = {
    "fever": "If you have a fever, take acetaminophen (paracetamol) or ibuprofen to reduce it. Stay hydrated and get enough rest. If fever is above 102°F (39°C) or lasts more than 3 days, consult a doctor.",
    "headache": "For headaches, use ibuprofen, aspirin, or acetaminophen. Migraines may be treated with triptans like sumatriptan.",
    "cold": "For cold and flu, use antihistamines, decongestants, and pain relievers. Vitamin C and honey may help soothe symptoms.",
    "stomach pain": "If you have stomach pain, try antacids for acidity. If pain persists, consult a doctor.",
    "diabetes": "Diabetes is managed with insulin, metformin, and diet control. Avoid excessive sugar intake.",
    "high blood pressure": "To manage high blood pressure, use beta-blockers, ACE inhibitors, and maintain a low-sodium diet with regular exercise.",
    "skin issues": "For acne, use benzoyl peroxide or salicylic acid. For eczema, apply moisturizers and steroid creams.",
    "allergies": "Antihistamines like loratadine or cetirizine help. Severe allergic reactions may require an EpiPen.",
    "default": "I'm not sure, but I recommend consulting a doctor for proper diagnosis and treatment."
}

# ✅ Function to Identify the Most Relevant Medical Advice
def get_medical_advice(user_input):
    for key in HEALTH_CONTEXT.keys():
        if key in user_input.lower():
            return HEALTH_CONTEXT[key]
    return HEALTH_CONTEXT["default"]

# ✅ AI Chatbot Response Generator
def healthcare_chatbot(user_input):
    medical_advice = get_medical_advice(user_input)

    # Generate a conversational response
    prompt = f"You are a helpful medical assistant. A user asked: {user_input}\nYour response should be clear and informative: "
    response = text_gen_pipeline(prompt, max_length=100, num_return_sequences=1)

    # Extract response text
    ai_response = response[0]["generated_text"].split(":")[-1].strip()

    # Ensure a reliable response
    return ai_response if len(ai_response.split()) > 5 else medical_advice
    

st.title("🩺 AI Health Assistant")
st.subheader("Hello! How can I assist you today?")

user_input = st.text_input("Type your health-related question here...", help="E.g., 'I'm having a fever. What should I do?'")

if st.button("Ask"):
    if user_input:
        with st.spinner("Processing...💭"):
            response = healthcare_chatbot(user_input)
        st.markdown(f"**🤖 AI Response:** {response}")
    else:
        st.warning("⚠️ Please enter a health-related question!")

st.markdown("---")
st.info("💡 **Disclaimer:** This AI health assistant provides general medical advice. Always consult a healthcare professional for serious concerns.")
