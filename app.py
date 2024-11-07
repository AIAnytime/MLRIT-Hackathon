# disease_app.py (Simplified Version with Only Multiselect)

# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# 1. Load the Trained Model and Symptom Binarizer
# -----------------------------

@st.cache_resource()
def load_model():
    clf = joblib.load('disease_classifier.joblib')
    mlb = joblib.load('symptom_binarizer.joblib')
    known_symptoms = joblib.load('known_symptoms.joblib')
    return clf, mlb, known_symptoms

clf, mlb, known_symptoms = load_model()

# -----------------------------
# 2. Define the Streamlit App Layout
# -----------------------------

# Set the title of the app
st.title("Disease Prediction Based on Symptoms")

st.write("""
### Select the symptoms you are experiencing, and we'll predict the most probable disease.
""")

# -----------------------------
# 3. User Input for Symptoms (Only Multiselect)
# -----------------------------

# Using Multiselect for input
selected_symptoms = st.multiselect(
    "Select the symptoms you are experiencing:",
    options=known_symptoms,
    default=[]
)

# Preprocess the input
def preprocess_input(selected_symptoms, known_symptoms):
    # Replace spaces with underscores and lowercase the symptoms to match training
    input_symptoms = [symptom.replace(" ", "_").lower() for symptom in selected_symptoms]
    return input_symptoms

valid_symptoms = preprocess_input(selected_symptoms, known_symptoms)

# -----------------------------
# 4. Prediction
# -----------------------------

if st.button("Predict Disease"):
    if not valid_symptoms:
        st.error("Please select at least one symptom to make a prediction.")
    else:
        # Create input DataFrame
        input_data = pd.DataFrame({'symptoms': [valid_symptoms]})
        
        # Transform using the loaded MultiLabelBinarizer
        input_encoded = mlb.transform(input_data['symptoms'])
        
        # Convert to DataFrame with known symptoms as columns
        input_encoded = pd.DataFrame(input_encoded, columns=mlb.classes_)
        
        # Predict using the loaded model
        prediction = clf.predict(input_encoded)
        prediction_proba = clf.predict_proba(input_encoded)
        
        # Get the probability of the predicted class
        predicted_disease = prediction[0]
        disease_index = np.where(clf.classes_ == predicted_disease)[0][0]
        disease_probability = prediction_proba[0][disease_index]
        
        # Display the prediction
        st.success(f"The most probable disease is **{predicted_disease}** with a probability of **{disease_probability * 100:.2f}%**.")
        
        # Display probabilities for all classes
        st.write("### Prediction Probabilities:")
        proba_df = pd.DataFrame(prediction_proba, columns=clf.classes_)
        proba_df = proba_df.T.rename(columns={0: 'Probability'})
        proba_df = proba_df.sort_values(by='Probability', ascending=False)
        st.dataframe(proba_df)
