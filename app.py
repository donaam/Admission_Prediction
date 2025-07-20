import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open("random_forest_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Graduate Admission Predictor")
st.write("Predict whether a student is likely to be admitted.")

# Input form


def user_input_features():
    gre_score = st.slider("GRE Score (out of 340)", 260, 340, 320)
    toefl_score = st.slider("TOEFL Score (out of 120)", 90, 120, 110)
    university_rating = st.selectbox("University Rating (1-5)",
                                     [1, 2, 3, 4, 5])
    sop = st.slider("SOP Strength (1.0 - 5.0)", 1.0, 5.0, 3.5, step=0.5)
    lor = st.slider("LOR Strength (1.0 - 5.0)", 1.0, 5.0, 3.0, step=0.5)
    cgpa = st.slider("CGPA (out of 10)", 6.0, 10.0, 8.5)
    research = st.radio("Research Experience", ['No', 'Yes'])
    research_binary = 1 if research == 'Yes' else 0

    # Matching the exact training column names (including space in 'LOR ')
    features = pd.DataFrame({
        'GRE Score': [gre_score],
        'TOEFL Score': [toefl_score],
        'University Rating': [university_rating],
        'SOP': [sop],
        'LOR ': [lor],  # <-- note the space
        'CGPA': [cgpa],
        'Research': [research_binary]
    })
    return features

# Get input


input_df = user_input_features()

# Show input
st.subheader("Input Data")
st.write(input_df)

# Prediction
if st.button("Predict Admission"):
    # Ensure column order and names match
    prediction = model.predict(input_df)[0]
    result = "Admitted" if prediction == 1 else "Not Admitted"
    st.subheader("Result")
    st.success(result)

# Show probability Display of admission
proba = model.predict_proba(input_df)[0][1]  # Probability of class 1
st.info(f"Estimated Probability of Admission: {proba:.2%}")
