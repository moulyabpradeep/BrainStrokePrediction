import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the trained model, scaler, and label encoders
with open('stroke_prediction_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('label_encoders.pkl', 'rb') as file:
    label_encoders = pickle.load(file)

# Load the dataset for EDA
data_path = 'stroke_data.csv'  # Path to your dataset
data = pd.read_csv(data_path)

# Set the page title and styling
st.markdown("""
    <style>
        .main {
            background-color: #f0f8ff; /* Light blue background */
            color: #000; /* Default text color */
        }
        .footer {
            background-color: #003366; /* Dark blue footer */
            color: #fff; /* White text color in footer */
        }
    </style>
""", unsafe_allow_html=True)

# Set the page title
st.title("Stroke Prediction")

# Initialize session state if not already initialized
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Navigation functions
def go_to_notes():
    st.session_state.page = 'notes'

def go_to_eda():
    st.session_state.page = 'eda'

def go_home():
    st.session_state.page = 'home'

def go_to_model_chosen():
    st.session_state.page = 'model_chosen'

# Sidebar with image
image_path = "https://www.thevascularcenter.in/wp-content/uploads/2023/06/brain-stroke-treatment.jpg"
st.sidebar.image(image_path, caption="Brain Stroke")

# Sidebar buttons for navigation
st.sidebar.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
if st.sidebar.button("Documents"):
    go_to_notes()
st.sidebar.markdown('</div>', unsafe_allow_html=True)

st.sidebar.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
if st.sidebar.button("EDA Visualizations"):
    go_to_eda()
st.sidebar.markdown('</div>', unsafe_allow_html=True)

st.sidebar.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
if st.sidebar.button("Model Chosen"):
    go_to_model_chosen()
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Main content based on the current page
if st.session_state.page == 'notes':
    st.markdown("""
        <h2 style='color: #003366;'>What is stroke?</h2>
        <p>A stroke occurs when the blood supply to part of your brain is interrupted or reduced, preventing brain tissue from getting oxygen and nutrients. Brain cells begin to die in minutes. ​A stroke is a medical emergency, and prompt treatment is crucial. Early action can reduce brain damage and other complications. Most strokes are caused by an abrupt blockage of arteries leading to the brain (ischemic stroke). Other strokes are caused by bleeding into brain tissue when a blood vessel bursts (hemorrhagic stroke). Because stroke occurs rapidly and requires immediate treatment, stroke is also called a brain attack. When the symptoms of a stroke last only a short time (less than an hour), this is called a transient ischemic attack (TIA) or mini-stroke.</p>
        
        <h2 style='color: #003366;'>What are the symptoms of stroke?</h2>
        <p>The effects of a stroke depend on which part of the brain is injured, and how severely it is injured. Since different parts of the brain control different areas and functions, it is usually the area immediately surrounding the stroke that is affected.</p>
        <p>Strokes may cause:</p>
        <ul>
            <li>Sudden weakness in limbs</li>
            <li>Loss of sensation</li>
            <li>Difficulty with speaking, seeing, or walking</li>
        </ul>
        <p>Sometimes people with stroke have a headache, but stroke can also be completely painless. It is very important to recognize the warning signs of stroke and to get immediate medical attention if they occur.</p>
    """, unsafe_allow_html=True)

    if st.button("Go Back"):
        go_home()
elif st.session_state.page == 'eda':
    st.write("## EDA Visualizations")

    # Generate and display EDA plots
    st.write("### Distribution of Age")
    fig, ax = plt.subplots()
    sns.histplot(data['age'], kde=True, ax=ax)
    st.pyplot(fig)

    st.write("### Gender Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='gender', data=data, ax=ax)
    st.pyplot(fig)

    st.write("### Average Glucose Level by Stroke")
    fig, ax = plt.subplots()
    sns.boxplot(x='stroke', y='avg_glucose_level', data=data, ax=ax)
    st.pyplot(fig)

    st.write("### BMI Distribution by Stroke")
    fig, ax = plt.subplots()
    sns.boxplot(x='stroke', y='bmi', data=data, ax=ax)
    st.pyplot(fig)
    
    if st.button("Go Back"):
        go_home()
elif st.session_state.page == 'model_chosen':
    st.write("## Model Chosen")
    st.image("image1.png", caption="Model Image 1")
    st.markdown("""
        <h6 style='color: #003366;'>We chose the Random Forest Classifier due to its high accuracy of nearly 95% 
        and an AUC of 80%, which demonstrates strong overall performance in distinguishing between stroke and non-stroke cases.
        Despite its low recall and precision, the model's robustness and ability to handle complex datasets make it a suitable choice.
        </h6><br/>
        """, unsafe_allow_html=True)
    
    st.image("image2.png", caption="Model Image 2")
    
    st.markdown("""
        <h6 style='color: #003366;'>
The confusion matrix shows the model’s performance on stroke prediction. Out of 1,945 predictions:

* True Negatives (902): The model correctly identified 902 non-stroke cases.
* False Positives (73): The model incorrectly predicted 73 non-stroke cases as stroke.
* False Negatives (37): The model missed 37 actual stroke cases, predicting them as non-stroke.
* True Positives (933): The model correctly identified 933 stroke cases.

This matrix reflects a high number of true positives and true negatives, but also highlights areas for improvement in reducing false positives and false negatives.
</h6><br/>""", unsafe_allow_html=True)
    
    
    st.image("image3.png", caption="Model Image 3")
    st.markdown("""
        <h6 style='color: #003366;'>
    The ROC (Receiver Operating Characteristic) curve illustrates the performance of the classification model. The curve is plotted with the True Positive Rate (TPR) against the False Positive Rate (FPR) at various threshold settings. 

Key points:
* The curve hugs the top left corner, indicating a high True Positive Rate and a low False Positive Rate.
* The Area Under the Curve (AUC) is 0.99, suggesting an excellent model performance with high discriminative ability. 
* The diagonal dashed line represents the performance of a random classifier (AUC = 0.5). The model's ROC curve being well above this line confirms that it significantly outperforms random guessing.
</h6><br/>""", unsafe_allow_html=True)
    
    st.image("image4.png", caption="Model Image 4")
    st.markdown("""<h6 style='color: #003366;'>
The pie chart illustrates this distribution, highlighting the predominance of correct predictions and the relatively low error rate:

* The model's accuracy is 94.3%, indicating excellent performance.
* Correct predictions constitute 94.3% of the total predictions.
* Incorrect predictions make up only 5.7% of the total. </h6><br/>""", unsafe_allow_html=True) 
    
    
    if st.button("Go Back"):
        go_home()
else:
    st.header("Fill in the details below to predict the likelihood of a stroke")

    # Input fields with initial values and session state management
    age = st.number_input("Age", min_value=0, max_value=120, value=25, step=1, key="age")
    gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="gender")
    hypertension = st.selectbox("Hypertension", ["Yes", "No"], key="hypertension")
    heart_disease = st.selectbox("Heart Disease", ["Yes", "No"], key="heart_disease")
    ever_married = st.selectbox("Ever Married", ["Yes", "No"], key="ever_married")
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "Children", "Never_worked"], key="work_type")
    residence_type = st.selectbox("Residence Type", ["Urban", "Rural"], key="residence_type")
    avg_glucose_level = st.text_input("Average Glucose Level", key="avg_glucose_level")
    bmi = st.text_input("BMI", key="bmi")
    smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes"], key="smoking_status")

    # Validate and preprocess input data
    def is_valid_float(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    # Predict button
    if st.button("Predict"):
        if not avg_glucose_level or not bmi:
            st.error("Please fill in all mandatory fields.")
        elif not is_valid_float(avg_glucose_level) or not is_valid_float(bmi):
            st.error("Please enter valid float values for Average Glucose Level and BMI.")
        else:
            # Preprocess input data
            input_data = {
                'gender': gender,
                'age': age,
                'hypertension': 1 if hypertension == "Yes" else 0,
                'heart_disease': 1 if heart_disease == "Yes" else 0,
                'avg_glucose_level': float(avg_glucose_level),
                'bmi': float(bmi),
                'smoking_status': smoking_status
            }

            input_df = pd.DataFrame([input_data])

            for column in label_encoders:
                input_df[column] = label_encoders[column].transform(input_df[column])

            # Standardize the features
            input_scaled = scaler.transform(input_df)

            # Make prediction
            prediction = model.predict(input_scaled)

            # Display results
            result_placeholder = st.empty()
            if prediction[0] == 1:
                result_placeholder.markdown("""
                    <div style='background-color: #ff4c4c; color: white; text-align:center; font-size: 24px; padding: 20px; border-radius: 8px;'>
                        Yes (Likely to get Brain Stroke)
                    </div>
                """, unsafe_allow_html=True)
            else:
                result_placeholder.markdown("""
                    <div style='background-color: #4caf50; color: white; text-align:center; font-size: 24px; padding: 20px; border-radius: 8px;'>
                        No (You are safe, you won't get Brain Stroke)
                    </div>
                """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <style>
        .footer {
            position: relative;
            bottom: 0;
            width: 100%;
            background-color: #003366; /* Dark blue footer */
            color: #fff; /* White text color in footer */
            text-align: center;
            padding: 10px;
            box-shadow: 0 -2px 5px rgba(0,0,0,0.1);
            margin-top: 20px;
        }
    </style>
    <div class="footer">
        <p>&copy; 2024 <a href="mailto:moulyabangalore.pradeep@edu.sait.ca" style="color: #fff;">Moulya Pradeep</a> and <a href="mailto:utkarsh.dogra@edu.sait.ca" style="color: #fff;">Utkarsh Dogra</a> | All rights reserved.</p>
    </div>
""", unsafe_allow_html=True)
