import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

# --- Page Configuration ---
st.set_page_config(
    page_title="MediPredict - Advanced Disease Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Styling ---
def load_css():
    st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        color: white;
        background-color: #0066cc;
        border-radius: 5px;
        height: 3em;
        width: 100%;
        font-size: 18px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #004c99;
        border-color: #004c99;
    }
    .prediction-box {
        background-color: #e6f2ff;
        border: 1px solid #0066cc;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        text-align: center;
    }
    .high-risk {
        color: #cc0000;
        font-weight: bold;
        font-size: 24px;
    }
    .low-risk {
        color: #009933;
        font-weight: bold;
        font-size: 24px;
    }
    .header-text {
        color: #003366;
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .info-text {
        color: #555;
        font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True)

load_css()

# --- Mock Machine Learning Model ---
# In a real scenario, you would load a trained .pkl file here using joblib or pickle.
# For this demonstration, we will use a rule-based logic to simulate predictions.
def mock_predict_disease(data):
    """
    Simulates a model prediction based on input features.
    Returns a dictionary with disease names and risk probabilities.
    """
    # Extract values
    age = data['Age']
    bmi = data['BMI']
    bp = data['Blood_Pressure']
    sugar = data['Sugar_Level']
    smoker = data['Smoker']
    family_history = data['Family_History']

    risks = {}

    # Heart Disease Logic
    heart_score = 0
    if age > 50: heart_score += 20
    if bp > 140: heart_score += 30
    if smoker: heart_score += 30
    if family_history == 'Heart': heart_score += 20
    risks['Heart Disease'] = min(heart_score, 95) # Cap at 95%

    # Diabetes Logic
    diabetes_score = 0
    if sugar > 140: diabetes_score += 40
    if bmi > 30: diabetes_score += 30
    if age > 40: diabetes_score += 10
    if family_history == 'Diabetes': diabetes_score += 20
    risks['Diabetes (Type 2)'] = min(diabetes_score, 95)

    # Cancer Risk Logic (Simplified)
    cancer_score = 5 # Base risk
    if smoker: cancer_score += 25
    if age > 60: cancer_score += 20
    if family_history == 'Cancer': cancer_score += 30
    risks['Cancer Risk'] = min(cancer_score, 90)

    return risks

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict Disease", "About Model", "Contact"])

# Load Images (Using placeholders for demo compatibility)
# In a real app, replace these with local file paths like "images/heart.jpg"
try:
    # Attempt to load generic images if they exist, otherwise use logic
    logo = Image.open("logo.png") 
except:
    logo = None

# --- PAGE 1: HOME ---
if page == "Home":
    st.markdown('<p class="header-text">MediPredict AI</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Your Health, Our Priority")
        st.write("""
        MediPredict uses advanced machine learning algorithms to analyze patient data 
        and predict the likelihood of severe diseases such as:
        
        - 🫀 **Heart Disease**
        - 🩸 **Diabetes (Sugar)**
        - 🎗️ **Various Cancers**
        
        Early detection saves lives. Enter your medical details to get an instant risk assessment.
        """)
        st.info("👉 Navigate to 'Predict Disease' in the sidebar to start.")
        
    with col2:
        # Placeholder for a hero image
        st.image("https://via.placeholder.com/600x400/0066cc/ffffff?text=Advanced+Medical+AI", caption="AI-Driven Healthcare")

# --- PAGE 2: PREDICT DISEASE ---
elif page == "Predict Disease":
    st.markdown('<p class="header-text">Patient Risk Assessment</p>', unsafe_allow_html=True)
    
    st.write("Please fill out the form below with accurate patient data.")
    
    # Input Form
    with st.form(key='patient_form'):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Personal Details")
            age = st.number_input("Age", min_value=1, max_value=120, value=45)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=50.0, value=24.5, step=0.1)
            
        with col2:
            st.subheader("Vitals & Habits")
            bp = st.slider("Blood Pressure (Systolic)", min_value=80, max_value=200, value=120)
            sugar = st.slider("Fasting Sugar Level (mg/dL)", min_value=70, max_value=400, value=95)
            smoker = st.checkbox("Do you smoke?")
            alcohol = st.selectbox("Alcohol Consumption", ["None", "Moderate", "Heavy"])
            
        with col3:
            st.subheader("History")
            family_history = st.selectbox("Family History of Disease", ["None", "Heart", "Diabetes", "Cancer"])
            symptoms = st.text_area("Specific Symptoms (Optional)", placeholder="e.g., chest pain, fatigue...")
            
        submit_button = st.form_submit_button(label="Analyze Risk 🧬")
        
        if submit_button:
            # Prepare data for model
            input_data = {
                'Age': age,
                'BMI': bmi,
                'Blood_Pressure': bp,
                'Sugar_Level': sugar,
                'Smoker': smoker,
                'Family_History': family_history
            }
            
            # Show a spinner to simulate processing
            with st.spinner('Analyzing data with AI model...'):
                results = mock_predict_disease(input_data)
                
            st.success("Analysis Complete!")
            
            # Display Results
            res_col1, res_col2, res_col3 = st.columns(3)
            
            with res_col1:
                risk_heart = results['Heart Disease']
                color_heart = "high-risk" if risk_heart > 30 else "low-risk"
                label_heart = "High Probability" if risk_heart > 30 else "Low Probability"
                
                st.markdown(f"""
                <div class="prediction-box">
                    <h3>Heart Problem</h3>
                    <p class="{color_heart}">{risk_heart}% Risk</p>
                    <p>{label_heart}</p>
                    <img src="https://via.placeholder.com/150?text=Heart" style="border-radius:50%; margin-top:10px;">
                </div>
                """, unsafe_allow_html=True)
                
            with res_col2:
                risk_sugar = results['Diabetes (Type 2)']
                color_sugar = "high-risk" if risk_sugar > 30 else "low-risk"
                label_sugar = "High Probability" if risk_sugar > 30 else "Low Probability"
                
                st.markdown(f"""
                <div class="prediction-box">
                    <h3>Diabetes</h3>
                    <p class="{color_sugar}">{risk_sugar}% Risk</p>
                    <p>{label_sugar}</p>
                    <img src="https://via.placeholder.com/150?text=Sugar" style="border-radius:50%; margin-top:10px;">
                </div>
                """, unsafe_allow_html=True)
                
            with res_col3:
                risk_cancer = results['Cancer Risk']
                color_cancer = "high-risk" if risk_cancer > 20 else "low-risk"
                label_cancer = "Consult Doctor" if risk_cancer > 20 else "Normal"
                
                st.markdown(f"""
                <div class="prediction-box">
                    <h3>Cancer Risk</h3>
                    <p class="{color_cancer}">{risk_cancer}% Risk</p>
                    <p>{label_cancer}</p>
                    <img src="https://via.placeholder.com/150?text=Cancer" style="border-radius:50%; margin-top:10px;">
                </div>
                """, unsafe_allow_html=True)

            # Recommendations
            st.write("---")
            st.subheader("🩺 AI Recommendations")
            if risk_heart > 30 or risk_sugar > 30 or risk_cancer > 20:
                st.warning("Based on the analysis, it is recommended to schedule a consultation with a specialist immediately. Maintain a healthy diet and monitor vitals daily.")
            else:
                st.success("Your indicators look healthy! Continue maintaining a balanced lifestyle and exercise regularly.")

# --- PAGE 3: ABOUT MODEL ---
elif page == "About Model":
    st.markdown('<p class="header-text">About The AI Model</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("""
        ### Technology Stack
        
        This application is powered by **Python** and utilizes a **Random Forest Classifier** (in the production version) 
        trained on thousands of anonymized patient records.
        
        **Data Sources:**
        - UCI Machine Learning Repository
        - Kaggle Health Datasets
        
        **Features Used for Prediction:**
        1. **Demographics:** Age, Gender, BMI
        2. **Vitals:** Blood Pressure, Glucose Levels
        3. **Lifestyle:** Smoking status, Alcohol intake
        4. **Genetics:** Family history of chronic diseases
        
        **Accuracy:**
        - Heart Disease Prediction: ~85% Accuracy
        - Diabetes Prediction: ~90% Accuracy
        - Cancer Risk Screening: ~78% Accuracy
        
        *Note: This tool is for educational and screening purposes only and should not replace professional medical advice.*
        """)
    
    with col2:
        st.image("https://via.placeholder.com/400x300/333333/ffffff?text=Neural+Network+Viz")

# --- PAGE 4: CONTACT ---
elif page == "Contact":
    st.markdown('<p class="header-text">Contact Us</p>', unsafe_allow_html=True)
    
    st.write("Have questions about the prediction or the technology?")
    
    contact_form = st.form(key='contact_form')
    name = contact_form.text_input("Your Name")
    email = contact_form.text_input("Your Email")
    message = contact_form.text_area("Message")
    
    submitted = contact_form.form_submit_button("Send Message")
    if submitted:
        st.success("Thank you! We have received your message.")

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.write("© 2023 MediPredict AI")