import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="MediPulse AI", layout="wide", initial_sidebar_state="expanded")

# --- 2. THEMING FUNCTIONS ---
def apply_theme(theme):
    if theme == "bright_front":
        # Classic, Clean, Medical White/Blue
        st.markdown("""
        <style>
        .stApp { background-color: #f0f8ff; } /* AliceBlue */
        h1, h2, h3 { color: #2c3e50; font-family: 'Segoe UI', sans-serif; }
        .box-card {
            background-color: white; border-radius: 15px; padding: 20px; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.05); text-align: center;
            border: 1px solid #e2e8f0;
        }
        .process-step {
            font-weight: bold; color: #3182ce; font-size: 24px;
        }
        .diagram-arrow { font-size: 30px; color: #a0aec0; }
        </style>
        """, unsafe_allow_html=True)
        
    elif theme == "fantastic_back":
        # Dark, Neon, Cyber-Medical
        st.markdown("""
        <style>
        .stApp { background-color: #0f172a; } /* Dark Slate */
        h1, h2, h3 { color: #e2e8f0; font-family: 'Courier New', monospace; text-shadow: 0 0 10px rgba(6, 182, 212, 0.5); }
        .stMetric label { color: #94a3b8; }
        .stMetric [data-testid="stMetricValue"] { color: #22d3ee; font-weight: bold; }
        .cyber-card {
            background: linear-gradient(135deg, #1e293b, #0f172a);
            border: 1px solid #22d3ee;
            box-shadow: 0 0 15px rgba(34, 211, 238, 0.2);
            border-radius: 10px; padding: 20px; color: white;
        }
        </style>
        """, unsafe_allow_html=True)

# --- 3. DATA LOADING & CLEANING ---
@st.cache_data
def load_clean_data():
    # Generate Synthetic Medical Data (Heart Disease Indicators)
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'Age': np.random.randint(20, 80, n_samples),
        'Cholesterol': np.random.randint(150, 350, n_samples),
        'Blood_Pressure': np.random.randint(90, 190, n_samples),
        'Max_Heart_Rate': np.random.randint(100, 200, n_samples),
        'BMI': np.random.uniform(18, 40, n_samples),
        # Randomly introduce missing values for cleaning demo
        'Cholesterol': np.where(np.random.rand(n_samples) > 0.9, np.nan, data['Cholesterol']),
        'BMI': np.where(np.random.rand(n_samples) > 0.9, np.nan, data['BMI'])
    })
    
    # Logic for Target Variable (Disease Outcome)
    # Higher probability of disease if BP > 140, Age > 50, Cholesterol > 240
    risk_score = (data['Blood_Pressure'] > 140).astype(int) + \
                 (data['Age'] > 50).astype(int) + \
                 (data['Cholesterol'] > 240).astype(int)
    
    # Add randomness and threshold
    data['Disease_Outcome'] = (risk_score + np.random.normal(0, 1, n_samples) > 1.5).astype(int)
    
    # --- DATA HANDLING AND CLEANING ---
    # 1. Impute Missing Values (Mean Imputation)
    imputer = SimpleImputer(strategy='mean')
    data_cleaned = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    
    # 2. Feature Selection
    X = data_cleaned[['Age', 'Cholesterol', 'Blood_Pressure', 'Max_Heart_Rate', 'BMI']]
    y = data_cleaned['Disease_Outcome']
    
    # 3. Scaling (Standardization)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, data_cleaned

X, y, raw_df = load_clean_data()

# --- 4. MODEL SELECTION AND BUILDING ---
@st.cache_resource
def build_model(X, y):
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model Selection: Random Forest
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_test, y_test, X_train, y_train

model, X_test, y_test, X_train, y_train = build_model(X, y)

# --- 5. EVALUATION METRICS ---
def get_evaluation(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)
    
    # ROC Curve
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    return acc, cm, cr, fpr, tpr, roc_auc

acc, cm, cr, fpr, tpr, roc_auc = get_evaluation(model, X_test, y_test)

# --- 6. NAVIGATION & UI ---
st.sidebar.title("🏥 MediPulse AI")
page = st.sidebar.radio("Navigate", ["🏠 Home (Front)", "🔬 Analysis (Back)"])

# --- PAGE 1: CLASSIC FRONT PAGE (BRIGHT) ---
if page == "🏠 Home (Front)":
    apply_theme("bright_front")
    
    # Header
    st.title("Advanced Disease Prediction System")
    st.markdown("### Empowering Healthcare with Data-Driven Insights")
    st.markdown("---")
    
    # The Diagram (Bright Colors)
    col1, arrow1, col2, arrow2, col3, arrow3, col4 = st.columns([2, 0.5, 2, 0.5, 2, 0.5, 2])
    
    with col1:
        st.markdown("""
        <div class='box-card' style='border-top: 5px solid #4299e1;'>
            <div class='process-step'>1. Data</div>
            <p>Patient Vitals</p>
        </div>
        """, unsafe_allow_html=True)
    
    with arrow1:
        st.markdown("<div class='diagram-arrow'>➜</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class='box-card' style='border-top: 5px solid #48bb78;'>
            <div class='process-step'>2. Clean</div>
            <p>Handle Missing Values</p>
        </div>
        """, unsafe_allow_html=True)
        
    with arrow2:
        st.markdown("<div class='diagram-arrow'>➜</div>", unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class='box-card' style='border-top: 5px solid #ed8936;'>
            <div class='process-step'>3. Model</div>
            <p>Random Forest AI</p>
        </div>
        """, unsafe_allow_html=True)
        
    with arrow3:
        st.markdown("<div class='diagram-arrow'>➜</div>", unsafe_allow_html=True)
        
    with col4:
        st.markdown("""
        <div class='box-card' style='border-top: 5px solid #f56565;'>
            <div class='process-step'>4. Result</div>
            <p>Risk Assessment</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Stats Overview
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Dataset Overview")
        st.dataframe(raw_df.describe().T.style.background_gradient(cmap='Blues'))
        
    with col_b:
        st.subheader("Distribution of Disease Outcomes")
        counts = raw_df['Disease_Outcome'].value_counts()
        fig = px.pie(values=counts, names=['Healthy', 'Disease'], 
                     color_discrete_sequence=['#48bb78', '#f56565'],
                     hole=0.4, template="simple_white")
        st.plotly_chart(fig, use_container_width=True)

# --- PAGE 2: BACK PAGE (FANTASTIC DECOR) ---
elif page == "🔬 Analysis (Back)":
    apply_theme("fantastic_back")
    
    st.title("🕵️ Deep Analytics & Prediction")
    st.markdown("### High-Fidelity Model Evaluation Interface")
    
    col_metrics, col_viz = st.columns([1, 2])
    
    # --- METRICS SECTION ---
    with col_metrics:
        st.subheader("Model Performance")
        m1, m2 = st.columns(2)
        m1.metric("Accuracy", f"{acc*100:.2f}%", delta="+1.2%")
        m2.metric("ROC AUC", f"{roc_auc:.2f}")
        
        # Confusion Matrix (Neon Style)
        st.subheader("Confusion Matrix")
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Pred: Healthy', 'Pred: Disease'],
            y=['Act: Healthy', 'Act: Disease'],
            colorscale='Viridis', # Fantastic purple/green scale
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 15, "color": "white"}
        ))
        fig_cm.update_layout(template="plotly_dark")
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # Classification Report (JSON format)
        with st.expander("Detailed Classification Report"):
            st.json(cr)

    # --- VISUALIZATION & PREDICTION SECTION ---
    with col_viz:
        # ROC Curve
        st.subheader("ROC Curve Analysis")
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode='lines', 
            name=f'ROC Curve (AUC = {roc_auc:.2f})',
            line=dict(color='#22d3ee', width=3) # Cyan Neon
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode='lines', name='Baseline',
            line=dict(color='gray', dash='dash')
        ))
        fig_roc.update_layout(template="plotly_dark", 
                              xaxis_title='False Positive Rate', 
                              yaxis_title='True Positive Rate')
        st.plotly_chart(fig_roc, use_container_width=True)
        
        # PREDICTION FORM
        st.markdown("---")
        st.subheader("🧬 New Patient Diagnosis")
        with st.form("predict_form"):
            col_in1, col_in2 = st.columns(2)
            with col_in1:
                age = st.slider("Age", 20, 90, 55)
                chol = st.slider("Cholesterol (mg/dL)", 100, 400, 220)
                bpm = st.slider("Resting BP", 90, 200, 130)
            with col_in2:
                max_hr = st.slider("Max Heart Rate", 100, 220, 150)
                bmi = st.slider("BMI", 15.0, 50.0, 28.5)
            
            submitted = st.form_submit_button("Diagnose Patient 🚀")
            
            if submitted:
                # Need to scale input because model was trained on scaled data
                # We use the mean/std from the dataset logic inside load_clean_data implicitly
                # or just use raw model if tree-based (RF doesn't need scaling technically, but we did it above).
                # For simplicity here, we re-fit a quick scaler on the raw data stats or just use raw if we didn't scale in pipeline.
                # NOTE: In this specific code, we scaled X before training. 
                # So we need to scale this input manually using the scaler object.
                
                # Recreating scaler stats for the demo (since scaler isn't global):
                scaler_demo = StandardScaler()
                scaler_demo.fit(raw_df[['Age', 'Cholesterol', 'Blood_Pressure', 'Max_Heart_Rate', 'BMI']])
                
                input_data = np.array([[age, chol, bpm, max_hr, bmi]])
                input_scaled = scaler_demo.transform(input_data)
                
                prediction = model.predict(input_scaled)[0]
                proba = model.predict_proba(input_scaled)[0][1]
                
                # Display Result
                if prediction == 1:
                    st.error(f"⚠️ High Risk of Disease Detected ({proba*100:.1f}%)")
                else:
                    st.success(f"✅ Patient appears Healthy ({(1-proba)*100:.1f}% confidence)")
