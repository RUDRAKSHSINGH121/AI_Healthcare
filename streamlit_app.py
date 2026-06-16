import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import joblib

# Ensure the working directory is the script's directory so relative paths work
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE:
    os.chdir(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Import local ML modules
try:
    from ml_models import MLModelTrainer
    from data_preprocessing import DataPreprocessor
except ImportError:
    # Fallback placeholders if imports fail on deployment
    st.error("Could not import ml_models or data_preprocessing. Please ensure all project files are uploaded.")

# Set page config
st.set_page_config(
    page_title="AI Healthcare Diagnostics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for a modern, premium feel
st.markdown("""
    <style>
    /* Premium visual layout */
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background: linear-gradient(135deg, #0d6efd 0%, #0a58ca 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(13, 110, 253, 0.15);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(13, 110, 253, 0.25);
        background: linear-gradient(135deg, #0b5edd 0%, #084ebd 100%);
    }
    .card {
        background-color: white;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 24px;
        border-left: 5px solid #0d6efd;
    }
    .card-success {
        border-left: 5px solid #198754;
    }
    .card-warning {
        border-left: 5px solid #ffc107;
    }
    .card-danger {
        border-left: 5px solid #dc3545;
    }
    h1, h2, h3 {
        color: #1e293b;
        font-weight: 700;
    }
    .text-muted {
        color: #64748b;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize modules
@st.cache_resource
def get_trainer_and_preprocessor():
    return MLModelTrainer(), DataPreprocessor()

trainer, preprocessor = get_trainer_and_preprocessor()

# Helper function to check if models exist
def get_model_status():
    models = {
        'heart': 'models/heart_disease_model.pkl',
        'diabetes': 'models/diabetes_model.pkl',
        'breast_cancer': 'models/breast_cancer_model.pkl',
        'liver': 'models/liver_disease_model.pkl',
        'kidney': 'models/kidney_disease_model.pkl'
    }
    status = {}
    for name, path in models.items():
        status[name] = os.path.exists(path)
    return status

# Sidebar Layout
with st.sidebar:
    st.image("https://img.icons8.com/external-flat-icons-inmotus-design/67/external-doctor-healthy-lifestyle-flat-icons-inmotus-design.png", width=64)
    st.markdown("# **AI Diagnostics**")
    st.markdown("---")
    
    # Navigation option
    navigation = st.radio(
        "Go to page:",
        ["🏠 Dashboard / Home", 
         "❤️ Heart Disease Prediction", 
         "🩸 Diabetes Prediction", 
         "🎗️ Breast Cancer Prediction", 
         "🫁 Liver Disease Prediction", 
         "🫘 Kidney Disease Prediction"]
    )
    
    st.markdown("---")
    st.markdown("### **Model Status**")
    
    status = get_model_status()
    all_trained = all(status.values())
    
    for model_name, trained in status.items():
        label = model_name.replace('_', ' ').title()
        if trained:
            st.markdown(f"🟢 **{label}**: Ready")
        else:
            st.markdown(f"🔴 **{label}**: Missing")
            
    st.markdown("---")
    
    # Quick Action to Train Models
    if not all_trained:
        st.warning("Some models are missing! Please train them to enable diagnostic predictions.")
    
    if st.button("🔄 Train / Reset All Models"):
        with st.spinner("Training models... This might take a few seconds."):
            try:
                # Preprocess/Create sample datasets if needed
                preprocessor.create_sample_datasets()
                # Train all models
                trainer.train_heart_disease_model()
                trainer.train_diabetes_model()
                trainer.train_breast_cancer_model()
                trainer.train_liver_disease_model()
                trainer.train_kidney_disease_model()
                st.success("All models trained successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error training models: {e}")

# Main Content Pages
if navigation == "🏠 Dashboard / Home":
    st.markdown("# 🏥 AI-Driven Healthcare Diagnostics")
    st.markdown("Welcome to the **AI Healthcare Diagnostics System**. This platform leverages advanced Machine Learning algorithms to assess patient risk levels for multiple health conditions.")
    st.markdown("---")
    
    # Stats column
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Diagnostic Profiles", value="5")
    with col2:
        st.metric(label="System Status", value="Online" if all_trained else "Models Untrained", delta="Ready" if all_trained else "Action Required")
    with col3:
        st.metric(label="Algorithms Used", value="Random Forest, SVM, GB, LR")
        
    st.markdown("### Supported Disease Risk Assessments")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class="card">
            <h4>❤️ Heart Disease Prediction</h4>
            <p class="text-muted">Analyzes 13 cardiovascular metrics (blood pressure, cholesterol, resting ECG, maximum heart rate) to determine risk probability.</p>
        </div>
        <div class="card">
            <h4>🩸 Diabetes Prediction</h4>
            <p class="text-muted">Evaluates diabetes probability using 8 health indicators (glucose, insulin levels, BMI, pregnancy history, age).</p>
        </div>
        <div class="card">
            <h4>🎗️ Breast Cancer Prediction</h4>
            <p class="text-muted">Classifies tumor characteristics using 10 parameters (radius mean, texture, perimeter, area, smoothness, symmetry).</p>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="card">
            <h4>🫁 Liver Disease Prediction</h4>
            <p class="text-muted">Uses liver function test results including bilirubin, proteins, enzymes (alkaline phosphatase, transferases) to detect disease.</p>
        </div>
        <div class="card">
            <h4>🫘 Kidney Disease Prediction</h4>
            <p class="text-muted">Examines 24 clinical variables (specific gravity, albumin, sugar, red blood cells, urea, creatinine, etc.) to assess chronic kidney disease risk.</p>
        </div>
        <div class="card">
            <h4>⚠️ Clinical Disclaimer</h4>
            <p class="text-muted"><b>For demonstration purposes only.</b> These assessments are powered by machine learning algorithms trained on sample data and should not substitute professional medical advice.</p>
        </div>
        """, unsafe_allow_html=True)

elif navigation == "❤️ Heart Disease Prediction":
    st.markdown("# ❤️ Heart Disease Risk Prediction")
    st.write("Fill out the patient's health parameters below to assess their cardiovascular disease risk.")
    st.markdown("---")
    
    if not status['heart']:
        st.error("Heart Disease model is not trained yet. Please click 'Train / Reset All Models' in the sidebar.")
    else:
        with st.form("heart_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input("Age (years)", min_value=1, max_value=120, value=45)
                sex = st.selectbox("Sex", options=[("Female", 0), ("Male", 1)], format_func=lambda x: x[0])
                cp = st.selectbox("Chest Pain Type", options=[
                    ("Typical Angina", 0),
                    ("Atypical Angina", 1),
                    ("Non-anginal Pain", 2),
                    ("Asymptomatic", 3)
                ], format_func=lambda x: x[0])
                trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=250, value=120)
                chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
                fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
                restecg = st.selectbox("Resting ECG Results", options=[
                    ("Normal", 0),
                    ("ST-T Wave Abnormality", 1),
                    ("Left Ventricular Hypertrophy", 2)
                ], format_func=lambda x: x[0])
                
            with col2:
                thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
                exang = st.selectbox("Exercise Induced Angina", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
                oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
                slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[
                    ("Upsloping", 0),
                    ("Flat", 1),
                    ("Downsloping", 2)
                ], format_func=lambda x: x[0])
                ca = st.selectbox("Number of Major Vessels (0-3)", options=[0, 1, 2, 3])
                thal = st.selectbox("Thalassemia", options=[
                    ("Normal", 0),
                    ("Fixed Defect", 1),
                    ("Reversible Defect", 2)
                ], format_func=lambda x: x[0])
                
            submit = st.form_submit_button("Run Heart Disease Diagnostics")
            
        if submit:
            features = [
                float(age), int(sex[1]), int(cp[1]), float(trestbps), float(chol),
                int(fbs[1]), int(restecg[1]), float(thalach), int(exang[1]),
                float(oldpeak), int(slope[1]), int(ca), int(thal[1])
            ]
            
            with st.spinner("Analyzing parameters..."):
                res = trainer.predict_heart_disease(features)
                
                # Render results nicely
                risk_color = "red" if res['risk_level'] == "High" else "orange" if res['risk_level'] == "Medium" else "green"
                card_class = "card-danger" if res['risk_level'] == "High" else "card-warning" if res['risk_level'] == "Medium" else "card-success"
                
                st.markdown(f"""
                <div class="card {card_class}">
                    <h3>Diagnostic Summary</h3>
                    <p>Risk Level: <b style="color:{risk_color}; font-size: 1.2rem;">{res['risk_level']}</b></p>
                    <p>Prediction: <b>{"Heart Disease Detected" if res['prediction'] == 1 else "No Sign of Heart Disease"}</b></p>
                </div>
                """, unsafe_allow_html=True)
                
                st.write("**Probability Analysis:**")
                st.progress(res['probability'])
                st.write(f"Confidence score: {res['probability']:.2%}")

elif navigation == "🩸 Diabetes Prediction":
    st.markdown("# 🩸 Diabetes Risk Prediction")
    st.write("Fill out the patient's physiological indicators below to assess their diabetes risk.")
    st.markdown("---")
    
    if not status['diabetes']:
        st.error("Diabetes model is not trained yet. Please click 'Train / Reset All Models' in the sidebar.")
    else:
        with st.form("diabetes_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=1)
                glucose = st.number_input("Glucose Level (mg/dl)", min_value=50, max_value=300, value=120)
                bp = st.number_input("Blood Pressure (mm Hg)", min_value=40, max_value=150, value=70)
                skin_thickness = st.number_input("Skin Thickness (mm)", min_value=5, max_value=100, value=20)
                
            with col2:
                insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=1000, value=80)
                bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=80.0, value=25.0, step=0.1)
                dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.001, format="%.3f")
                age = st.number_input("Age (years)", min_value=1, max_value=120, value=30)
                
            submit = st.form_submit_button("Run Diabetes Diagnostics")
            
        if submit:
            features = [
                int(pregnancies), float(glucose), float(bp), float(skin_thickness),
                float(insulin), float(bmi), float(dpf), float(age)
            ]
            
            with st.spinner("Analyzing parameters..."):
                res = trainer.predict_diabetes(features)
                
                risk_color = "red" if res['risk_level'] == "High" else "orange" if res['risk_level'] == "Medium" else "green"
                card_class = "card-danger" if res['risk_level'] == "High" else "card-warning" if res['risk_level'] == "Medium" else "card-success"
                
                st.markdown(f"""
                <div class="card {card_class}">
                    <h3>Diagnostic Summary</h3>
                    <p>Risk Level: <b style="color:{risk_color}; font-size: 1.2rem;">{res['risk_level']}</b></p>
                    <p>Prediction: <b>{"Diabetes Risk Detected" if res['prediction'] == 1 else "No Sign of Diabetes Risk"}</b></p>
                </div>
                """, unsafe_allow_html=True)
                
                st.write("**Probability Analysis:**")
                st.progress(res['probability'])
                st.write(f"Confidence score: {res['probability']:.2%}")

elif navigation == "🎗️ Breast Cancer Prediction":
    st.markdown("# 🎗️ Breast Cancer Prediction")
    st.write("Enter the tumor measurements from biopsy reports below to assess breast cancer risk.")
    st.markdown("---")
    
    if not status['breast_cancer']:
        st.error("Breast Cancer model is not trained yet. Please click 'Train / Reset All Models' in the sidebar.")
    else:
        with st.form("breast_cancer_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                radius_mean = st.number_input("Mean Radius", min_value=1.0, max_value=50.0, value=14.0, step=0.01)
                texture_mean = st.number_input("Mean Texture", min_value=1.0, max_value=50.0, value=19.0, step=0.01)
                perimeter_mean = st.number_input("Mean Perimeter", min_value=10.0, max_value=300.0, value=92.0, step=0.01)
                area_mean = st.number_input("Mean Area", min_value=10.0, max_value=3000.0, value=650.0, step=0.01)
                smoothness_mean = st.number_input("Mean Smoothness", min_value=0.0, max_value=1.0, value=0.096, step=0.0001, format="%.5f")
                
            with col2:
                compactness_mean = st.number_input("Mean Compactness", min_value=0.0, max_value=1.0, value=0.104, step=0.0001, format="%.5f")
                concavity_mean = st.number_input("Mean Concavity", min_value=0.0, max_value=1.0, value=0.089, step=0.0001, format="%.5f")
                concave_points_mean = st.number_input("Mean Concave Points", min_value=0.0, max_value=1.0, value=0.049, step=0.0001, format="%.5f")
                symmetry_mean = st.number_input("Mean Symmetry", min_value=0.0, max_value=1.0, value=0.181, step=0.0001, format="%.5f")
                fractal_dimension_mean = st.number_input("Mean Fractal Dimension", min_value=0.0, max_value=1.0, value=0.063, step=0.0001, format="%.5f")
                
            submit = st.form_submit_button("Run Breast Cancer Diagnostics")
            
        if submit:
            features = [
                float(radius_mean), float(texture_mean), float(perimeter_mean), float(area_mean),
                float(smoothness_mean), float(compactness_mean), float(concavity_mean),
                float(concave_points_mean), float(symmetry_mean), float(fractal_dimension_mean)
            ]
            
            with st.spinner("Analyzing measurements..."):
                res = trainer.predict_breast_cancer(features)
                
                risk_color = "red" if res['risk_level'] == "High" else "orange" if res['risk_level'] == "Medium" else "green"
                card_class = "card-danger" if res['risk_level'] == "High" else "card-warning" if res['risk_level'] == "Medium" else "card-success"
                
                st.markdown(f"""
                <div class="card {card_class}">
                    <h3>Diagnostic Summary</h3>
                    <p>Risk Level: <b style="color:{risk_color}; font-size: 1.2rem;">{res['risk_level']}</b></p>
                    <p>Prediction: <b>{"Malignant (High Risk)" if res['prediction'] == 1 else "Benign (Low Risk)"}</b></p>
                </div>
                """, unsafe_allow_html=True)
                
                st.write("**Probability Analysis:**")
                st.progress(res['probability'])
                st.write(f"Confidence score: {res['probability']:.2%}")

elif navigation == "🫁 Liver Disease Prediction":
    st.markdown("# 🫁 Liver Disease Prediction")
    st.write("Fill out the patient's liver panel results below to evaluate liver health status.")
    st.markdown("---")
    
    if not status['liver']:
        st.error("Liver Disease model is not trained yet. Please click 'Train / Reset All Models' in the sidebar.")
    else:
        with st.form("liver_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input("Age", min_value=1, max_value=120, value=40)
                gender = st.selectbox("Gender", options=[("Female", 0), ("Male", 1)], format_func=lambda x: x[0])
                tb = st.number_input("Total Bilirubin", min_value=0.0, max_value=50.0, value=1.0, step=0.1)
                db = st.number_input("Direct Bilirubin", min_value=0.0, max_value=25.0, value=0.3, step=0.1)
                alkphos = st.number_input("Alkaline Phosphotase", min_value=10, max_value=2000, value=200)
                
            with col2:
                sgpt = st.number_input("Alamine Aminotransferase (SGPT)", min_value=1, max_value=2000, value=50)
                sgot = st.number_input("Aspartate Aminotransferase (SGOT)", min_value=1, max_value=2000, value=50)
                tp = st.number_input("Total Proteins", min_value=1.0, max_value=15.0, value=6.5, step=0.1)
                alb = st.number_input("Albumin", min_value=0.5, max_value=10.0, value=3.5, step=0.1)
                ag_ratio = st.number_input("Albumin/Globulin Ratio", min_value=0.1, max_value=5.0, value=1.0, step=0.01)
                
            submit = st.form_submit_button("Run Liver Disease Diagnostics")
            
        if submit:
            features = [
                float(age), int(gender[1]), float(tb), float(db), float(alkphos),
                float(sgpt), float(sgot), float(tp), float(alb), float(ag_ratio)
            ]
            
            with st.spinner("Analyzing liver panel..."):
                res = trainer.predict_liver_disease(features)
                
                risk_color = "red" if res['risk_level'] == "High" else "orange" if res['risk_level'] == "Medium" else "green"
                card_class = "card-danger" if res['risk_level'] == "High" else "card-warning" if res['risk_level'] == "Medium" else "card-success"
                
                st.markdown(f"""
                <div class="card {card_class}">
                    <h3>Diagnostic Summary</h3>
                    <p>Risk Level: <b style="color:{risk_color}; font-size: 1.2rem;">{res['risk_level']}</b></p>
                    <p>Prediction: <b>{"Liver Disease Detected" if res['prediction'] == 1 else "No Sign of Liver Disease"}</b></p>
                </div>
                """, unsafe_allow_html=True)
                
                st.write("**Probability Analysis:**")
                st.progress(res['probability'])
                st.write(f"Confidence score: {res['probability']:.2%}")

elif navigation == "🫘 Kidney Disease Prediction":
    st.markdown("# 🫘 Chronic Kidney Disease Prediction")
    st.write("Examine chronic kidney disease (CKD) risk using the clinical parameters below.")
    st.markdown("---")
    
    if not status['kidney']:
        st.error("Kidney Disease model is not trained yet. Please click 'Train / Reset All Models' in the sidebar.")
    else:
        with st.form("kidney_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input("Age", min_value=1, max_value=120, value=50)
                bp = st.number_input("Blood Pressure (mm Hg)", min_value=40, max_value=200, value=80)
                sg = st.number_input("Specific Gravity", min_value=1.000, max_value=1.035, value=1.020, step=0.001, format="%.3f")
                al = st.selectbox("Albumin Level (0-5)", options=[0, 1, 2, 3, 4, 5])
                su = st.selectbox("Sugar Level (0-5)", options=[0, 1, 2, 3, 4, 5])
                rbc = st.selectbox("Red Blood Cells", options=[("Normal", 0), ("Abnormal", 1)], format_func=lambda x: x[0])
                pc = st.selectbox("Pus Cell", options=[("Normal", 0), ("Abnormal", 1)], format_func=lambda x: x[0])
                pcc = st.selectbox("Pus Cell Clumps", options=[("Not Present", 0), ("Present", 1)], format_func=lambda x: x[0])
                ba = st.selectbox("Bacteria", options=[("Not Present", 0), ("Present", 1)], format_func=lambda x: x[0])
                bgr = st.number_input("Blood Glucose Random (mg/dl)", min_value=50, max_value=500, value=120)
                bu = st.number_input("Blood Urea (mg/dl)", min_value=5, max_value=300, value=40)
                sc = st.number_input("Serum Creatinine (mg/dl)", min_value=0.1, max_value=20.0, value=1.2, step=0.1)
                
            with col2:
                sod = st.number_input("Sodium (mEq/L)", min_value=100, max_value=180, value=138)
                pot = st.number_input("Potassium (mEq/L)", min_value=2.0, max_value=10.0, value=4.5, step=0.1)
                hemo = st.number_input("Hemoglobin (g/dL)", min_value=3.0, max_value=22.0, value=13.0, step=0.1)
                pcv = st.number_input("Packed Cell Volume", min_value=10, max_value=60, value=40)
                wbcc = st.number_input("White Blood Cell Count", min_value=1000, max_value=30000, value=8000, step=100)
                rbcc = st.number_input("Red Blood Cell Count (millions/cmm)", min_value=1.0, max_value=10.0, value=4.5, step=0.1)
                htn = st.selectbox("Hypertension", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
                dm = st.selectbox("Diabetes Mellitus", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
                cad = st.selectbox("Coronary Artery Disease", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
                appet = st.selectbox("Appetite", options=[("Good", 0), ("Poor", 1)], format_func=lambda x: x[0])
                pe = st.selectbox("Pedal Edema", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
                ane = st.selectbox("Anemia", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
                
            submit = st.form_submit_button("Run Kidney Disease Diagnostics")
            
        if submit:
            features = [
                float(age), float(bp), float(sg), int(al), int(su), int(rbc[1]),
                int(pc[1]), int(pcc[1]), int(ba[1]), float(bgr), float(bu),
                float(sc), float(sod), float(pot), float(hemo), float(pcv),
                float(wbcc), float(rbcc), int(htn[1]), int(dm[1]), int(cad[1]),
                int(appet[1]), int(pe[1]), int(ane[1])
            ]
            
            with st.spinner("Analyzing parameters..."):
                res = trainer.predict_kidney_disease(features)
                
                # Note: Model targets could be inverted based on training encoding, let's verify.
                # In train_models.py: 0=CKD, 1=Not CKD.
                is_ckd = res['prediction'] == 0
                risk_status = "CKD (Chronic Kidney Disease) Detected" if is_ckd else "No Sign of Chronic Kidney Disease"
                
                # Invert probability visualization if prediction represents 0 as positive risk
                risk_prob = 1.0 - res['probability'] if is_ckd else res['probability']
                
                risk_color = "red" if is_ckd else "green"
                card_class = "card-danger" if is_ckd else "card-success"
                
                st.markdown(f"""
                <div class="card {card_class}">
                    <h3>Diagnostic Summary</h3>
                    <p>Risk Status: <b style="color:{risk_color}; font-size: 1.2rem;">{risk_status}</b></p>
                </div>
                """, unsafe_allow_html=True)
                
                st.write("**Probability Analysis:**")
                st.progress(risk_prob)
                st.write(f"Confidence score: {risk_prob:.2%}")
