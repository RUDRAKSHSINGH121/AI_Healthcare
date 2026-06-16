from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import numpy as np
import joblib
import os
from src.data_preprocessing import DataPreprocessor
from src.ml_models import MLModelTrainer
import json

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Initialize components
preprocessor = DataPreprocessor()
trainer = MLModelTrainer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/heart-disease')
def heart_disease():
    return render_template('heart_disease.html')

@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')

@app.route('/breast-cancer')
def breast_cancer():
    return render_template('breast_cancer.html')

@app.route('/liver-disease')
def liver_disease():
    return render_template('liver_disease.html')

@app.route('/kidney-disease')
def kidney_disease():
    return render_template('kidney_disease.html')

@app.route('/diagnosis-dashboard')
def diagnosis_dashboard():
    return render_template('diagnosis_dashboard.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict/heart', methods=['POST'])
def predict_heart_disease():
    try:
        # Get form data
        data = request.get_json()
        
        # Extract features
        features = [
            float(data['age']),
            int(data['sex']),
            int(data['cp']),
            float(data['trestbps']),
            float(data['chol']),
            int(data['fbs']),
            int(data['restecg']),
            float(data['thalach']),
            int(data['exang']),
            float(data['oldpeak']),
            int(data['slope']),
            int(data['ca']),
            int(data['thal'])
        ]
        
        # Make prediction
        result = trainer.predict_heart_disease(features)
        
        return jsonify({
            'success': True,
            'prediction': result['prediction'],
            'probability': result['probability'],
            'risk_level': result['risk_level'],
            'message': f"Heart disease risk: {result['risk_level']} (Probability: {result['probability']:.2%})"
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    try:
        # Get form data
        data = request.get_json()
        
        # Extract features
        features = [
            int(data['pregnancies']),
            float(data['glucose']),
            float(data['blood_pressure']),
            float(data['skin_thickness']),
            float(data['insulin']),
            float(data['bmi']),
            float(data['diabetes_pedigree']),
            float(data['age'])
        ]
        
        # Make prediction
        result = trainer.predict_diabetes(features)
        
        return jsonify({
            'success': True,
            'prediction': result['prediction'],
            'probability': result['probability'],
            'risk_level': result['risk_level'],
            'message': f"Diabetes risk: {result['risk_level']} (Probability: {result['probability']:.2%})"
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/predict/breast-cancer', methods=['POST'])
def predict_breast_cancer():
    try:
        # Get form data
        data = request.get_json()
        
        # Extract features
        features = [
            float(data['radius_mean']),
            float(data['texture_mean']),
            float(data['perimeter_mean']),
            float(data['area_mean']),
            float(data['smoothness_mean']),
            float(data['compactness_mean']),
            float(data['concavity_mean']),
            float(data['concave_points_mean']),
            float(data['symmetry_mean']),
            float(data['fractal_dimension_mean'])
        ]
        
        # Make prediction
        result = trainer.predict_breast_cancer(features)
        
        return jsonify({
            'success': True,
            'prediction': result['prediction'],
            'probability': result['probability'],
            'risk_level': result['risk_level'],
            'message': f"Breast cancer risk: {result['risk_level']} (Probability: {result['probability']:.2%})"
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/predict/liver-disease', methods=['POST'])
def predict_liver_disease():
    try:
        # Get form data
        data = request.get_json()
        
        # Extract features
        features = [
            float(data['age']),
            int(data['gender']),
            float(data['total_bilirubin']),
            float(data['direct_bilirubin']),
            float(data['alkaline_phosphotase']),
            float(data['alamine_aminotransferase']),
            float(data['aspartate_aminotransferase']),
            float(data['total_proteins']),
            float(data['albumin']),
            float(data['albumin_and_globulin_ratio'])
        ]
        
        # Make prediction
        result = trainer.predict_liver_disease(features)
        
        return jsonify({
            'success': True,
            'prediction': result['prediction'],
            'probability': result['probability'],
            'risk_level': result['risk_level'],
            'message': f"Liver disease risk: {result['risk_level']} (Probability: {result['probability']:.2%})"
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/predict/kidney-disease', methods=['POST'])
def predict_kidney_disease():
    try:
        # Get form data
        data = request.get_json()
        
        # Extract features
        features = [
            float(data['age']),
            float(data['bp']),
            float(data['sg']),
            int(data['al']),
            int(data['su']),
            int(data['rbc']),
            int(data['pc']),
            int(data['pcc']),
            int(data['ba']),
            float(data['bgr']),
            float(data['bu']),
            float(data['sc']),
            float(data['sod']),
            float(data['pot']),
            float(data['hemo']),
            float(data['pcv']),
            float(data['wbcc']),
            float(data['rbcc']),
            int(data['htn']),
            int(data['dm']),
            int(data['cad']),
            int(data['appet']),
            int(data['pe']),
            int(data['ane'])
        ]
        
        # Make prediction
        result = trainer.predict_kidney_disease(features)
        
        return jsonify({
            'success': True,
            'prediction': result['prediction'],
            'probability': result['probability'],
            'risk_level': result['risk_level'],
            'message': f"Kidney disease risk: {result['risk_level']} (Probability: {result['probability']:.2%})"
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/models/status')
def models_status():
    heart_model_path = 'models/heart_disease_model.pkl'
    diabetes_model_path = 'models/diabetes_model.pkl'
    breast_cancer_model_path = 'models/breast_cancer_model.pkl'
    liver_model_path = 'models/liver_disease_model.pkl'
    kidney_model_path = 'models/kidney_disease_model.pkl'
    
    return jsonify({
        'heart_model': os.path.exists(heart_model_path),
        'diabetes_model': os.path.exists(diabetes_model_path),
        'breast_cancer_model': os.path.exists(breast_cancer_model_path),
        'liver_model': os.path.exists(liver_model_path),
        'kidney_model': os.path.exists(kidney_model_path)
    })

@app.route('/train/models', methods=['POST'])
def train_models():
    try:
        if not os.path.exists('data/heart_disease.csv'):
            preprocessor.create_sample_datasets()
        
        heart_model, heart_features = trainer.train_heart_disease_model()
        diabetes_model, diabetes_features = trainer.train_diabetes_model()
        breast_cancer_model, breast_cancer_features = trainer.train_breast_cancer_model()
        liver_model, liver_features = trainer.train_liver_disease_model()
        kidney_model, kidney_features = trainer.train_kidney_disease_model()
        
        return jsonify({
            'success': True,
            'message': 'All models trained successfully!',
            'heart_features': heart_features,
            'diabetes_features': diabetes_features,
            'breast_cancer_features': breast_cancer_features,
            'liver_features': liver_features,
            'kidney_features': kidney_features
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    if not os.path.exists('data/heart_disease.csv'):
        preprocessor.create_sample_datasets()
    
    app.run(debug=True, host='0.0.0.0', port=5000)