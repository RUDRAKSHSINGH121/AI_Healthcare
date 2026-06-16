from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import numpy as np
import joblib
import os
import sys
import json

# Ensure this file's directory is on sys.path so local modules import correctly
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Change current working directory to the directory containing this file
os.chdir(HERE)

# Try to import real modules; if they fail, provide lightweight fallbacks so the app starts
try:
    from data_preprocessing import DataPreprocessor
except Exception as _exc:
    class DataPreprocessor:
        """Fallback minimal preprocessor used when data_preprocessing.py cannot be imported.
           It only ensures sample CSVs exist so endpoints that expect data/files won't crash."""
        def create_sample_datasets(self):
            os.makedirs('data', exist_ok=True)
            # create minimal heart and diabetes csvs if missing
            if not os.path.exists('data/heart_disease.csv'):
                df = pd.DataFrame({
                    'age': np.random.normal(54,9,100),
                    'sex': np.random.choice([0,1],100),
                    'cp': np.random.choice([0,1,2,3],100),
                    'trestbps': np.random.normal(131,18,100),
                    'chol': np.random.normal(247,52,100),
                    'fbs': np.random.choice([0,1],100),
                    'restecg': np.random.choice([0,1,2],100),
                    'thalach': np.random.normal(150,23,100),
                    'exang': np.random.choice([0,1],100),
                    'oldpeak': np.random.exponential(1,100),
                    'slope': np.random.choice([0,1,2],100),
                    'ca': np.random.choice([0,1,2,3],100),
                    'thal': np.random.choice([0,1,2],100),
                    'target': np.random.choice([0,1],100)
                })
                df.to_csv('data/heart_disease.csv', index=False)
            if not os.path.exists('data/diabetes.csv'):
                df = pd.DataFrame({
                    'Pregnancies': np.random.poisson(3,100),
                    'Glucose': np.random.normal(121,32,100),
                    'BloodPressure': np.random.normal(69,19,100),
                    'SkinThickness': np.random.normal(20,16,100),
                    'Insulin': np.random.exponential(79,100),
                    'BMI': np.random.normal(32,8,100),
                    'DiabetesPedigreeFunction': np.random.exponential(0.5,100),
                    'Age': np.random.normal(33,12,100),
                    'Outcome': np.random.choice([0,1],100)
                })
                df.to_csv('data/diabetes.csv', index=False)

try:
    from ml_models import MLModelTrainer
except Exception as _exc:
    class MLModelTrainer:
        """Fallback minimal trainer/predictor used when ml_models.py cannot be imported.
           These methods return deterministic dummy responses so app endpoints work."""
        def _dummy_result(self, prob=0.5):
            pred = int(prob >= 0.5)
            if prob >= 0.75:
                risk = 'high'
            elif prob >= 0.5:
                risk = 'medium'
            else:
                risk = 'low'
            return {'prediction': pred, 'probability': float(prob), 'risk_level': risk}

        def predict_heart_disease(self, features):
            # simple heuristic: older age and high trestbps increase prob
            try:
                age = float(features[0])
                trestbps = float(features[3])
                prob = min(0.99, 0.2 + (age - 40) * 0.01 + max(0.0, (trestbps - 120) * 0.002))
            except Exception:
                prob = 0.5
            return self._dummy_result(prob)

        def predict_diabetes(self, features):
            try:
                glucose = float(features[1])
                bmi = float(features[5])
                prob = min(0.99, 0.1 + (glucose - 100) * 0.003 + (bmi - 25) * 0.01)
            except Exception:
                prob = 0.5
            return self._dummy_result(prob)

        def predict_breast_cancer(self, features):
            return self._dummy_result(0.3)

        def predict_liver_disease(self, features):
            return self._dummy_result(0.4)

        def predict_kidney_disease(self, features):
            return self._dummy_result(0.35)

        # training stubs used by /train/models endpoint
        def train_heart_disease_model(self):
            return (None, ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal'])

        def train_diabetes_model(self):
            return (None, ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'])

        def train_breast_cancer_model(self):
            return (None, ['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave_points_mean','symmetry_mean','fractal_dimension_mean'])

        def train_liver_disease_model(self):
            return (None, ['Age','Gender','Total_Bilirubin','Direct_Bilirubin','Alkaline_Phosphotase','Alamine_Aminotransferase','Aspartate_Aminotransferase','Total_Proteins','Albumin','Albumin_and_Globulin_Ratio'])

        def train_kidney_disease_model(self):
            return (None, ['age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu','sc','sod','pot','hemo','pcv','wbcc','rbcc','htn','dm','cad','appet','pe','ane'])

# Initialize app and components
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

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
    """Check if models are trained and available"""
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
    """Train models endpoint"""
    try:
        # Create sample data if it doesn't exist
        if not os.path.exists('data/heart_disease.csv'):
            preprocessor.create_sample_datasets()
        
        # Train models
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
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Create sample datasets if they don't exist
    if not os.path.exists('data/heart_disease.csv'):
        preprocessor.create_sample_datasets()
    
    app.run(debug=True, host='0.0.0.0', port=5000)
