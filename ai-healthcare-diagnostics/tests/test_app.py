from flask import json
import pytest
from src.app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'Welcome to AI Healthcare Diagnostics' in response.data

def test_heart_disease_page(client):
    response = client.get('/heart-disease')
    assert response.status_code == 200
    assert b'Heart Disease Prediction' in response.data

def test_diabetes_page(client):
    response = client.get('/diabetes')
    assert response.status_code == 200
    assert b'Diabetes Prediction' in response.data

def test_breast_cancer_page(client):
    response = client.get('/breast-cancer')
    assert response.status_code == 200
    assert b'Breast Cancer Prediction' in response.data

def test_liver_disease_page(client):
    response = client.get('/liver-disease')
    assert response.status_code == 200
    assert b'Liver Disease Prediction' in response.data

def test_kidney_disease_page(client):
    response = client.get('/kidney-disease')
    assert response.status_code == 200
    assert b'Kidney Disease Prediction' in response.data

def test_about_page(client):
    response = client.get('/about')
    assert response.status_code == 200
    assert b'About Us' in response.data

def test_predict_heart_disease(client):
    response = client.post('/predict/heart', json={
        'age': 63,
        'sex': 1,
        'cp': 3,
        'trestbps': 145,
        'chol': 233,
        'fbs': 1,
        'restecg': 0,
        'thalach': 150,
        'exang': 0,
        'oldpeak': 2.3,
        'slope': 0,
        'ca': 0,
        'thal': 1
    })
    json_data = json.loads(response.data)
    assert json_data['success'] is True

def test_predict_diabetes(client):
    response = client.post('/predict/diabetes', json={
        'pregnancies': 6,
        'glucose': 148,
        'blood_pressure': 72,
        'skin_thickness': 35,
        'insulin': 0,
        'bmi': 33.6,
        'diabetes_pedigree': 0.627,
        'age': 50
    })
    json_data = json.loads(response.data)
    assert json_data['success'] is True

def test_predict_breast_cancer(client):
    response = client.post('/predict/breast-cancer', json={
        'radius_mean': 17.99,
        'texture_mean': 10.38,
        'perimeter_mean': 122.8,
        'area_mean': 1001.0,
        'smoothness_mean': 0.118,
        'compactness_mean': 0.277,
        'concavity_mean': 0.300,
        'concave_points_mean': 0.147,
        'symmetry_mean': 0.241,
        'fractal_dimension_mean': 0.078
    })
    json_data = json.loads(response.data)
    assert json_data['success'] is True

def test_predict_liver_disease(client):
    response = client.post('/predict/liver-disease', json={
        'age': 62,
        'gender': 1,
        'total_bilirubin': 0.7,
        'direct_bilirubin': 0.1,
        'alkaline_phosphotase': 70,
        'alamine_aminotransferase': 20,
        'aspartate_aminotransferase': 20,
        'total_proteins': 7.0,
        'albumin': 4.0,
        'albumin_and_globulin_ratio': 0.9
    })
    json_data = json.loads(response.data)
    assert json_data['success'] is True

def test_predict_kidney_disease(client):
    response = client.post('/predict/kidney-disease', json={
        'age': 50,
        'bp': 80,
        'sg': 1.2,
        'al': 1,
        'su': 0,
        'rbc': 1,
        'pc': 0,
        'pcc': 0,
        'ba': 0,
        'bgr': 100,
        'bu': 20,
        'sc': 1.0,
        'sod': 135,
        'pot': 4.5,
        'hemo': 15.0,
        'pcv': 45.0,
        'wbcc': 6000,
        'rbcc': 5.0,
        'htn': 0,
        'dm': 0,
        'cad': 0,
        'appet': 1,
        'pe': 0,
        'ane': 0
    })
    json_data = json.loads(response.data)
    assert json_data['success'] is True

def test_models_status(client):
    response = client.get('/api/models/status')
    json_data = json.loads(response.data)
    assert json_data['heart_model'] is True
    assert json_data['diabetes_model'] is True
    assert json_data['breast_cancer_model'] is True
    assert json_data['liver_model'] is True
    assert json_data['kidney_model'] is True

def test_train_models(client):
    response = client.post('/train/models')
    json_data = json.loads(response.data)
    assert json_data['success'] is True
    assert 'All models trained successfully!' in json_data['message']