import sys
import os
import json

# Ensure project root is on sys.path so `src` package imports work when running this script directly
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.app import app

payloads = {
    '/predict/heart': {
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
    },
    '/predict/diabetes': {
        'pregnancies': 6,
        'glucose': 148,
        'blood_pressure': 72,
        'skin_thickness': 35,
        'insulin': 0,
        'bmi': 33.6,
        'diabetes_pedigree': 0.627,
        'age': 50
    },
    '/predict/breast-cancer': {
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
    },
    '/predict/liver-disease': {
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
    },
    '/predict/kidney-disease': {
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
    }
}

with app.test_client() as client:
    for path, payload in payloads.items():
        resp = client.post(path, json=payload)
        print('\n===', path, 'status=', resp.status_code)
        try:
            print(json.dumps(resp.get_json(), indent=2))
        except Exception:
            print(resp.get_data(as_text=True))

# Also print models status
with app.test_client() as client:
    resp = client.get('/api/models/status')
    print('\n=== /api/models/status', resp.status_code)
    print(json.dumps(resp.get_json(), indent=2))
