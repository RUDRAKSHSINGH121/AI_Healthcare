from pathlib import Path

class Config:
    # Model paths
    HEART_DISEASE_MODEL_PATH = Path('models/heart_disease_model.pkl')
    DIABETES_MODEL_PATH = Path('models/diabetes_model.pkl')
    BREAST_CANCER_MODEL_PATH = Path('models/breast_cancer_model.pkl')
    LIVER_DISEASE_MODEL_PATH = Path('models/liver_disease_model.pkl')
    KIDNEY_DISEASE_MODEL_PATH = Path('models/kidney_disease_model.pkl')

    # Data paths
    HEART_DISEASE_DATA_PATH = Path('data/heart_disease.csv')
    DIABETES_DATA_PATH = Path('data/diabetes.csv')
    BREAST_CANCER_DATA_PATH = Path('data/breast_cancer.csv')
    LIVER_DISEASE_DATA_PATH = Path('data/liver_disease.csv')
    KIDNEY_DISEASE_DATA_PATH = Path('data/kidney_disease.csv')

    # Other constants
    SECRET_KEY = 'your-secret-key-here'
    DEBUG = True
    HOST = '0.0.0.0'
    PORT = 5000