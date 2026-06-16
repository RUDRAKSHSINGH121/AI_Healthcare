import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

class DataPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.label_encoders = {}
        
    def preprocess_heart_disease_data(self, df):
        """Preprocess heart disease dataset"""
        # Handle missing values
        df = df.fillna(df.median())
        
        # Feature engineering
        df['age_group'] = pd.cut(df['age'], bins=[0, 40, 60, 100], labels=['young', 'middle', 'old'])
        df['bp_category'] = pd.cut(df['trestbps'], bins=[0, 120, 140, 300], labels=['normal', 'high', 'very_high'])
        
        # Encode categorical variables
        categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'age_group', 'bp_category']
        
        for col in categorical_cols:
            if col in df.columns:
                if f'heart_{col}' in self.label_encoders:
                    le = self.label_encoders[f'heart_{col}']
                    df[col] = le.transform(df[col].astype(str))
                else:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[f'heart_{col}'] = le
        
        # Scale numerical features
        numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        if 'heart' in self.scalers:
            scaler = self.scalers['heart']
            df[numerical_cols] = scaler.transform(df[numerical_cols])
        else:
            scaler = StandardScaler()
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
            self.scalers['heart'] = scaler
        
        return df
    
    def preprocess_diabetes_data(self, df):
        """Preprocess diabetes dataset"""
        # Handle missing values
        df = df.fillna(df.median())
        
        # Feature engineering
        df['bmi_category'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, 100], labels=['underweight', 'normal', 'overweight', 'obese'])
        df['age_group'] = pd.cut(df['Age'], bins=[0, 30, 50, 70, 100], labels=['young', 'middle', 'senior', 'elderly'])
        
        # Encode categorical variables
        categorical_cols = ['bmi_category', 'age_group']
        
        for col in categorical_cols:
            if col in df.columns:
                if f'diabetes_{col}' in self.label_encoders:
                    le = self.label_encoders[f'diabetes_{col}']
                    df[col] = le.transform(df[col].astype(str))
                else:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[f'diabetes_{col}'] = le
        
        # Scale numerical features
        numerical_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        if 'diabetes' in self.scalers:
            scaler = self.scalers['diabetes']
            df[numerical_cols] = scaler.transform(df[numerical_cols])
        else:
            scaler = StandardScaler()
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
            self.scalers['diabetes'] = scaler
        
        return df
    
    def create_sample_datasets(self):
        """Create sample datasets for demonstration"""
        # Heart Disease Dataset
        np.random.seed(42)
        n_samples = 1000
        
        heart_data = {
            'age': np.random.normal(54, 9, n_samples),
            'sex': np.random.choice([0, 1], n_samples),
            'cp': np.random.choice([0, 1, 2, 3], n_samples),
            'trestbps': np.random.normal(131, 18, n_samples),
            'chol': np.random.normal(247, 52, n_samples),
            'fbs': np.random.choice([0, 1], n_samples),
            'restecg': np.random.choice([0, 1, 2], n_samples),
            'thalach': np.random.normal(150, 23, n_samples),
            'exang': np.random.choice([0, 1], n_samples),
            'oldpeak': np.random.exponential(1, n_samples),
            'slope': np.random.choice([0, 1, 2], n_samples),
            'ca': np.random.choice([0, 1, 2, 3], n_samples),
            'thal': np.random.choice([0, 1, 2], n_samples),
            'target': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
        }
        
        heart_df = pd.DataFrame(heart_data)
        heart_df.to_csv('data/heart_disease.csv', index=False)
        
        # Diabetes Dataset
        diabetes_data = {
            'Pregnancies': np.random.poisson(3, n_samples),
            'Glucose': np.random.normal(121, 32, n_samples),
            'BloodPressure': np.random.normal(69, 19, n_samples),
            'SkinThickness': np.random.normal(20, 16, n_samples),
            'Insulin': np.random.exponential(79, n_samples),
            'BMI': np.random.normal(32, 8, n_samples),
            'DiabetesPedigreeFunction': np.random.exponential(0.5, n_samples),
            'Age': np.random.normal(33, 12, n_samples),
            'Outcome': np.random.choice([0, 1], n_samples, p=[0.65, 0.35])
        }
        
        diabetes_df = pd.DataFrame(diabetes_data)
        diabetes_df.to_csv('data/diabetes.csv', index=False)
        
        # Breast Cancer Dataset
        breast_cancer_data = {
            'radius_mean': np.random.normal(14, 3, n_samples),
            'texture_mean': np.random.normal(19, 4, n_samples),
            'perimeter_mean': np.random.normal(92, 25, n_samples),
            'area_mean': np.random.normal(655, 350, n_samples),
            'smoothness_mean': np.random.normal(0.096, 0.014, n_samples),
            'compactness_mean': np.random.normal(0.104, 0.053, n_samples),
            'concavity_mean': np.random.normal(0.089, 0.080, n_samples),
            'concave_points_mean': np.random.normal(0.049, 0.039, n_samples),
            'symmetry_mean': np.random.normal(0.181, 0.027, n_samples),
            'fractal_dimension_mean': np.random.normal(0.063, 0.007, n_samples),
            'diagnosis': np.random.choice([0, 1], n_samples, p=[0.63, 0.37])  # 0=Benign, 1=Malignant
        }
        
        breast_cancer_df = pd.DataFrame(breast_cancer_data)
        breast_cancer_df.to_csv('data/breast_cancer.csv', index=False)
        
        # Liver Disease Dataset
        liver_data = {
            'Age': np.random.normal(45, 16, n_samples),
            'Gender': np.random.choice([0, 1], n_samples),  # 0=Female, 1=Male
            'Total_Bilirubin': np.random.exponential(1.2, n_samples),
            'Direct_Bilirubin': np.random.exponential(0.4, n_samples),
            'Alkaline_Phosphotase': np.random.normal(290, 138, n_samples),
            'Alamine_Aminotransferase': np.random.normal(55, 40, n_samples),
            'Aspartate_Aminotransferase': np.random.normal(60, 45, n_samples),
            'Total_Proteins': np.random.normal(6.8, 0.8, n_samples),
            'Albumin': np.random.normal(3.5, 0.5, n_samples),
            'Albumin_and_Globulin_Ratio': np.random.normal(1.0, 0.2, n_samples),
            'Dataset': np.random.choice([0, 1], n_samples, p=[0.58, 0.42])  # 0=No Disease, 1=Disease
        }
        
        liver_df = pd.DataFrame(liver_data)
        liver_df.to_csv('data/liver_disease.csv', index=False)
        
        # Kidney Disease Dataset
        kidney_data = {
            'age': np.random.normal(50, 17, n_samples),
            'bp': np.random.normal(76, 13, n_samples),
            'sg': np.random.normal(1.02, 0.01, n_samples),
            'al': np.random.choice([0, 1, 2, 3, 4, 5], n_samples),
            'su': np.random.choice([0, 1, 2, 3, 4, 5], n_samples),
            'rbc': np.random.choice([0, 1], n_samples),
            'pc': np.random.choice([0, 1], n_samples),
            'pcc': np.random.choice([0, 1], n_samples),
            'ba': np.random.choice([0, 1], n_samples),
            'bgr': np.random.normal(148, 57, n_samples),
            'bu': np.random.normal(57, 19, n_samples),
            'sc': np.random.normal(3.07, 1.2, n_samples),
            'sod': np.random.normal(138, 4, n_samples),
            'pot': np.random.normal(4.6, 0.6, n_samples),
            'hemo': np.random.normal(12.5, 2.5, n_samples),
            'pcv': np.random.normal(38, 7, n_samples),
            'wbcc': np.random.normal(8400, 3000, n_samples),
            'rbcc': np.random.normal(4.7, 0.6, n_samples),
            'htn': np.random.choice([0, 1], n_samples),
            'dm': np.random.choice([0, 1], n_samples),
            'cad': np.random.choice([0, 1], n_samples),
            'appet': np.random.choice([0, 1], n_samples),
            'pe': np.random.choice([0, 1], n_samples),
            'ane': np.random.choice([0, 1], n_samples),
            'classification': np.random.choice([0, 1], n_samples, p=[0.38, 0.62])  # 0=CKD, 1=Not CKD
        }
        
        kidney_df = pd.DataFrame(kidney_data)
        kidney_df.to_csv('data/kidney_disease.csv', index=False)
        
        return heart_df, diabetes_df, breast_cancer_df, liver_df, kidney_df
    
    def preprocess_breast_cancer_data(self, df):
        """Preprocess breast cancer dataset"""
        # Handle missing values
        df = df.fillna(df.median())
        
        # Scale all features (all are numerical)
        numerical_cols = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 
                         'smoothness_mean', 'compactness_mean', 'concavity_mean', 
                         'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean']
        
        if 'breast_cancer' in self.scalers:
            scaler = self.scalers['breast_cancer']
            df[numerical_cols] = scaler.transform(df[numerical_cols])
        else:
            scaler = StandardScaler()
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
            self.scalers['breast_cancer'] = scaler
        
        return df
    
    def preprocess_liver_disease_data(self, df):
        """Preprocess liver disease dataset"""
        # Handle missing values
        df = df.fillna(df.median())
        
        # Scale numerical features
        numerical_cols = ['Age', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase',
                         'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 
                         'Total_Proteins', 'Albumin', 'Albumin_and_Globulin_Ratio']
        
        if 'liver' in self.scalers:
            scaler = self.scalers['liver']
            df[numerical_cols] = scaler.transform(df[numerical_cols])
        else:
            scaler = StandardScaler()
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
            self.scalers['liver'] = scaler
        
        return df
    
    def preprocess_kidney_disease_data(self, df):
        """Preprocess kidney disease dataset"""
        # Handle missing values
        df = df.fillna(df.median())
        
        # Scale numerical features
        numerical_cols = ['age', 'bp', 'sg', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc']
        
        if 'kidney' in self.scalers:
            scaler = self.scalers['kidney']
            df[numerical_cols] = scaler.transform(df[numerical_cols])
        else:
            scaler = StandardScaler()
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
            self.scalers['kidney'] = scaler
        
        return df
    
    def save_preprocessors(self):
        """Save scalers and encoders"""
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.scalers, 'models/scalers.pkl')
        joblib.dump(self.label_encoders, 'models/label_encoders.pkl')
    
    def load_preprocessors(self):
        """Load scalers and encoders"""
        self.scalers = joblib.load('models/scalers.pkl')
        self.label_encoders = joblib.load('models/label_encoders.pkl')

if __name__ == "__main__":
    HERE = os.path.dirname(os.path.abspath(__file__))
    if HERE:
        os.chdir(HERE)
    preprocessor = DataPreprocessor()
    heart_df, diabetes_df = preprocessor.create_sample_datasets()
    
    # Preprocess datasets
    processed_heart = preprocessor.preprocess_heart_disease_data(heart_df.copy())
    processed_diabetes = preprocessor.preprocess_diabetes_data(diabetes_df.copy())
    
    # Save preprocessors
    preprocessor.save_preprocessors()
    
    print("Sample datasets created and preprocessed successfully!")
    print(f"Heart disease dataset shape: {processed_heart.shape}")
    print(f"Diabetes dataset shape: {processed_diabetes.shape}")
