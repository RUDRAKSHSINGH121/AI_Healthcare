from sklearn.model_selection import train_test_split
import pandas as pd
import os

class DataPreprocessor:
    def __init__(self):
        self.data_dir = 'data'
    
    def create_sample_datasets(self):
        # Sample dataset creation for heart disease
        heart_data = {
            'age': [29, 35, 45, 50, 60],
            'sex': [1, 0, 1, 1, 0],
            'cp': [0, 1, 2, 1, 0],
            'trestbps': [130, 140, 150, 160, 120],
            'chol': [200, 240, 300, 180, 220],
            'fbs': [0, 1, 0, 0, 1],
            'restecg': [0, 1, 0, 1, 0],
            'thalach': [150, 160, 170, 140, 130],
            'exang': [0, 1, 0, 1, 0],
            'oldpeak': [1.0, 2.0, 1.5, 0.5, 1.2],
            'slope': [1, 2, 1, 2, 1],
            'ca': [0, 1, 0, 1, 0],
            'thal': [1, 2, 3, 1, 2],
            'target': [0, 1, 1, 1, 0]
        }
        heart_df = pd.DataFrame(heart_data)
        heart_df.to_csv(os.path.join(self.data_dir, 'heart_disease.csv'), index=False)

        # Sample dataset creation for diabetes
        diabetes_data = {
            'pregnancies': [0, 1, 2, 3, 4],
            'glucose': [85, 90, 95, 100, 105],
            'blood_pressure': [70, 80, 75, 85, 90],
            'skin_thickness': [20, 25, 30, 35, 40],
            'insulin': [0, 100, 150, 200, 250],
            'bmi': [22.0, 25.0, 28.0, 30.0, 32.0],
            'diabetes_pedigree': [0.5, 0.6, 0.7, 0.8, 0.9],
            'age': [21, 25, 30, 35, 40],
            'outcome': [0, 1, 1, 1, 0]
        }
        diabetes_df = pd.DataFrame(diabetes_data)
        diabetes_df.to_csv(os.path.join(self.data_dir, 'diabetes.csv'), index=False)

        # Additional datasets can be created similarly for breast cancer, liver disease, and kidney disease.
        # Sample dataset for breast cancer
        breast_data = {
            'radius_mean': [17.99, 20.57, 19.69, 11.42, 20.29],
            'texture_mean': [10.38, 17.77, 21.25, 20.38, 14.34],
            'perimeter_mean': [122.8, 132.9, 130.0, 77.58, 135.1],
            'area_mean': [1001.0, 1326.0, 1203.0, 386.1, 1297.0],
            'smoothness_mean': [0.118, 0.084, 0.109, 0.142, 0.100],
            'compactness_mean': [0.277, 0.078, 0.159, 0.283, 0.132],
            'concavity_mean': [0.300, 0.086, 0.121, 0.241, 0.198],
            'concave_points_mean': [0.147, 0.070, 0.078, 0.105, 0.104],
            'symmetry_mean': [0.241, 0.181, 0.181, 0.259, 0.180],
            'fractal_dimension_mean': [0.078, 0.056, 0.059, 0.097, 0.058],
            'diagnosis': ['M', 'M', 'M', 'B', 'M']
        }
        breast_df = pd.DataFrame(breast_data)
        breast_df.to_csv(os.path.join(self.data_dir, 'breast_cancer.csv'), index=False)

        # Sample dataset for liver disease
        liver_data = {
            'age': [62, 25, 46, 50, 44],
            'gender': [1, 0, 1, 1, 0],
            'total_bilirubin': [0.7, 1.2, 0.5, 1.1, 0.9],
            'direct_bilirubin': [0.1, 0.2, 0.05, 0.15, 0.1],
            'alkaline_phosphotase': [70, 90, 85, 60, 80],
            'alamine_aminotransferase': [20, 30, 25, 22, 19],
            'aspartate_aminotransferase': [20, 25, 30, 18, 21],
            'total_proteins': [7.0, 6.5, 7.2, 6.8, 7.1],
            'albumin': [4.0, 3.5, 4.1, 3.8, 4.2],
            'albumin_and_globulin_ratio': [0.9, 1.0, 0.85, 0.95, 1.1],
            'Dataset': [1, 0, 1, 0, 1]
        }
        liver_df = pd.DataFrame(liver_data)
        liver_df.to_csv(os.path.join(self.data_dir, 'liver_disease.csv'), index=False)

        # Sample dataset for kidney disease
        kidney_data = {
            'age': [50, 40, 60, 30, 55],
            'bp': [80, 70, 90, 75, 85],
            'sg': [1.02, 1.01, 1.015, 1.02, 1.025],
            'al': [0, 1, 0, 0, 1],
            'su': [0, 0, 1, 0, 0],
            'rbc': [1, 0, 1, 1, 0],
            'pc': [0, 0, 0, 0, 0],
            'pcc': [0, 0, 0, 0, 0],
            'ba': [0, 0, 0, 0, 0],
            'bgr': [100, 110, 120, 90, 130],
            'bu': [20, 25, 30, 15, 22],
            'sc': [1.0, 1.1, 1.2, 0.9, 1.3],
            'sod': [135, 140, 130, 138, 136],
            'pot': [4.5, 4.0, 4.8, 3.8, 4.2],
            'hemo': [15.0, 14.5, 13.0, 16.0, 14.0],
            'pcv': [45.0, 44.0, 42.0, 46.0, 43.0],
            'wbcc': [6000, 7000, 8000, 5500, 6500],
            'rbcc': [5.0, 4.5, 4.8, 5.2, 4.9],
            'htn': [0, 0, 1, 0, 1],
            'dm': [0, 0, 1, 0, 0],
            'cad': [0, 0, 0, 0, 0],
            'appet': [1, 1, 0, 1, 1],
            'pe': [0, 0, 0, 0, 0],
            'ane': [0, 0, 0, 0, 0],
            'classification': ['notckd', 'notckd', 'ckd', 'notckd', 'ckd']
        }
        kidney_df = pd.DataFrame(kidney_data)
        kidney_df.to_csv(os.path.join(self.data_dir, 'kidney_disease.csv'), index=False)

    def load_data(self, filename):
        return pd.read_csv(os.path.join(self.data_dir, filename))

    def preprocess_data(self, df):
        # Implement preprocessing steps such as handling missing values, encoding categorical variables, etc.
        return df

    def split_data(self, df, target_column):
        X = df.drop(columns=[target_column])
        y = df[target_column]
        return train_test_split(X, y, test_size=0.2, random_state=42)