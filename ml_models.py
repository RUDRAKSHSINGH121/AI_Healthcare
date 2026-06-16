import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib
import os
from data_preprocessing import DataPreprocessor

class MLModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_models = {}
        self.preprocessor = DataPreprocessor()
        
    def train_heart_disease_model(self):
        """Train heart disease prediction model"""
        # Load and preprocess data
        heart_df = pd.read_csv('data/heart_disease.csv')
        processed_df = self.preprocessor.preprocess_heart_disease_data(heart_df.copy())
        
        # Prepare features and target
        X = processed_df.drop('target', axis=1)
        y = processed_df['target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Define models
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        # Train and evaluate models
        best_score = 0
        best_model_name = None
        
        for name, model in models.items():
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            auc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            
            print(f"\n{name} Model:")
            print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"AUC Score: {auc_score:.4f}")
            
            # Store model
            self.models[f'heart_{name}'] = model
            
            # Track best model
            if accuracy > best_score:
                best_score = accuracy
                best_model_name = name
                self.best_models['heart'] = model
        
        # Hyperparameter tuning for best model
        if best_model_name == 'RandomForest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
            grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy')
            grid_search.fit(X_train, y_train)
            self.best_models['heart'] = grid_search.best_estimator_
            print(f"\nBest parameters for RandomForest: {grid_search.best_params_}")
        
        # Final evaluation
        final_model = self.best_models['heart']
        y_pred_final = final_model.predict(X_test)
        final_accuracy = accuracy_score(y_test, y_pred_final)
        
        print(f"\nFinal Heart Disease Model Accuracy: {final_accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_final))
        
        # Save model
        os.makedirs('models', exist_ok=True)
        joblib.dump(final_model, 'models/heart_disease_model.pkl')
        
        return final_model, X.columns.tolist()
    
    def train_diabetes_model(self):
        """Train diabetes prediction model"""
        # Load and preprocess data
        diabetes_df = pd.read_csv('data/diabetes.csv')
        processed_df = self.preprocessor.preprocess_diabetes_data(diabetes_df.copy())
        
        # Prepare features and target
        X = processed_df.drop('Outcome', axis=1)
        y = processed_df['Outcome']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Define models
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        # Train and evaluate models
        best_score = 0
        best_model_name = None
        
        for name, model in models.items():
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            auc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            
            print(f"\n{name} Model:")
            print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"AUC Score: {auc_score:.4f}")
            
            # Store model
            self.models[f'diabetes_{name}'] = model
            
            # Track best model
            if accuracy > best_score:
                best_score = accuracy
                best_model_name = name
                self.best_models['diabetes'] = model
        
        # Hyperparameter tuning for best model
        if best_model_name == 'RandomForest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
            grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy')
            grid_search.fit(X_train, y_train)
            self.best_models['diabetes'] = grid_search.best_estimator_
            print(f"\nBest parameters for RandomForest: {grid_search.best_params_}")
        
        # Final evaluation
        final_model = self.best_models['diabetes']
        y_pred_final = final_model.predict(X_test)
        final_accuracy = accuracy_score(y_test, y_pred_final)
        
        print(f"\nFinal Diabetes Model Accuracy: {final_accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_final))
        
        # Save model
        os.makedirs('models', exist_ok=True)
        joblib.dump(final_model, 'models/diabetes_model.pkl')
        
        return final_model, X.columns.tolist()
    
    def train_breast_cancer_model(self):
        """Train breast cancer prediction model"""
        # Load and preprocess data
        breast_cancer_df = pd.read_csv('data/breast_cancer.csv')
        processed_df = self.preprocessor.preprocess_breast_cancer_data(breast_cancer_df.copy())
        
        # Prepare features and target
        X = processed_df.drop('diagnosis', axis=1)
        y = processed_df['diagnosis']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Define models
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        # Train and evaluate models
        best_score = 0
        best_model_name = None
        
        for name, model in models.items():
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            auc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            
            print(f"\n{name} Model:")
            print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"AUC Score: {auc_score:.4f}")
            
            # Store model
            self.models[f'breast_cancer_{name}'] = model
            
            # Track best model
            if accuracy > best_score:
                best_score = accuracy
                best_model_name = name
                self.best_models['breast_cancer'] = model
        
        # Final evaluation
        final_model = self.best_models['breast_cancer']
        y_pred_final = final_model.predict(X_test)
        final_accuracy = accuracy_score(y_test, y_pred_final)
        
        print(f"\nFinal Breast Cancer Model Accuracy: {final_accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_final))
        
        # Save model
        os.makedirs('models', exist_ok=True)
        joblib.dump(final_model, 'models/breast_cancer_model.pkl')
        
        return final_model, X.columns.tolist()
    
    def train_liver_disease_model(self):
        """Train liver disease prediction model"""
        # Load and preprocess data
        liver_df = pd.read_csv('data/liver_disease.csv')
        processed_df = self.preprocessor.preprocess_liver_disease_data(liver_df.copy())
        
        # Prepare features and target
        X = processed_df.drop('Dataset', axis=1)
        y = processed_df['Dataset']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Define models
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        # Train and evaluate models
        best_score = 0
        best_model_name = None
        
        for name, model in models.items():
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            auc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            
            print(f"\n{name} Model:")
            print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"AUC Score: {auc_score:.4f}")
            
            # Store model
            self.models[f'liver_{name}'] = model
            
            # Track best model
            if accuracy > best_score:
                best_score = accuracy
                best_model_name = name
                self.best_models['liver'] = model
        
        # Final evaluation
        final_model = self.best_models['liver']
        y_pred_final = final_model.predict(X_test)
        final_accuracy = accuracy_score(y_test, y_pred_final)
        
        print(f"\nFinal Liver Disease Model Accuracy: {final_accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_final))
        
        # Save model
        os.makedirs('models', exist_ok=True)
        joblib.dump(final_model, 'models/liver_disease_model.pkl')
        
        return final_model, X.columns.tolist()
    
    def train_kidney_disease_model(self):
        """Train kidney disease prediction model"""
        # Load and preprocess data
        kidney_df = pd.read_csv('data/kidney_disease.csv')
        processed_df = self.preprocessor.preprocess_kidney_disease_data(kidney_df.copy())
        
        # Prepare features and target
        X = processed_df.drop('classification', axis=1)
        y = processed_df['classification']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Define models
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        # Train and evaluate models
        best_score = 0
        best_model_name = None
        
        for name, model in models.items():
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            auc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            
            print(f"\n{name} Model:")
            print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"AUC Score: {auc_score:.4f}")
            
            # Store model
            self.models[f'kidney_{name}'] = model
            
            # Track best model
            if accuracy > best_score:
                best_score = accuracy
                best_model_name = name
                self.best_models['kidney'] = model
        
        # Final evaluation
        final_model = self.best_models['kidney']
        y_pred_final = final_model.predict(X_test)
        final_accuracy = accuracy_score(y_test, y_pred_final)
        
        print(f"\nFinal Kidney Disease Model Accuracy: {final_accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_final))
        
        # Save model
        os.makedirs('models', exist_ok=True)
        joblib.dump(final_model, 'models/kidney_disease_model.pkl')
        
        return final_model, X.columns.tolist()
    
    def predict_heart_disease(self, features):
        """Predict heart disease for given features"""
        # Load model and preprocessors
        model = joblib.load('models/heart_disease_model.pkl')
        
        # Load preprocessors
        try:
            scalers = joblib.load('models/scalers.pkl')
            label_encoders = joblib.load('models/label_encoders.pkl')
            self.preprocessor.scalers = scalers
            self.preprocessor.label_encoders = label_encoders
        except:
            pass  # Use default preprocessors if files don't exist
        
        # Create DataFrame with the input features
        feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        df = pd.DataFrame([features], columns=feature_names)
        
        # Apply the same preprocessing as training
        processed_df = self.preprocessor.preprocess_heart_disease_data(df)
        
        # Make prediction
        prediction = model.predict(processed_df)
        probability = model.predict_proba(processed_df)
        
        return {
            'prediction': int(prediction[0]),
            'probability': float(probability[0][1]),
            'risk_level': 'High' if probability[0][1] > 0.7 else 'Medium' if probability[0][1] > 0.4 else 'Low'
        }
    
    def predict_diabetes(self, features):
        """Predict diabetes for given features"""
        # Load model and preprocessors
        model = joblib.load('models/diabetes_model.pkl')
        
        # Load preprocessors
        try:
            scalers = joblib.load('models/scalers.pkl')
            label_encoders = joblib.load('models/label_encoders.pkl')
            self.preprocessor.scalers = scalers
            self.preprocessor.label_encoders = label_encoders
        except:
            pass  # Use default preprocessors if files don't exist
        
        # Create DataFrame with the input features
        feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        df = pd.DataFrame([features], columns=feature_names)
        
        # Apply the same preprocessing as training
        processed_df = self.preprocessor.preprocess_diabetes_data(df)
        
        # Make prediction
        prediction = model.predict(processed_df)
        probability = model.predict_proba(processed_df)
        
        return {
            'prediction': int(prediction[0]),
            'probability': float(probability[0][1]),
            'risk_level': 'High' if probability[0][1] > 0.7 else 'Medium' if probability[0][1] > 0.4 else 'Low'
        }
    
    def predict_breast_cancer(self, features):
        """Predict breast cancer for given features"""
        # Load model and preprocessors
        model = joblib.load('models/breast_cancer_model.pkl')
        
        # Load preprocessors
        try:
            scalers = joblib.load('models/scalers.pkl')
            label_encoders = joblib.load('models/label_encoders.pkl')
            self.preprocessor.scalers = scalers
            self.preprocessor.label_encoders = label_encoders
        except:
            pass  # Use default preprocessors if files don't exist
        
        # Create DataFrame with the input features
        feature_names = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 
                        'smoothness_mean', 'compactness_mean', 'concavity_mean', 
                        'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean']
        df = pd.DataFrame([features], columns=feature_names)
        
        # Apply the same preprocessing as training
        processed_df = self.preprocessor.preprocess_breast_cancer_data(df)
        
        # Make prediction
        prediction = model.predict(processed_df)
        probability = model.predict_proba(processed_df)
        
        return {
            'prediction': int(prediction[0]),
            'probability': float(probability[0][1]),
            'risk_level': 'High' if probability[0][1] > 0.7 else 'Medium' if probability[0][1] > 0.4 else 'Low'
        }
    
    def predict_liver_disease(self, features):
        """Predict liver disease for given features"""
        # Load model and preprocessors
        model = joblib.load('models/liver_disease_model.pkl')
        
        # Load preprocessors
        try:
            scalers = joblib.load('models/scalers.pkl')
            label_encoders = joblib.load('models/label_encoders.pkl')
            self.preprocessor.scalers = scalers
            self.preprocessor.label_encoders = label_encoders
        except:
            pass  # Use default preprocessors if files don't exist
        
        # Create DataFrame with the input features
        feature_names = ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase',
                        'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 
                        'Total_Proteins', 'Albumin', 'Albumin_and_Globulin_Ratio']
        df = pd.DataFrame([features], columns=feature_names)
        
        # Apply the same preprocessing as training
        processed_df = self.preprocessor.preprocess_liver_disease_data(df)
        
        # Make prediction
        prediction = model.predict(processed_df)
        probability = model.predict_proba(processed_df)
        
        return {
            'prediction': int(prediction[0]),
            'probability': float(probability[0][1]),
            'risk_level': 'High' if probability[0][1] > 0.7 else 'Medium' if probability[0][1] > 0.4 else 'Low'
        }
    
    def predict_kidney_disease(self, features):
        """Predict kidney disease for given features"""
        # Load model and preprocessors
        model = joblib.load('models/kidney_disease_model.pkl')
        
        # Load preprocessors
        try:
            scalers = joblib.load('models/scalers.pkl')
            label_encoders = joblib.load('models/label_encoders.pkl')
            self.preprocessor.scalers = scalers
            self.preprocessor.label_encoders = label_encoders
        except:
            pass  # Use default preprocessors if files don't exist
        
        # Create DataFrame with the input features
        feature_names = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 
                        'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
        df = pd.DataFrame([features], columns=feature_names)
        
        # Apply the same preprocessing as training
        processed_df = self.preprocessor.preprocess_kidney_disease_data(df)
        
        # Make prediction
        prediction = model.predict(processed_df)
        probability = model.predict_proba(processed_df)
        
        return {
            'prediction': int(prediction[0]),
            'probability': float(probability[0][1]),
            'risk_level': 'High' if probability[0][1] > 0.7 else 'Medium' if probability[0][1] > 0.4 else 'Low'
        }

if __name__ == "__main__":
    HERE = os.path.dirname(os.path.abspath(__file__))
    if HERE:
        os.chdir(HERE)
    trainer = MLModelTrainer()
    
    print("Training Heart Disease Model...")
    heart_model, heart_features = trainer.train_heart_disease_model()
    
    print("\nTraining Diabetes Model...")
    diabetes_model, diabetes_features = trainer.train_diabetes_model()
    
    print("\nModels trained and saved successfully!")
    print(f"Heart disease features: {heart_features}")
    print(f"Diabetes features: {diabetes_features}")
