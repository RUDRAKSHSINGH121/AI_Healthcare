#!/usr/bin/env python3
"""
AI Healthcare Diagnostics - Model Training Script
This script initializes and trains the machine learning models for the healthcare diagnostics system.
"""

import os
import sys
from data_preprocessing import DataPreprocessor
from ml_models import MLModelTrainer

def main():
    """Main function to train all models"""
    # Change current working directory to the directory containing this script
    HERE = os.path.dirname(os.path.abspath(__file__))
    if HERE:
        os.chdir(HERE)
        
    print("🏥 AI Healthcare Diagnostics - Model Training")
    print("=" * 50)
    
    try:
        # Create necessary directories
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
        print("📊 Initializing data preprocessing...")
        preprocessor = DataPreprocessor()
        
        # Create sample datasets if they don't exist
        if not os.path.exists('data/heart_disease.csv') or not os.path.exists('data/diabetes.csv'):
            print("📈 Creating sample datasets...")
            heart_df, diabetes_df = preprocessor.create_sample_datasets()
            print(f"✅ Created heart disease dataset: {heart_df.shape}")
            print(f"✅ Created diabetes dataset: {diabetes_df.shape}")
        else:
            print("✅ Sample datasets already exist")
        
        print("\n🤖 Initializing model trainer...")
        trainer = MLModelTrainer()
        
        print("\n❤️ Training Heart Disease Model...")
        print("-" * 30)
        heart_model, heart_features = trainer.train_heart_disease_model()
        print(f"✅ Heart disease model trained successfully!")
        print(f"📋 Features used: {len(heart_features)}")
        
        print("\n🩸 Training Diabetes Model...")
        print("-" * 30)
        diabetes_model, diabetes_features = trainer.train_diabetes_model()
        print(f"✅ Diabetes model trained successfully!")
        print(f"📋 Features used: {len(diabetes_features)}")
        
        print("\n🎗️ Training Breast Cancer Model...")
        print("-" * 30)
        breast_cancer_model, breast_cancer_features = trainer.train_breast_cancer_model()
        print(f"✅ Breast cancer model trained successfully!")
        print(f"📋 Features used: {len(breast_cancer_features)}")
        
        print("\n🫀 Training Liver Disease Model...")
        print("-" * 30)
        liver_model, liver_features = trainer.train_liver_disease_model()
        print(f"✅ Liver disease model trained successfully!")
        print(f"📋 Features used: {len(liver_features)}")
        
        print("\n🫘 Training Kidney Disease Model...")
        print("-" * 30)
        kidney_model, kidney_features = trainer.train_kidney_disease_model()
        print(f"✅ Kidney disease model trained successfully!")
        print(f"📋 Features used: {len(kidney_features)}")
        
        print("\n🎉 All models trained successfully!")
        print("=" * 50)
        print("🚀 You can now run the Flask application:")
        print("   python app.py")
        print("🌐 Then visit: http://localhost:5000")
        print("\n📊 Available Disease Predictions:")
        print("   • Heart Disease (13 parameters)")
        print("   • Diabetes (8 parameters)")
        print("   • Breast Cancer (10 parameters)")
        print("   • Liver Disease (10 parameters)")
        print("   • Kidney Disease (24 parameters)")
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        print("🔧 Please check the error message and try again")
        sys.exit(1)

if __name__ == "__main__":
    main()
