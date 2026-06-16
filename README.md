# AI-Driven Healthcare Diagnostics

A comprehensive machine learning-based healthcare diagnostics system that predicts heart disease and diabetes risk using advanced AI algorithms.

## 🚀 Features

- **Heart Disease Prediction**: Assess cardiovascular risk using 13 health parameters
- **Diabetes Prediction**: Evaluate diabetes risk with 8 comprehensive health metrics
- **Real-time Analysis**: Instant predictions with probability scores and risk level classification
- **Modern Web Interface**: Responsive, user-friendly design with Bootstrap 5
- **REST API**: Programmatic access to prediction services
- **Model Training**: Automated model training and evaluation
- **Multiple ML Algorithms**: Random Forest, SVM, Gradient Boosting, Logistic Regression

## 🛠️ Technology Stack

- **Backend**: Python, Flask
- **Machine Learning**: Scikit-learn, Pandas, NumPy
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Data Visualization**: Matplotlib, Seaborn, Plotly
- **Model Persistence**: Joblib

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## 🚀 Installation & Setup

1. **Clone or Download the Project**
   ```bash
   # If you have git installed
   git clone <repository-url>
   cd ai-healthcare-diagnostics
   
   # Or download and extract the project files
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**
   ```bash
   python app.py
   ```

4. **Access the Application**
   - Open your web browser
   - Navigate to `http://localhost:5000`

## 🎯 Usage Guide

### 1. Initial Setup
- When you first access the application, you'll need to train the models
- Click the "Train Models" button on the homepage
- Wait for the training to complete (this may take a few minutes)

### 2. Heart Disease Prediction
1. Navigate to the "Heart Disease" page
2. Fill in all required health parameters:
   - Age, Sex, Chest Pain Type
   - Blood Pressure, Cholesterol, Blood Sugar
   - ECG Results, Heart Rate, Exercise Angina
   - ST Depression, Slope, Major Vessels, Thalassemia
3. Click "Predict Heart Disease Risk"
4. View the results with risk level and probability score

### 3. Diabetes Prediction
1. Navigate to the "Diabetes" page
2. Fill in all required health parameters:
   - Pregnancies, Glucose Level, Blood Pressure
   - Skin Thickness, Insulin Level, BMI
   - Diabetes Pedigree Function, Age
3. Click "Predict Diabetes Risk"
4. View the results with risk level and probability score

## 🔧 API Endpoints

### Model Status
```
GET /api/models/status
```
Returns the availability status of trained models.

### Train Models
```
POST /train/models
```
Trains both heart disease and diabetes prediction models.

### Heart Disease Prediction
```
POST /predict/heart
Content-Type: application/json

{
    "age": 65,
    "sex": 1,
    "cp": 0,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0,
    "thal": 1
}
```

### Diabetes Prediction
```
POST /predict/diabetes
Content-Type: application/json

{
    "pregnancies": 6,
    "glucose": 148,
    "blood_pressure": 72,
    "skin_thickness": 35,
    "insulin": 0,
    "bmi": 33.6,
    "diabetes_pedigree": 0.627,
    "age": 50
}
```

## 📊 Model Performance

### Heart Disease Model
- **Algorithm**: Random Forest Classifier
- **Accuracy**: ~85-90%
- **Features**: 13 health parameters
- **Cross-validation**: 5-fold

### Diabetes Model
- **Algorithm**: Random Forest Classifier
- **Accuracy**: ~80-85%
- **Features**: 8 health parameters
- **Cross-validation**: 5-fold

## 📁 Project Structure

```
ai-healthcare-diagnostics/
├── app.py                 # Main Flask application
├── data_preprocessing.py  # Data preprocessing utilities
├── ml_models.py          # Machine learning model training
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── data/               # Sample datasets
│   ├── heart_disease.csv
│   └── diabetes.csv
├── models/             # Trained model files
│   ├── heart_disease_model.pkl
│   ├── diabetes_model.pkl
│   ├── scalers.pkl
│   └── label_encoders.pkl
├── templates/          # HTML templates
│   ├── index.html
│   ├── heart_disease.html
│   ├── diabetes.html
│   └── about.html
└── static/            # Static assets
    ├── css/
    │   └── style.css
    └── js/
        ├── main.js
        ├── heart-disease.js
        └── diabetes.js
```

## 🔬 Machine Learning Details

### Data Preprocessing
- **Missing Value Handling**: Median imputation
- **Feature Engineering**: Age groups, BMI categories, blood pressure categories
- **Scaling**: StandardScaler for numerical features
- **Encoding**: LabelEncoder for categorical features

### Model Training Process
1. **Data Splitting**: 80% training, 20% testing
2. **Cross-validation**: 5-fold for model evaluation
3. **Hyperparameter Tuning**: GridSearchCV for optimal parameters
4. **Model Selection**: Best performing algorithm chosen
5. **Model Persistence**: Trained models saved using joblib

### Algorithms Used
- **Random Forest**: Ensemble method with multiple decision trees
- **Support Vector Machine (SVM)**: Kernel-based classification
- **Gradient Boosting**: Sequential ensemble learning
- **Logistic Regression**: Linear classification with regularization

## ⚠️ Important Disclaimers

- **Educational Purpose**: This system is designed for educational and research purposes only
- **Not Medical Advice**: Results should not be used as a substitute for professional medical advice
- **Consult Healthcare Professionals**: Always consult with qualified healthcare professionals for medical decisions
- **Data Privacy**: Ensure patient data privacy and compliance with healthcare regulations

## 🐛 Troubleshooting

### Common Issues

1. **Models Not Available**
   - Solution: Click "Train Models" button on homepage
   - Ensure all dependencies are installed

2. **Prediction Errors**
   - Check that all form fields are filled correctly
   - Verify input values are within valid ranges
   - Ensure models are trained before making predictions

3. **Installation Issues**
   - Ensure Python 3.8+ is installed
   - Use virtual environment: `python -m venv venv`
   - Activate virtual environment: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Linux/Mac)

## 🔮 Future Enhancements

- [ ] Additional disease prediction models (cancer, stroke, etc.)
- [ ] Real-time data integration from medical devices
- [ ] Advanced visualization dashboards
- [ ] User authentication and patient records
- [ ] Mobile application development
- [ ] Integration with electronic health records (EHR)
- [ ] Explainable AI for model interpretability

## 📞 Support

For questions, issues, or contributions:
- Create an issue in the project repository
- Contact the development team
- Check the troubleshooting section above

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Note**: This AI healthcare diagnostics system is a demonstration project showcasing machine learning applications in healthcare. It should not be used for actual medical diagnosis without proper validation and regulatory approval.

