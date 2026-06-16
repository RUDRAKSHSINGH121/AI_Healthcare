# ai-healthcare-diagnostics

This project is an AI-powered healthcare diagnostics application that provides predictions for various health conditions, including heart disease, diabetes, breast cancer, liver disease, and kidney disease. The application is built using Flask and utilizes machine learning models for making predictions based on user input.

## Project Structure

```
ai-healthcare-diagnostics
├── src
│   ├── app.py                     # Main application file that sets up the Flask web server and defines routes
│   ├── data_preprocessing.py       # Contains the DataPreprocessor class for data preprocessing tasks
│   ├── ml_models.py                # Exports the MLModelTrainer class for training and predicting with models
│   ├── utils.py                    # Utility functions for various tasks
│   ├── config.py                   # Configuration settings for the application
│   ├── templates                   # HTML templates for the web application
│   │   ├── index.html              # Homepage template
│   │   ├── heart_disease.html      # Heart disease prediction page template
│   │   ├── diabetes.html           # Diabetes prediction page template
│   │   ├── breast_cancer.html      # Breast cancer prediction page template
│   │   ├── liver_disease.html      # Liver disease prediction page template
│   │   ├── kidney_disease.html     # Kidney disease prediction page template
│   │   ├── diagnosis_dashboard.html # Diagnosis dashboard page template
│   │   └── about.html              # About page template
│   └── static                      # Static files (CSS and JavaScript)
│       ├── css
│       │   └── styles.css          # CSS styles for the application
│       └── js
│           └── main.js             # JavaScript code for client-side interactions
├── data                            # Directory containing CSV datasets for training and testing
├── models                          # Directory containing trained machine learning model files (.pkl)
├── tests                           # Directory containing unit tests
│   ├── test_app.py                # Unit tests for app.py functionality
│   └── test_models.py             # Unit tests for machine learning model functionalities
├── requirements.txt                # Python dependencies required for the project
├── .gitignore                      # Files and directories to be ignored by Git
├── Dockerfile                      # Instructions for building a Docker image for the application
├── docker-compose.yml              # Definition for running multi-container Docker applications
└── README.md                       # Documentation for the project
```

## Setup Instructions

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd ai-healthcare-diagnostics
   ```

2. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```
   python src/app.py
   ```

4. **Access the application**:
   Open your web browser and go to `http://localhost:5000`.

## Usage

- Navigate through the application to access different health diagnostics.
- Input the required data for predictions and view the results.
- Use the diagnosis dashboard for an overview of predictions.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License.