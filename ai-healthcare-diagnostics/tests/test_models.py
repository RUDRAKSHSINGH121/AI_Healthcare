from unittest import TestCase
from src.ml_models import MLModelTrainer

class TestMLModels(TestCase):
    def setUp(self):
        self.trainer = MLModelTrainer()

    def test_heart_disease_prediction(self):
        features = [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]
        result = self.trainer.predict_heart_disease(features)
        self.assertIn('prediction', result)
        self.assertIn('probability', result)
        self.assertIn('risk_level', result)

    def test_diabetes_prediction(self):
        features = [6, 148, 72, 35, 0, 33.6, 0.627, 50]
        result = self.trainer.predict_diabetes(features)
        self.assertIn('prediction', result)
        self.assertIn('probability', result)
        self.assertIn('risk_level', result)

    def test_breast_cancer_prediction(self):
        features = [17.99, 10.38, 122.8, 1001.0, 0.1186, 0.2776, 0.3001, 0.1471, 0.2419, 0.0789]
        result = self.trainer.predict_breast_cancer(features)
        self.assertIn('prediction', result)
        self.assertIn('probability', result)
        self.assertIn('risk_level', result)

    def test_liver_disease_prediction(self):
        features = [65, 1, 0.7, 0.1, 200, 25, 20, 7.5, 3.5, 0.8]
        result = self.trainer.predict_liver_disease(features)
        self.assertIn('prediction', result)
        self.assertIn('probability', result)
        self.assertIn('risk_level', result)

    def test_kidney_disease_prediction(self):
        features = [50, 80, 1.02, 0, 0, 1, 0, 0, 0, 120, 15, 1.2, 140, 4.5, 15, 40, 6000, 200, 0, 0, 0, 0, 0, 0]
        result = self.trainer.predict_kidney_disease(features)
        self.assertIn('prediction', result)
        self.assertIn('probability', result)
        self.assertIn('risk_level', result)