def validate_input(data, expected_keys):
    """Validate input data against expected keys."""
    for key in expected_keys:
        if key not in data:
            raise ValueError(f"Missing expected key: {key}")

def format_prediction_result(prediction, probability, risk_level):
    """Format the prediction result for output."""
    return {
        'prediction': prediction,
        'probability': probability,
        'risk_level': risk_level
    }

def log_message(message):
    """Log messages to the console or a file."""
    print(message)  # Replace with a logging framework as needed