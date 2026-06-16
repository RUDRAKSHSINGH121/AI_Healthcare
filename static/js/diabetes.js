// Diabetes Prediction JavaScript

document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('diabetes-form');
    const resultsSection = document.getElementById('results-section');
    const predictionResult = document.getElementById('prediction-result');
    
    if (form) {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            predictDiabetes();
        });
    }
});

function predictDiabetes() {
    const form = document.getElementById('diabetes-form');
    const submitBtn = form.querySelector('button[type="submit"]');
    const resultsSection = document.getElementById('results-section');
    const predictionResult = document.getElementById('prediction-result');
    
    // Show loading state
    const originalContent = showLoading(submitBtn);
    
    // Collect form data
    const formData = new FormData(form);
    const data = Object.fromEntries(formData.entries());
    
    // Validate form data
    if (!validateDiabetesForm(data)) {
        hideLoading(submitBtn, originalContent);
        showAlert('warning', 'Please fill in all required fields with valid values.');
        return;
    }
    
    // Make prediction request
    fetch('/predict/diabetes', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        hideLoading(submitBtn, originalContent);
        
        if (data.success) {
            // Display results
            predictionResult.innerHTML = createPredictionResult(
                data.prediction, 
                data.probability, 
                data.risk_level, 
                'diabetes'
            );
            
            // Show results section with animation
            resultsSection.style.display = 'block';
            resultsSection.scrollIntoView({ behavior: 'smooth' });
            
            // Show success message
            showAlert('success', 'Diabetes risk assessment completed successfully!');
        } else {
            throw new Error(data.error || 'Prediction failed');
        }
    })
    .catch(error => {
        hideLoading(submitBtn, originalContent);
        console.error('Error predicting diabetes:', error);
        showAlert('danger', 'Failed to predict diabetes risk: ' + error.message);
    });
}

function validateDiabetesForm(data) {
    const requiredFields = [
        'pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 
        'insulin', 'bmi', 'diabetes_pedigree', 'age'
    ];
    
    for (const field of requiredFields) {
        if (!data[field] || data[field].trim() === '') {
            return false;
        }
    }
    
    // Validate numeric ranges
    const pregnancies = parseInt(data.pregnancies);
    const glucose = parseInt(data.glucose);
    const bloodPressure = parseInt(data.blood_pressure);
    const skinThickness = parseInt(data.skin_thickness);
    const insulin = parseInt(data.insulin);
    const bmi = parseFloat(data.bmi);
    const diabetesPedigree = parseFloat(data.diabetes_pedigree);
    const age = parseInt(data.age);
    
    if (pregnancies < 0 || pregnancies > 20) return false;
    if (glucose < 50 || glucose > 300) return false;
    if (bloodPressure < 40 || bloodPressure > 150) return false;
    if (skinThickness < 5 || skinThickness > 100) return false;
    if (insulin < 0 || insulin > 1000) return false;
    if (bmi < 10 || bmi > 80) return false;
    if (diabetesPedigree < 0 || diabetesPedigree > 3) return false;
    if (age < 1 || age > 120) return false;
    
    return true;
}

function showAlert(type, message) {
    // Remove existing alerts
    const existingAlerts = document.querySelectorAll('.alert');
    existingAlerts.forEach(alert => alert.remove());
    
    // Create new alert
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Insert alert at the top of the form
    const form = document.getElementById('diabetes-form');
    if (form) {
        form.parentNode.insertBefore(alertDiv, form);
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }
}

function showLoading(element) {
    const originalContent = element.innerHTML;
    element.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Analyzing...';
    element.disabled = true;
    return originalContent;
}

function hideLoading(element, originalContent) {
    element.innerHTML = originalContent;
    element.disabled = false;
}

function createPredictionResult(prediction, probability, riskLevel, diseaseType) {
    const riskClass = getRiskLevelClass(riskLevel);
    const diseaseIcon = 'fas fa-tint';
    const diseaseName = 'Diabetes';
    
    return `
        <div class="row">
            <div class="col-md-6">
                <div class="text-center mb-4">
                    <i class="${diseaseIcon} fa-4x text-info mb-3"></i>
                    <h3>${diseaseName} Risk Assessment</h3>
                </div>
            </div>
            <div class="col-md-6">
                <div class="mb-4">
                    <h5>Risk Level</h5>
                    <span class="badge ${riskClass} fs-6">${riskLevel.toUpperCase()} RISK</span>
                </div>
                
                <div class="mb-4">
                    <h5>Probability Score</h5>
                    ${createProgressBar(probability * 100, riskLevel)}
                </div>
                
                <div class="mb-4">
                    <h5>Prediction</h5>
                    <p class="fs-5">
                        ${prediction === 1 ? 
                            `<i class="fas fa-exclamation-triangle text-warning me-2"></i>Risk Detected` : 
                            `<i class="fas fa-check-circle text-success me-2"></i>No Risk Detected`
                        }
                    </p>
                </div>
                
                <div class="alert alert-info">
                    <h6 class="alert-heading">
                        <i class="fas fa-info-circle me-2"></i>Important Note
                    </h6>
                    <p class="mb-0 small">
                        This prediction is for educational purposes only. Please consult with a healthcare 
                        professional for proper medical diagnosis and treatment.
                    </p>
                </div>
            </div>
        </div>
    `;
}

function getRiskLevelClass(riskLevel) {
    switch(riskLevel.toLowerCase()) {
        case 'low':
            return 'risk-low';
        case 'medium':
            return 'risk-medium';
        case 'high':
            return 'risk-high';
        default:
            return 'bg-secondary';
    }
}

function createProgressBar(percentage, riskLevel) {
    const riskClass = getRiskLevelClass(riskLevel);
    return `
        <div class="progress mb-3">
            <div class="progress-bar ${riskClass}" role="progressbar" 
                 style="width: ${percentage}%" 
                 aria-valuenow="${percentage}" 
                 aria-valuemin="0" 
                 aria-valuemax="100">
                ${formatPercentage(percentage / 100)}
            </div>
        </div>
    `;
}

function formatPercentage(value) {
    return (value * 100).toFixed(1) + '%';
}

