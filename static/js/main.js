// Main JavaScript for AI Healthcare Diagnostics

document.addEventListener('DOMContentLoaded', function() {
    // Check model status on page load
    checkModelStatus();
    
    // Setup train models button
    setupTrainModelsButton();
});

function checkModelStatus() {
    fetch('/api/models/status')
        .then(response => response.json())
        .then(data => {
            updateModelStatus('heart-model-status', data.heart_model);
            updateModelStatus('diabetes-model-status', data.diabetes_model);
            updateModelStatus('breast-cancer-model-status', data.breast_cancer_model);
            updateModelStatus('liver-model-status', data.liver_model);
            updateModelStatus('kidney-model-status', data.kidney_model);
            
            // Update overall status
            const allModelsAvailable = data.heart_model && data.diabetes_model && 
                                     data.breast_cancer_model && data.liver_model && data.kidney_model;
            updateModelStatus('all-models-status', allModelsAvailable);
        })
        .catch(error => {
            console.error('Error checking model status:', error);
            updateModelStatus('heart-model-status', false);
            updateModelStatus('diabetes-model-status', false);
            updateModelStatus('breast-cancer-model-status', false);
            updateModelStatus('liver-model-status', false);
            updateModelStatus('kidney-model-status', false);
            updateModelStatus('all-models-status', false);
        });
}

function updateModelStatus(elementId, isAvailable) {
    const element = document.getElementById(elementId);
    if (element) {
        if (isAvailable) {
            element.className = 'badge bg-success';
            element.textContent = 'Available';
        } else {
            element.className = 'badge bg-warning';
            element.textContent = 'Not Trained';
        }
    }
}

function setupTrainModelsButton() {
    const trainBtn = document.getElementById('train-models-btn');
    if (trainBtn) {
        trainBtn.addEventListener('click', function() {
            trainModels();
        });
    }
}

function trainModels() {
    const trainBtn = document.getElementById('train-models-btn');
    const originalText = trainBtn.innerHTML;
    
    // Show loading state
    trainBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Training Models...';
    trainBtn.disabled = true;
    
    fetch('/train/models', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Show success message
            trainBtn.innerHTML = '<i class="fas fa-check me-2"></i>Models Trained!';
            trainBtn.className = 'btn btn-success btn-lg';
            
            // Update model status
            setTimeout(() => {
                checkModelStatus();
            }, 1000);
            
            // Show success alert
            showAlert('success', 'Models trained successfully! You can now use the prediction features.');
        } else {
            throw new Error(data.error || 'Failed to train models');
        }
    })
    .catch(error => {
        console.error('Error training models:', error);
        trainBtn.innerHTML = originalText;
        trainBtn.disabled = false;
        showAlert('danger', 'Failed to train models: ' + error.message);
    });
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
    
    // Insert alert at the top of the main content
    const container = document.querySelector('.container');
    if (container) {
        container.insertBefore(alertDiv, container.firstChild);
        
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
    element.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
    element.disabled = true;
    return originalContent;
}

function hideLoading(element, originalContent) {
    element.innerHTML = originalContent;
    element.disabled = false;
}

function formatPercentage(value) {
    return (value * 100).toFixed(1) + '%';
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

function createPredictionResult(prediction, probability, riskLevel, diseaseType) {
    const riskClass = getRiskLevelClass(riskLevel);
    const diseaseIcon = diseaseType === 'heart' ? 'fas fa-heart' : 'fas fa-tint';
    const diseaseName = diseaseType === 'heart' ? 'Heart Disease' : 'Diabetes';
    
    return `
        <div class="row">
            <div class="col-md-6">
                <div class="text-center mb-4">
                    <i class="${diseaseIcon} fa-4x text-primary mb-3"></i>
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
