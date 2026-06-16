document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('breast-cancer-form');
    const resultsSection = document.getElementById('results-section');
    const predictionResult = document.getElementById('prediction-result');

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Collect form data
        const formData = {
            radius_mean: parseFloat(form.radius_mean.value),
            texture_mean: parseFloat(form.texture_mean.value),
            perimeter_mean: parseFloat(form.perimeter_mean.value),
            area_mean: parseFloat(form.area_mean.value),
            smoothness_mean: parseFloat(form.smoothness_mean.value),
            compactness_mean: parseFloat(form.compactness_mean.value),
            concavity_mean: parseFloat(form.concavity_mean.value),
            concave_points_mean: parseFloat(form.concave_points_mean.value),
            symmetry_mean: parseFloat(form.symmetry_mean.value),
            fractal_dimension_mean: parseFloat(form.fractal_dimension_mean.value)
        };

        try {
            const response = await fetch('/predict/breast-cancer', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });

            const data = await response.json();

            if (data.success) {
                const risk = data.risk_level.charAt(0).toUpperCase() + data.risk_level.slice(1);
                const probability = (data.probability * 100).toFixed(1);

                predictionResult.innerHTML = `
                    <div class="alert ${data.risk_level === 'high' ? 'alert-danger' : data.risk_level === 'medium' ? 'alert-warning' : 'alert-success'} mb-4">
                        <h4 class="alert-heading">
                            <i class="fas ${data.risk_level === 'high' ? 'fa-exclamation-triangle' : 'fa-info-circle'} me-2"></i>
                            ${risk} Risk
                        </h4>
                        <p class="mb-0">The model predicts a ${probability}% probability of breast cancer.</p>
                    </div>
                    <div class="text-center">
                        <button onclick="location.reload()" class="btn btn-outline-primary">
                            <i class="fas fa-redo me-2"></i>Start New Prediction
                        </button>
                    </div>
                `;

                resultsSection.style.display = 'block';
                resultsSection.scrollIntoView({ behavior: 'smooth' });
            } else {
                throw new Error(data.error || 'Prediction failed');
            }
        } catch (error) {
            predictionResult.innerHTML = `
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-circle me-2"></i>
                    Error: ${error.message}
                </div>
            `;
            resultsSection.style.display = 'block';
        }
    });
});