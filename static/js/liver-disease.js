document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('liver-disease-form');
    const resultsSection = document.getElementById('results-section');
    const predictionResult = document.getElementById('prediction-result');

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Collect form data
        const formData = {
            age: parseFloat(form.age.value),
            gender: parseInt(form.gender.value),
            total_bilirubin: parseFloat(form.total_bilirubin.value),
            direct_bilirubin: parseFloat(form.direct_bilirubin.value),
            alkaline_phosphotase: parseFloat(form.alkaline_phosphotase.value),
            alamine_aminotransferase: parseFloat(form.alamine_aminotransferase.value),
            aspartate_aminotransferase: parseFloat(form.aspartate_aminotransferase.value),
            total_proteins: parseFloat(form.total_proteins.value),
            albumin: parseFloat(form.albumin.value),
            albumin_and_globulin_ratio: parseFloat(form.albumin_and_globulin_ratio.value)
        };

        try {
            const response = await fetch('/predict/liver-disease', {
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
                        <p class="mb-0">The model predicts a ${probability}% probability of liver disease.</p>
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