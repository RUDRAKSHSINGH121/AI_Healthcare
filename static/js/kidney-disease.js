document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('kidney-disease-form');
    const resultsSection = document.getElementById('results-section');
    const predictionResult = document.getElementById('prediction-result');

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Collect form data
        const formData = {
            age: parseFloat(form.age.value),
            bp: parseFloat(form.bp.value),
            sg: parseFloat(form.sg.value),
            al: parseInt(form.al.value),
            su: parseInt(form.su.value),
            rbc: parseInt(form.rbc.value),
            pc: parseInt(form.pc.value),
            pcc: parseInt(form.pcc.value),
            ba: parseInt(form.ba.value),
            bgr: parseFloat(form.bgr.value),
            bu: parseFloat(form.bu.value),
            sc: parseFloat(form.sc.value),
            sod: parseFloat(form.sod.value),
            pot: parseFloat(form.pot.value),
            hemo: parseFloat(form.hemo.value),
            pcv: parseFloat(form.pcv.value),
            wbcc: parseFloat(form.wbcc.value),
            rbcc: parseFloat(form.rbcc.value),
            htn: parseInt(form.htn.value),
            dm: parseInt(form.dm.value),
            cad: parseInt(form.cad.value),
            appet: parseInt(form.appet.value),
            pe: parseInt(form.pe.value),
            ane: parseInt(form.ane.value)
        };

        try {
            const response = await fetch('/predict/kidney-disease', {
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
                        <p class="mb-0">The model predicts a ${probability}% probability of kidney disease.</p>
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