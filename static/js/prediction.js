// Enhanced Prediction functionality with download capabilities

let currentPredictionData = null;
let currentSmilesHash = null;

function predictProperty(propertyType) {
    const smiles = document.getElementById('smiles-input').value.trim();
    const resultsDiv = document.getElementById('results');
    const loadingDiv = document.getElementById('loading');
    const errorDiv = document.getElementById('error');
    
    // Clear previous results
    resultsDiv.innerHTML = '';
    errorDiv.innerHTML = '';
    resultsDiv.style.display = 'none';
    errorDiv.style.display = 'none';
    currentPredictionData = null;
    
    // Validate input
    if (!smiles) {
        showError('Please enter a SMILES string');
        return;
    }
    
    // Show loading
    loadingDiv.style.display = 'block';
    
    // Make API request
    fetch('/api/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            smiles: smiles,
            property_type: propertyType
        })
    })
    .then(response => response.json())
    .then(data => {
        loadingDiv.style.display = 'none';
        
        if (data.error) {
            showError(data.error);
        } else {
            currentPredictionData = data;
            currentSmilesHash = smiles.replace(/[\/\\:*]/g, '_').substring(0, 100);
            displayResults(data, propertyType);
        }
    })
    .catch(error => {
        loadingDiv.style.display = 'none';
        showError('Network error: ' + error.message);
    });
}

function showError(message) {
    const errorDiv = document.getElementById('error');
    errorDiv.innerHTML = `<p><strong>⚠️ Error:</strong> ${message}</p>`;
    errorDiv.style.display = 'block';
    errorDiv.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function displayResults(data, propertyType) {
    const resultsDiv = document.getElementById('results');
    
    // Property-specific titles and descriptions
    const propertyInfo = {
        'esol': {
            title: 'Water Solubility (ESOL)',
            description: 'Aqueous solubility prediction',
            icon: '💧'
        },
        'lipo': {
            title: 'Lipophilicity (LogP)',
            description: 'Oil/water partition coefficient',
            icon: '🧪'
        },
        'freesolv': {
            title: 'Hydration Free Energy',
            description: 'Thermodynamic solvation energy',
            icon: '⚡'
        }
    };
    
    const info = propertyInfo[propertyType] || { title: 'Property', description: '', icon: '🔬' };
    
    let html = `
        <div class="result-card" style="animation-delay: 0.1s">
            <h3>${info.icon} Prediction Results</h3>
            <div class="prediction-value">
                <h4>${info.title}</h4>
                <p class="value">${data.predicted_value.toFixed(3)} ${data.unit}</p>
                <p style="color: var(--text-secondary); margin-top: 12px; font-size: 0.95em;">${info.description}</p>
            </div>
            
            <div class="molecule-info">
                <p><strong>Input SMILES:</strong> ${data.smiles}</p>
                <p><strong>Canonical SMILES:</strong> ${data.canonical_smiles}</p>
                <p><strong>Number of Atoms:</strong> ${data.num_atoms}</p>
                <p><strong>Timestamp:</strong> ${new Date(data.timestamp).toLocaleString()}</p>
            </div>
            
            ${generateDownloadSection(data)}
        </div>
        
        <div class="result-card" style="animation-delay: 0.2s">
            <h3>📊 Explanation Quality</h3>
            <div class="quality-metrics">
                <div class="metric">
                    <span class="metric-label">Consistency:</span>
                    <span class="metric-value">${(data.explanation_quality.consistency * 100).toFixed(1)}%</span>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${data.explanation_quality.consistency * 100}%"></div>
                    </div>
                    <p style="color: var(--text-secondary); font-size: 0.9em; margin-top: 8px;">Agreement between dual baselines</p>
                </div>
            </div>
        </div>
    `;
    
    // Add fragment analysis if available
    if (data.fragments && Object.keys(data.fragments).length > 0) {
        html += `
            <div class="result-card" style="animation-delay: 0.4s">
                <h3>🧬 Fragment Analysis</h3>
                <p style="color: var(--text-secondary); margin-bottom: 20px;">Chemical groups and their contributions to the predicted property</p>
                <div class="fragments-list">
        `;
        
        // Sort fragments by attribution value
        const sortedFragments = Object.entries(data.fragments).sort((a, b) => 
            Math.abs(b[1].attribution) - Math.abs(a[1].attribution)
        );
        
        for (const [name, frag] of sortedFragments) {
            const absValue = Math.abs(frag.attribution);
            const barWidth = Math.min((absValue * 100), 100).toFixed(1);
            const colorClass = frag.attribution > 0 ? 'positive' : 'negative';
            
            html += `
                <div class="fragment-item ${colorClass}">
                    <div class="fragment-header">
                        <span class="fragment-name">${name}</span>
                        <span class="fragment-value ${colorClass}">${frag.attribution.toFixed(4)}</span>
                    </div>
                    <div class="fragment-bar ${colorClass}" style="width: ${barWidth}%"></div>
                    <p class="fragment-interpretation">${frag.interpretation}</p>
                    <p class="fragment-atoms">Atoms: ${frag.atoms.join(', ')}</p>
                </div>
            `;
        }
        
        html += `
                </div>
            </div>
        `;
    }
    
    // Add baseline attributions comparison
    if (data.baseline_attributions && data.baseline_attributions.ig_zero && data.baseline_attributions.ig_skeleton) {
        html += `
            <div class="result-card" style="animation-delay: 0.5s">
                <h3>🔬 Baseline Attributions Comparison</h3>
                <p style="color: var(--text-secondary); margin-bottom: 20px;">Comparing different baseline methods for robust explanations</p>
                
                <div class="baseline-section">
                    <h4>1️⃣ IG Zero Baseline</h4>
                    <p class="baseline-description">Compares molecule to empty space (all zeros) - shows raw importance</p>
                    <div class="attributions-chart">
                        <canvas id="ig-zero-canvas"></canvas>
                    </div>
                </div>
                
                <div class="baseline-section">
                    <h4>2️⃣ IG Skeleton Baseline</h4>
                    <p class="baseline-description">Compares molecule to carbon-only skeleton - highlights heteroatom contributions</p>
                    <div class="attributions-chart">
                        <canvas id="ig-skeleton-canvas"></canvas>
                    </div>
                </div>
                
                <div class="baseline-section">
                    <h4>3️⃣ MB-EIG (Multi-Baseline Average)</h4>
                    <p class="baseline-description">Robust average of both baselines - final explanation used</p>
                    <div class="attributions-chart">
                        <canvas id="mb-eig-canvas"></canvas>
                    </div>
                </div>
            </div>
        `;
    }
    
    // Add generated images if available
    if (data.images) {
        if (data.images.attribution_map) {
            html += `
                <div class="result-card" style="animation-delay: 0.6s">
                    <h3>🎨 Molecule Attribution Map</h3>
                    <p style="color: var(--text-secondary); margin-bottom: 15px;">Visual representation of atom importance (darker blue = more important)</p>
                    <div class="image-container">
                        <img src="${data.images.attribution_map}?t=${Date.now()}" alt="Attribution Map" class="result-image" />
                    </div>
                </div>
            `;
        }
        
        if (data.images.analysis_plot) {
            html += `
                <div class="result-card" style="animation-delay: 0.7s">
                    <h3>📊 Detailed Attribution Analysis</h3>
                    <p style="color: var(--text-secondary); margin-bottom: 15px;">Comprehensive visualization of all attribution methods and fragment analysis</p>
                    <div class="image-container">
                        <img src="${data.images.analysis_plot}?t=${Date.now()}" alt="Analysis Plot" class="result-image" />
                    </div>
                </div>
            `;
        }
    }
    
    resultsDiv.innerHTML = html;
    resultsDiv.style.display = 'block';
    
    // Scroll to results
    resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
    
    // Draw all attribution charts
    if (data.baseline_attributions && data.baseline_attributions.ig_zero) {
        setTimeout(() => {
            drawAttributionsChart(data.baseline_attributions.ig_zero, 'ig-zero-canvas', 'IG Zero');
            drawAttributionsChart(data.baseline_attributions.ig_skeleton, 'ig-skeleton-canvas', 'IG Skeleton');
            drawAttributionsChart(data.mb_eig_attributions, 'mb-eig-canvas', 'MB-EIG');
        }, 100);
    }
}

function generateDownloadSection(data) {
    if (!data.download_files) return '';
    
    return `
        <div class="download-section">
            <h4>📥 Download Results</h4>
            <p style="color: var(--text-secondary); margin-bottom: 16px;">Download all prediction results as a ZIP archive</p>
            <div class="download-buttons">
                <button class="download-btn" onclick="downloadAllFiles()" style="background: var(--gradient-primary);">📦 Download All Results (ZIP)</button>
            </div>
        </div>
    `;
}

function downloadAllFiles() {
    if (!currentSmilesHash) {
        showError('No prediction data available for download');
        return;
    }
    
    const link = document.createElement('a');
    link.href = `/api/download-all/${currentSmilesHash}`;
    link.download = `${currentSmilesHash}_results.zip`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

function drawAttributionsChart(attributions, canvasId, title) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.parentElement.clientWidth;
    const height = 250;
    canvas.width = width;
    canvas.height = height;
    
    const padding = 50;
    const chartWidth = width - 2 * padding;
    const chartHeight = height - 2 * padding;
    
    const maxValue = Math.max(...attributions.map(Math.abs), 0.01);
    const barWidth = Math.max(chartWidth / attributions.length, 8);
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Draw background
    ctx.fillStyle = '#f8fafc';
    ctx.fillRect(0, 0, width, height);
    
    // Draw axes
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.stroke();
    
    // Draw bars
    attributions.forEach((value, index) => {
        const barHeight = (Math.abs(value) / maxValue) * chartHeight;
        const x = padding + index * barWidth;
        const y = height - padding - barHeight;
        
        // Gradient for bars
        const gradient = ctx.createLinearGradient(x, y, x, height - padding);
        gradient.addColorStop(0, '#3b82f6');
        gradient.addColorStop(1, '#1e3a8a');
        
        ctx.fillStyle = gradient;
        ctx.fillRect(x + 2, y, barWidth - 4, barHeight);
        
        // Draw atom index
        if (barWidth > 12) {
            ctx.fillStyle = '#64748b';
            ctx.font = '10px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(index + 1, x + barWidth / 2, height - padding + 15);
        }
    });
    
    // Draw title
    ctx.fillStyle = '#1e3a8a';
    ctx.font = 'bold 14px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(title + ' - Atom Attributions', width / 2, 25);
    
    // Draw max value indicator
    ctx.fillStyle = '#64748b';
    ctx.font = '11px Arial';
    ctx.textAlign = 'right';
    ctx.fillText('Max: ' + maxValue.toFixed(3), width - padding, padding + 20);
    
    // Draw axis labels
    ctx.fillStyle = '#1e293b';
    ctx.font = 'bold 12px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Atom Index', width / 2, height - 10);
    
    ctx.save();
    ctx.translate(15, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText('Attribution Score', 0, 0);
    ctx.restore();
}

// Example SMILES for quick testing
function loadExample(smiles) {
    document.getElementById('smiles-input').value = smiles;
    document.getElementById('smiles-input').focus();
}

// Clear results
function clearResults() {
    document.getElementById('smiles-input').value = '';
    document.getElementById('results').style.display = 'none';
    document.getElementById('error').style.display = 'none';
    currentPredictionData = null;
    currentSmilesHash = null;
}
