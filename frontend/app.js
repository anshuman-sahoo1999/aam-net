// Global App State
const state = {
    sessionId: null,
    currentView: 'original', // original, processed, contours, vectors, heatmap
    motifs: [],
    features: [],
    angleMaps: [],
    classifications: [],
    analytics: {},
    images: {
        original: null, // Image element
        processed: null,
        contours: null,
        vectors: null,
        heatmap: null
    },
    selectedMotifId: null,
    activeModel: 'Random Forest',
    classChartInstance: null
};

// API Base URL (assumes same host when hosted via FastAPI, fallback to local localhost)
const API_BASE = window.location.origin;

// DOM Elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const progressSection = document.getElementById('progressSection');
const progressBar = document.getElementById('progressBar');
const progressPercent = document.getElementById('progressPercent');
const dashboardView = document.getElementById('dashboardView');
const chartsSection = document.getElementById('chartsSection');
const interactiveCanvas = document.getElementById('interactiveCanvas');
const canvasCtx = interactiveCanvas.getContext('2d');
const canvasLoader = document.getElementById('canvasLoader');
const telemetryContent = document.getElementById('telemetryContent');
const modelSelect = document.getElementById('modelSelect');
const reclassifyBtn = document.getElementById('reclassifyBtn');

// View Controls Buttons
const viewTabs = document.querySelectorAll('.btn-tab');

// Export Buttons
const exportCsvBtn = document.getElementById('exportCsvBtn');
const exportJsonBtn = document.getElementById('exportJsonBtn');
const exportZipBtn = document.getElementById('exportZipBtn');

// Init listeners
document.addEventListener('DOMContentLoaded', () => {
    setupDragAndDrop();
    setupViewTabs();
    setupModelSelector();
    setupExportButtons();
    setupResponsiveNavigation();
});

// Drag & Drop Setup
function setupDragAndDrop() {
    browseBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });
    
    // Clicking the dropZone card also triggers input selection
    dropZone.addEventListener('click', () => {
        fileInput.click();
    });
}

// File handling and validation
function handleFile(file) {
    const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg'];
    if (!allowedTypes.includes(file.type)) {
        alert('Invalid file format. Please select a PNG or JPEG image.');
        return;
    }

    const maxBytes = 10 * 1024 * 1024; // 10MB
    if (file.size > maxBytes) {
        alert('File is too large. Maximum size allowed is 10MB.');
        return;
    }

    // Begin processing pipeline
    runPipeline(file);
}

// Pipeline Orchestration
async function runPipeline(file) {
    // Show progress card, hide dashboard/charts if previously run
    progressSection.classList.remove('hidden');
    dashboardView.classList.add('hidden');
    chartsSection.classList.add('hidden');
    state.selectedMotifId = null;
    
    updateProgress(0, 'step1', 'Uploading & Preprocessing...');

    const formData = new FormData();
    formData.append('file', file);

    try {
        // Step 1: Upload & Preprocess
        const uploadRes = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData
        });

        if (!uploadRes.ok) {
            const err = await uploadRes.json();
            throw new Error(err.detail || 'Upload failed');
        }

        const uploadData = await uploadRes.json();
        state.sessionId = uploadData.session_id;
        state.motifs = uploadData.motifs;
        
        // Load original and processed images for canvas view
        updateProgress(25, 'step2', 'Detecting repeating patterns/motifs...');
        state.images.original = await loadImage(uploadData.original_preview);
        state.images.processed = await loadImage(uploadData.processed_preview);

        // Step 2: Extract features
        updateProgress(50, 'step3', 'Extracting geometric, shape, and GLCM texture features...');
        const featRes = await fetch(`${API_BASE}/extract-features`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: state.sessionId })
        });
        
        if (!featRes.ok) throw new Error('Feature extraction failed');
        const featData = await featRes.ok ? await featRes.json() : {};
        state.features = featData.features;

        // Step 3: PCA Orientation Angle Mapping
        updateProgress(75, 'step4', 'Mapping orientations using principal component analysis (PCA)...');
        const angleRes = await fetch(`${API_BASE}/angle-map`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: state.sessionId })
        });

        if (!angleRes.ok) throw new Error('PCA mapping failed');
        const angleData = await angleRes.json();
        state.angleMaps = angleData.angle_maps;
        
        // Load overlay, vector, and heatmap images
        state.images.contours = await loadImage(angleData.visualizations.contour_overlay);
        state.images.vectors = await loadImage(angleData.visualizations.orientation_vectors);
        state.images.heatmap = await loadImage(angleData.visualizations.angle_heatmap);
        
        // Load distribution plot image
        document.getElementById('distImage').src = angleData.visualizations.feature_distribution;

        // Step 4: Machine Learning Classification
        updateProgress(90, 'step4', 'Applying active classifier model...');
        await runClassification();

        // Finish
        updateProgress(100, 'step4', 'Specimen analysis complete!');
        setTimeout(() => {
            progressSection.classList.add('hidden');
            dashboardView.classList.remove('hidden');
            chartsSection.classList.remove('hidden');
            
            // Default view: Original
            setView('original');
            updateAnalyticsDashboard();
            renderClassChart();
        }, 800);

    } catch (error) {
        console.error(error);
        alert(`Error: ${error.message}`);
        progressSection.classList.add('hidden');
    }
}

// Classifier request trigger
async function runClassification() {
    canvasLoader.classList.remove('hidden');
    
    const classifyRes = await fetch(`${API_BASE}/classify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
            session_id: state.sessionId,
            model_type: state.activeModel
        })
    });

    if (!classifyRes.ok) {
        canvasLoader.classList.add('hidden');
        throw new Error('Classification request failed');
    }
    
    const classifyData = await classifyRes.json();
    state.classifications = classifyData.classifications;
    state.analytics = classifyData.analytics;
    canvasLoader.classList.add('hidden');
}

// Update the progress card display
function updateProgress(percent, activeStepId, message) {
    progressBar.style.width = `${percent}%`;
    progressPercent.innerText = `${percent}%`;
    
    // Clear all steps active classes
    document.querySelectorAll('.pipeline-steps .step').forEach(step => {
        step.classList.remove('active');
    });
    
    const activeStep = document.getElementById(activeStepId);
    if (activeStep) {
        activeStep.classList.add('active');
    }
    
    if (percent === 100) {
        document.querySelectorAll('.pipeline-steps .step').forEach(step => {
            step.classList.add('complete');
        });
    }
}

// Image Loader Helper
function loadImage(src) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = (e) => reject(e);
        img.src = src;
    });
}

// View switcher
function setupViewTabs() {
    viewTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            viewTabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            
            const view = tab.getAttribute('data-view');
            setView(view);
        });
    });
}

function setView(view) {
    state.currentView = view;
    renderCanvas();
}

// Render selected view on Canvas
function renderCanvas() {
    const img = state.images[state.currentView];
    if (!img) return;

    // Set dimensions
    interactiveCanvas.width = img.naturalWidth;
    interactiveCanvas.height = img.naturalHeight;
    
    // Draw
    canvasCtx.drawImage(img, 0, 0);
    
    // If selected motif is active, highlight it with a copper bounding box overlay
    if (state.selectedMotifId !== null) {
        const motif = state.motifs.find(m => m.id === state.selectedMotifId);
        if (motif) {
            const { x, y, width, height } = motif.bounding_box;
            canvasCtx.strokeStyle = 'hsl(24, 75%, 54%)';
            canvasCtx.lineWidth = 3;
            canvasCtx.strokeRect(x, y, width, height);
            canvasCtx.fillStyle = 'rgba(211, 94, 53, 0.15)';
            canvasCtx.fillRect(x, y, width, height);
        }
    }
}

// Interactive clicking on canvas
interactiveCanvas.addEventListener('click', (e) => {
    if (!state.images.original) return;
    
    // Calculate click coordinates in image coordinates
    const rect = interactiveCanvas.getBoundingClientRect();
    const scaleX = interactiveCanvas.width / rect.width;
    const scaleY = interactiveCanvas.height / rect.height;
    
    const clickX = (e.clientX - rect.left) * scaleX;
    const clickY = (e.clientY - rect.top) * scaleY;
    
    // Find closest motif centroid
    let closestMotif = null;
    let minDistance = Infinity;
    
    state.motifs.forEach(m => {
        const [cx, cy] = m.centroid;
        const dist = Math.hypot(clickX - cx, clickY - cy);
        if (dist < minDistance) {
            minDistance = dist;
            closestMotif = m;
        }
    });
    
    // Threshold distance: if clicked inside or close to the bounding box
    if (closestMotif) {
        const { x, y, width, height } = closestMotif.bounding_box;
        const padding = 30; // Max click allowance padding
        if (clickX >= x - padding && clickX <= x + width + padding &&
            clickY >= y - padding && clickY <= y + height + padding) {
            
            selectMotif(closestMotif.id);
            return;
        }
    }
    
    // Clicked empty space
    state.selectedMotifId = null;
    renderCanvas();
    renderEmptyTelemetry();
});

// Select motif and show data
function selectMotif(id) {
    state.selectedMotifId = id;
    renderCanvas();
    
    // Fetch info from state
    const meta = state.motifs.find(m => m.id === id);
    const feat = state.features.find(f => f.motif_id === id);
    const angle = state.angleMaps.find(a => a.motif_id === id);
    const cls = state.classifications.find(c => c.motif_id === id);
    
    if (!feat || !angle || !cls) return;
    
    let probBars = '';
    Object.entries(cls.probabilities).forEach(([className, score]) => {
        probBars += `
            <div style="margin-top: 0.5rem;">
                <div style="display:flex; justify-content:space-between; font-size:0.75rem;">
                    <span>${className}</span>
                    <span>${Math.round(score * 100)}%</span>
                </div>
                <div class="class-prob-bar">
                    <div class="class-prob-fill" style="width: ${score * 100}%"></div>
                </div>
            </div>
        `;
    });
    
    telemetryContent.innerHTML = `
        <div class="telemetry-item">
            <h4 style="color:var(--primary); font-family:var(--font-heading); font-size:1.1rem; margin-bottom:0.8rem;">Motif #${id} Telemetry</h4>
            <table class="telemetry-table">
                <tr>
                    <th>Predicted Class</th>
                    <td style="color:var(--secondary); font-size:1rem;">${cls.predicted_class}</td>
                </tr>
                <tr>
                    <th>Confidence</th>
                    <td>${Math.round(cls.confidence_score * 100)}%</td>
                </tr>
                <tr>
                    <th>Centroid Coord</th>
                    <td>(${Math.round(angle.x_coordinate)}, ${Math.round(angle.y_coordinate)})</td>
                </tr>
                <tr>
                    <th>Orientation Angle</th>
                    <td style="color:var(--primary);">${Math.round(angle.angle_degree)}&deg;</td>
                </tr>
                <tr>
                    <th>Major Axis Angle</th>
                    <td>${Math.round(angle.major_axis_angle)}&deg;</td>
                </tr>
                <tr>
                    <th>Minor Axis Angle</th>
                    <td>${Math.round(angle.minor_axis_angle)}&deg;</td>
                </tr>
                <tr>
                    <th>Circularity Ratio</th>
                    <td>${feat.circularity.toFixed(4)}</td>
                </tr>
                <tr>
                    <th>Texture Score (GLCM)</th>
                    <td>${feat.texture_score.toFixed(4)}</td>
                </tr>
                <tr>
                    <th>Perimeter</th>
                    <td>${Math.round(feat.perimeter)} px</td>
                </tr>
                <tr>
                    <th>Pixel Area</th>
                    <td>${Math.round(feat.area)} px&sup2;</td>
                </tr>
            </table>
            
            <h4 style="margin: 1.2rem 0 0.5rem; font-size:0.9rem; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom:0.3rem;">Classification Probabilities</h4>
            ${probBars}
        </div>
    `;
}

function renderEmptyTelemetry() {
    telemetryContent.innerHTML = `
        <div class="empty-state">
            <i class="fa-solid fa-arrow-pointer"></i>
            <p>Click on a motif shape on the left canvas to view geometric, texture, and orientation features.</p>
        </div>
    `;
}

// Update Top Analytics panel
function updateAnalyticsDashboard() {
    document.getElementById('statTotalMotifs').innerText = state.analytics.total_motifs;
    document.getElementById('statAvgAngle').innerText = `${Math.round(state.analytics.average_angle)}°`;
    document.getElementById('statDominantClass').innerText = state.analytics.most_frequent_class;
}

// Setup Model selection re-runs
function setupModelSelector() {
    reclassifyBtn.addEventListener('click', async () => {
        state.activeModel = modelSelect.value;
        try {
            await runClassification();
            updateAnalyticsDashboard();
            renderClassChart();
            if (state.selectedMotifId !== null) {
                selectMotif(state.selectedMotifId);
            }
        } catch (e) {
            alert('Reclassification failed: ' + e.message);
        }
    });
}

// Chart.js render
function renderClassChart() {
    const ctx = document.getElementById('classChart').getContext('2d');
    
    if (state.classChartInstance) {
        state.classChartInstance.destroy();
    }
    
    const labels = Object.keys(state.analytics.class_distribution);
    const data = Object.values(state.analytics.class_distribution);
    
    state.classChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Motif Count',
                data: data,
                backgroundColor: 'rgba(211, 94, 53, 0.6)',
                borderColor: 'rgba(211, 94, 53, 1)',
                borderWidth: 1.5,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                title: {
                    display: true,
                    text: 'Class Distribution Histogram',
                    color: '#fff',
                    font: { family: 'Outfit', size: 14 }
                }
            },
            scales: {
                y: {
                    ticks: { color: 'rgba(255, 255, 255, 0.7)' },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' }
                },
                x: {
                    ticks: { color: 'rgba(255, 255, 255, 0.7)' },
                    grid: { display: false }
                }
            }
        }
    });
}

// Export files Setup
function setupExportButtons() {
    exportCsvBtn.addEventListener('click', () => {
        if (!state.sessionId) return;
        window.open(`${API_BASE}/export?session_id=${state.sessionId}&format=csv`);
    });
    
    exportJsonBtn.addEventListener('click', () => {
        if (!state.sessionId) return;
        window.open(`${API_BASE}/export?session_id=${state.sessionId}&format=json`);
    });
    
    exportZipBtn.addEventListener('click', () => {
        if (!state.sessionId) return;
        window.open(`${API_BASE}/export?session_id=${state.sessionId}&format=zip`);
    });
}

// Documentation Tab Switcher
window.switchDocTab = function(tabId) {
    // Hide all tab content
    const contents = document.querySelectorAll('.doc-tab-content');
    contents.forEach(content => content.classList.add('hidden'));
    
    // Show selected tab content
    const activeContent = document.getElementById(`doc-${tabId}`);
    if (activeContent) {
        activeContent.classList.remove('hidden');
    }
    
    // Manage active state of buttons
    const buttons = document.querySelectorAll('.doc-tabs .btn-tab');
    buttons.forEach(btn => {
        btn.classList.remove('active');
        // Extract tab target from onclick attribute
        const onclickAttr = btn.getAttribute('onclick');
        if (onclickAttr && onclickAttr.includes(tabId)) {
            btn.classList.add('active');
        }
    });
};

// Responsive Navigation Event Handlers
function setupResponsiveNavigation() {
    const navToggle = document.getElementById('navToggle');
    const navMenu = document.getElementById('navMenu');
    const docDropdownToggle = document.getElementById('docDropdownToggle');
    
    // Toggle mobile menu drawer
    if (navToggle) {
        navToggle.addEventListener('click', (e) => {
            e.stopPropagation();
            navMenu.classList.toggle('active');
        });
    }
    
    // Toggle sub-dropdown list on mobile view when clicking "Documentation"
    if (docDropdownToggle) {
        docDropdownToggle.addEventListener('click', (e) => {
            if (window.innerWidth <= 992) {
                e.preventDefault();
                e.stopPropagation();
                docDropdownToggle.parentElement.classList.toggle('active');
            }
        });
    }
    
    // Close mobile menu if clicking anywhere else
    document.addEventListener('click', () => {
        if (navMenu && navMenu.classList.contains('active')) {
            navMenu.classList.remove('active');
        }
    });
    
    // Global helper to close mobile menu after clicking links
    window.closeMobileMenu = function() {
        if (navMenu) {
            navMenu.classList.remove('active');
            if (docDropdownToggle) {
                docDropdownToggle.parentElement.classList.remove('active');
            }
        }
    };
}


