# AAM-Net: An Angular-Aware Machine Learning Framework for Automated Motif Feature Extraction and Orientation Mapping in Traditional Handloom Textiles

An advanced, production-ready computer vision and machine learning web application designed for textile informatics researchers, senior developers, and computer vision engineers to analyze, segment, classify, and map the orientation angles of repeating motifs on handloom textile specimens.

---

## 🌟 Key Features

1. **Preprocessing Pipeline**: Integrates local contrast enhancement (CLAHE histogram equalization), Gaussian denoising, and Adaptive Gaussian thresholding.
2. **Motif Detection**: Dynamic contour extraction and bounding-box segmentation to detect repeat motifs.
3. **Multi-Feature Extractor**:
   - **Geometric & Shape**: Area, perimeter, circularity ratio.
   - **Scale/Rotation Invariance**: Hu Moments (Log-transformed).
   - **Edge/Orientation Histograms**: Histogram of Oriented Gradients (HOG) features.
   - **Local keypoints**: SIFT keypoint density evaluation.
   - **Texture analytics**: Gray-Level Co-occurrence Matrix (GLCM) calculating Contrast, Correlation, Energy, and Homogeneity.
4. **PCA Angle Mapping**: Applies Singular Value Decomposition (SVD) on contour coordinate distributions to determine the major and minor axis angles of each motif.
5. **Machine Learning Classifier**: Real-time evaluation using Random Forest, Support Vector Machine (SVM), and Gradient Boosting (XGBoost substitute) for classification into 6 textile classes: Floral, Geometric, Tribal, Temple Border, Animal, and Paisley.
6. **Analytics Dashboard**: Dynamic metrics showing average motif angles, dominant patterns, and distribution histograms.
7. **Interactive UI**: A modern, glassmorphic dark-theme UI with responsive canvas interaction, allowing real-time inspection.
8. **Export Engine**: Full ZIP package download including consolidated CSV report, JSON report, and PNG visualization plots.

---

## 📐 Mathematical Formulation

### 1. PCA-Based Orientation Mapping
To determine the principal orientation of a motif shape, we treat its contour points $P = \{(x_1, y_1), (x_2, y_2), \dots, (x_N, y_N)\}$ as a 2D dataset:

1. **Center Coordinates**: Subtract the mean (centroid) $(\bar{x}, \bar{y})$:
   $$\bar{x} = \frac{1}{N}\sum x_i, \quad \bar{y} = \frac{1}{N}\sum y_i$$
   $$P_{\text{centered}} = P - [\bar{x}, \bar{y}]$$

2. **Covariance Matrix**: Calculate the $2 \times 2$ covariance matrix $C$:
   $$C = \frac{1}{N-1} P_{\text{centered}}^T P_{\text{centered}}$$

3. **Eigen decomposition**: Solve the characteristic equation:
   $$\det(C - \lambda I) = 0$$
   This yields eigenvalues $\lambda_1, \lambda_2$ (where $\lambda_1 \ge \lambda_2$) and corresponding eigenvectors $\mathbf{v_1}, \mathbf{v_2}$.
   - The major eigenvector $\mathbf{v_1} = [v_{1x}, v_{1y}]^T$ aligns with the direction of maximum variance.
   - The principal orientation angle $\theta$ is:
     $$\theta = \arctan2(v_{1y}, v_{1x}) \pmod{180^\circ}$$

---

### 2. Texture Properties (GLCM)
Computed on a spatial Gray-Level Co-occurrence Matrix $P(i, j)$ with quantized gray levels $N_g=16$:

- **Contrast**: Measures local variations:
  $$\text{Contrast} = \sum_{i,j} |i-j|^2 P(i,j)$$
- **Homogeneity**: Measures structural closeness to diagonal:
  $$\text{Homogeneity} = \sum_{i,j} \frac{P(i,j)}{1 + |i-j|^2}$$
- **Energy (Angular Second Moment)**: Uniformity of the texture:
  $$\text{Energy} = \sum_{i,j} P(i,j)^2$$
- **Correlation**: Linear gray-tone dependencies:
  $$\text{Correlation} = \sum_{i,j} \frac{(i - \mu_i)(j - \mu_j)P(i,j)}{\sigma_i \sigma_j}$$

---

## 📁 Project Directory Structure

```text
/AAM-Net/
│
├── backend/
│   ├── main.py              # FastAPI application entrypoint
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── upload.py        # POST /upload
│   │   ├── features.py      # POST /extract-features
│   │   ├── angle_map.py     # POST /angle-map
│   │   ├── classify.py      # POST /classify
│   │   └── export.py        # GET /export
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── image_processing.py
│   │   ├── feature_extractor.py
│   │   ├── pca_mapper.py
│   │   └── classifier.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── session_store.py # In-memory dictionary state
│       └── visualization.py # OpenCV & Matplotlib drawing
│
├── frontend/
│   ├── index.html           # Structure & UI
│   ├── styles.css           # Premium vanilla HSL styling
│   └── app.js               # Pipeline orchestration & interactive canvas
│
├── vercel.json              # Serverless configuration
├── requirements.txt         # Core dependencies
└── README.md                # System documentation
```

---

## 🚀 Installation & Running Locally

### Prerequisites
Make sure you have Python 3.9+ installed.

### Step 1: Clone the repository and navigate
```bash
git clone <repository_url>
cd AAM-Net
```

### Step 2: Set up a virtual environment
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the server
```bash
# From the root directory:
uvicorn backend.main:app --reload --port 8000
```
Open your browser and navigate to `http://127.0.0.1:8000`.

---

## ⚡ Deployment to Vercel

The project is structured to deploy smoothly to Vercel as a hybrid application using `@vercel/python` serverless runtimes.

1. **Install Vercel CLI**:
   ```bash
   npm install -g vercel
   ```
2. **Deploy to Vercel**:
   ```bash
   vercel
   ```
3. **Vercel Settings**:
   Vercel will build your static files from `frontend/` and route API traffic to `backend/main.py` automatically based on the root [vercel.json](file:///c:/Users/Lenovo/OneDrive/Desktop/AAM-Net/vercel.json) file.

---

## 📊 Sample Dataset Structure

If you wish to train classifiers locally using your own handloom motif dataset, arrange your files in the following structure:

```text
/dataset/
│
├── Floral/
│   ├── motif_floral_01.png
│   └── motif_floral_02.png
│
├── Geometric/
│   ├── motif_geom_01.png
│   └── motif_geom_02.png
│
├── Tribal/
│   └── ...
│
├── Temple Border/
│   └── ...
│
├── Animal/
│   └── ...
│
└── Paisley/
    └── ...
```

### Extracting & Exporting Training Data
The class models in `backend/services/classifier.py` train dynamically on startup using synthetic signatures of these classes. To run offline training, write a simple script referencing `FeatureExtractorService.extract_all_features` for all images inside the class folders, save them to a `.csv` matrix, and replace the synthetic loader.
