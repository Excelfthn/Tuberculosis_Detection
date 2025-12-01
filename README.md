## 🫁 Tuberculosis Detection from Chest X-Ray Images

A complete machine learning pipeline for automated tuberculosis (TB) detection in chest X-ray images using traditional computer vision techniques and machine learning classifiers (SVM and KNN), plus a Streamlit web app.

---

## 🎯 Project Overview

This project implements an end‑to‑end workflow:

- **Image preprocessing** (CLAHE, denoising, normalization)
- **Lung segmentation** (Otsu thresholding + morphology)
- **Feature extraction** (GLCM + LBP)
- **Dimensionality reduction** (PCA)
- **Classification** with:
  - **SVM (RBF kernel)**
  - **KNN (k=7, distance‑weighted)**
- **Interactive Streamlit app** for easy testing and visualization

All experiments and model training are documented in the notebook `pcd-final-project.ipynb`.

---

## 📊 Performance

### SVM Classifier (Primary Model)

- **Accuracy**: 92.62%
- **Precision**: Normal 90%, TB 95%
- **Recall**: Normal 95%, TB 90%
- **F1-Score**: Normal 93%, TB 92%

### KNN Classifier

- **Accuracy**: 92.14%
- **Precision**: Normal 88%, TB 97%
- **Recall**: Normal 97%, TB 87%
- **F1-Score**: Normal 93%, TB 92%

Both models are trained on a **balanced dataset** (Normal = TB) using the same feature representation (GLCM + LBP + PCA).

---

## 🧪 Methodology

### 1. Image Preprocessing

- **Target size**: 512×512 pixels (`IMG_SIZE = 512`)
- **Pipeline**:
  - Convert to grayscale
  - CLAHE (`clipLimit=2.0`, `tileGridSize=(8, 8)`)
  - Median blur (`kernel=3`)
  - Normalize to \[0, 1]

Implementation reference: `preprocess_image` in `pcd-final-project.ipynb`.

### 2. Lung Segmentation

- **Steps**:
  - Otsu thresholding on 8‑bit grayscale
  - Morphological **closing** and **opening** (`disk(5)`)
  - Remove small objects (< 300 pixels)

Implementation reference: `segment_lungs` in the notebook.

### 3. Feature Extraction

- **GLCM (Gray-Level Co-occurrence Matrix)**:
  - Distances: \[1, 2, 4]
  - Angles: \[0, π/4, π/2, 3π/4]
  - Properties: contrast, correlation, energy, homogeneity  
  - **Output**: 48 features

- **LBP (Local Binary Pattern)**:
  - Parameters: P=16, R=2, `method='uniform'`
  - **Output**: 59‑bin normalized histogram

- **Combined feature vector**:
  - **Total**: 48 (GLCM) + 59 (LBP) = **107 features**

Implementation: `extract_glcm_features`, `extract_lbp_features`, `extract_all_features_from_lung`.

### 4. Dataset & Split

- Original dataset: `TB_Chest_Radiography_Database/`
  - `Normal/`
  - `Tuberculosis/`
- **Balancing**:
  - Randomly sample Normal images to match number of TB images
- **Split**:
  - Train/test = 70% / 30% with stratification (`train_test_split`)

### 5. Feature Scaling & PCA

- **Scaler**: `StandardScaler`
- **PCA**: `PCA(n_components=0.95)` (retain 95% variance)
- **Resulting dimension**: 5 principal components (for this dataset)

### 6. Classifiers

- **SVM (primary)**:
  - Kernel: RBF
  - C = 10
  - Gamma = 0.01
  - `class_weight='balanced'`
  - `probability=True` (for `predict_proba`)

- **KNN**:
  - `n_neighbors=7`
  - `weights='distance'`
  - `metric='minkowski'`, `p=2` (Euclidean)

---

## 💾 Saved Models

After running the training notebook, the following files are created in `trained_models/`:

- **`svm_model.pkl`** – Trained SVM classifier
- **`knn_model.pkl`** – Trained KNN classifier
- **`scaler.pkl`** – `StandardScaler` fitted on training features
- **`pca.pkl`** – PCA transformer fitted on scaled training features
- **`model_info.pkl`** – Metadata (accuracies, feature setup, image size, model types)

These are exactly the models used by the Streamlit app.

---

## 📓 Notebook: `pcd-final-project.ipynb`

The notebook contains:

- **Data loading & visualization**
  - Example Normal and TB images
- **Preprocessing & segmentation demos**
  - Before/after plots for preprocessing and lung masks
- **Feature extraction & dataset building**
  - Balanced dataset construction
  - Feature matrix `X_bal` (shape `(1400, 107)`)
- **Train/test split, scaling & PCA**
- **Model training & evaluation**
  - SVM training and metrics
  - KNN training and metrics
  - Confusion matrices and ROC curves for both models
- **Model saving**
  - Creates all `.pkl` files in `trained_models/`
- **Manual testing helper**
  - `test_image_with_visualization(image_path, model_type='svm'|'knn', show_processing_steps=False)`  
  - Visual overlay and detailed printout for any image

Run all cells sequentially to reproduce the results and generate the model files.

---

## 🌐 Streamlit Web App: `app.py`

The web app provides a professional interface for TB detection.

### Features

- **Model loading** from `trained_models/`
- **Image upload** (PNG/JPG/JPEG)
- **Classifier choice**:
  - SVM or KNN (via radio button)
- **Prediction display**:
  - Normal vs TB probabilities
  - Model used (SVM/KNN)
  - Colored result box (green for Normal, red for TB)
- **Visualizations**:
  - TB probability overlay (red mask on lungs)
  - Preprocessing / segmentation steps in a separate tab
  - Basic feature‑pipeline explanation and confidence summary
- **Safety notice**:
  - Clear disclaimer: research tool, not a medical device

### Running the App Locally

```bash
# (Optional) create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # macOS / Linux

# Install dependencies
pip install -r requirements.txt

# Train models (run once)
# Open and execute all cells in pcd-final-project.ipynb
# This will create the trained_models/ directory and .pkl files

# Launch Streamlit app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 📦 Requirements

Basic dependencies (see `requirements.txt` for exact versions):

- **Core**:
  - `numpy`
  - `matplotlib`
  - `opencv-python`
  - `scikit-image`
  - `scikit-learn`
  - `tqdm`
- **Web app**:
  - `streamlit`
  - `Pillow`

Install via:

```bash
pip install -r requirements.txt
```

---

## 📁 Repository Structure

```text
Tuberculosis_Detection/
├── pcd-final-project.ipynb     # Main notebook (training & experiments)
├── app.py                      # Streamlit web application
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── TB_Chest_Radiography_Database/
│   ├── Normal/                 # Normal chest X-rays
│   └── Tuberculosis/           # TB chest X-rays
└── trained_models/             # Created after training
    ├── svm_model.pkl           # Trained SVM classifier
    ├── knn_model.pkl           # Trained KNN classifier
    ├── scaler.pkl              # Feature scaler
    ├── pca.pkl                 # PCA transformer
    └── model_info.pkl          # Model metadata
```

---

## ⚠️ Disclaimer

This project is intended **for research and educational purposes only**.  
It is **not** a medical device and must **not** be used as a substitute for professional medical diagnosis or treatment. Always consult qualified healthcare professionals for clinical decisions.


