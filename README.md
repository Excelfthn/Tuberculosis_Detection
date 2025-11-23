# Tuberculosis Detection from Chest X-Ray Images

A machine learning project for automated tuberculosis detection in chest X-ray images using traditional computer vision techniques and Support Vector Machine (SVM) classification.

## 🎯 Project Overview

This project implements a complete pipeline for TB detection achieving **92.86% accuracy** through image preprocessing, lung segmentation, feature extraction, and machine learning classification.

## 🔬 Key Features

- **Advanced Image Preprocessing**: CLAHE enhancement, noise reduction, and normalization
- **Intelligent Lung Segmentation**: Otsu thresholding with morphological operations
- **Robust Feature Extraction**: GLCM (Gray-Level Co-occurrence Matrix) + LBP (Local Binary Pattern)
- **High Performance**: 92.86% accuracy with balanced precision and recall
- **Interactive Visualization**: Real-time TB probability visualization with colored overlays
- **Easy Testing**: Simple function to test any chest X-ray image

## 📊 Performance Results

| Metric | Normal | Tuberculosis | Overall |
|--------|--------|--------------|---------|
| **Precision** | 91% | 95% | 93% |
| **Recall** | 96% | 90% | 93% |
| **F1-Score** | 93% | 93% | 93% |
| **Accuracy** | - | - | **92.86%** |

## 🛠️ Methodology

### 1. Image Preprocessing
- Resize to 512×512 pixels for consistency
- Convert to grayscale
- Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Median blur for noise reduction
- Normalize to [0,1] range

### 2. Lung Segmentation
- Otsu thresholding for binary mask creation
- Morphological operations (closing, opening)
- Small object removal to clean the mask

### 3. Feature Extraction
- **GLCM Features**: Texture analysis including contrast, correlation, energy, and homogeneity
- **LBP Features**: Local Binary Pattern histogram with 59 bins for uniform patterns
- Combined feature vector of 107 dimensions

### 4. Machine Learning Pipeline
- Dataset balancing (equal Normal and TB samples)
- Train-test split (70%-30%) with stratification
- Feature scaling using StandardScaler
- Dimensionality reduction with PCA (95% variance retained)
- SVM classifier with RBF kernel

## 🚀 Quick Start

### Prerequisites
```bash
pip install opencv-python matplotlib numpy scikit-image scikit-learn tqdm
```

### Dataset Structure
```
TB_Chest_Radiography_Database/
├── Normal/
│   ├── Normal-1.png
│   ├── Normal-2.png
│   └── ...
└── Tuberculosis/
    ├── Tuberculosis-1.png
    ├── Tuberculosis-2.png
    └── ...
```

### Usage

1. **Training the Model**:
   - Run all cells in the notebook sequentially
   - The model will be automatically saved in `trained_models/` directory

2. **Testing Individual Images**:
   ```python
   # Test any chest X-ray image with visualization
   test_image_with_visualization(r"path/to/your/chest_xray.png")
   ```

3. **Understanding Results**:
   - Red overlay intensity shows TB probability in lung regions
   - Brighter red = Higher TB probability
   - Prediction confidence displayed with percentages

## 🌐 Streamlit Web Application

### Interactive TB Detection Interface
Run the web application for easy testing and visualization:

```bash
# Install dependencies
pip install -r requirements.txt

# Launch Streamlit app
streamlit run app.py
```

### Web App Features
- **📤 Drag & Drop Upload**: Easy image upload interface
- **🔬 Real-time Analysis**: Instant TB detection results
- **🎨 Visual Feedback**: Interactive probability overlays and processing steps
- **📊 Comprehensive Results**: Detailed confidence scores and model performance metrics
- **🎯 User-Friendly Design**: Clean, professional interface with progress indicators

The web app opens at `http://localhost:8501` and provides the complete TB detection pipeline in an intuitive interface.

## 🎨 Visualization Features

### Real-time TB Probability Visualization
- **Red Overlay**: Intensity corresponds to TB probability
- **Transparency**: 30% opacity to maintain image clarity  
- **Processing Steps**: Optional view of preprocessing pipeline
- **Detailed Results**: File path, prediction, and confidence scores

### Example Output
```
🔬 TB DETECTION RESULT
==================================================
📸 Image: Tuberculosis-57.png
📍 Full Path: C:\path\to\image.png
🎯 Prediction: Tuberculosis  
📊 Normal: 25.3%
📊 Tuberculosis: 74.7%
🔴 Red overlay intensity shows TB probability in lung regions
==================================================
```

## 📁 Repository Structure

```
Tuberculosis_Detection/
├── pcd-final-project.ipynb    # Main notebook with complete pipeline
├── README.md                  # This file
├── TB_Chest_Radiography_Database/  # Dataset directory
│   ├── Normal/               # Normal chest X-rays
│   └── Tuberculosis/         # TB chest X-rays  
└── trained_models/           # Saved model files (created after training)
    ├── svm_model.pkl         # Trained SVM classifier
    ├── scaler.pkl           # Feature scaler
    ├── pca.pkl              # PCA transformer
    └── model_info.pkl       # Model metadata
```

## 🔧 Technical Details

### Image Processing Pipeline
1. **Preprocessing**: CLAHE enhancement with 2.0 clip limit
2. **Segmentation**: Otsu + morphological operations (disk radius=5)
3. **Feature Extraction**: 48 GLCM + 59 LBP features
4. **Classification**: SVM with RBF kernel (C=10, γ=0.01)

### Model Parameters
- **SVM Kernel**: RBF (Radial Basis Function)
- **Regularization (C)**: 10
- **Gamma**: 0.01  
- **Class Weight**: Balanced
- **PCA Components**: 95% variance retention

## 📈 Model Performance Analysis

- **High Precision for TB (95%)**: Minimizes false positives
- **High Recall for Normal (96%)**: Excellent normal case detection
- **Balanced Performance**: Similar F1-scores for both classes
- **Robust Feature Set**: GLCM + LBP combination proves effective

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- TB Chest Radiography Database for providing the dataset
- scikit-image and scikit-learn communities for excellent libraries
- Computer vision and machine learning research community

## 📧 Contact

**Project Maintainer**: [Excelfthn](https://github.com/Excelfthn)
- GitHub: [@Excelfthn](https://github.com/Excelfthn)
- Repository: [Tuberculosis_Detection](https://github.com/Excelfthn/Tuberculosis_Detection)

---

⭐ **Star this repository if you found it helpful!** ⭐
