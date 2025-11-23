# Streamlit TB Detection App Runner

## Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Make sure you have trained models
Before running the Streamlit app, ensure you have trained the model using the Jupyter notebook (`pcd-final-project.ipynb`). The app expects these files in the `trained_models/` directory:
- `svm_model.pkl`
- `scaler.pkl` 
- `pca.pkl`

### 3. Run the Streamlit App
```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

## Features

### 🎨 Interactive Web Interface
- **File Upload**: Drag and drop or browse for chest X-ray images (PNG, JPG, JPEG)
- **Real-time Analysis**: Instant TB detection with probability scores
- **Visual Results**: Color-coded results with confidence indicators
- **Processing Visualization**: Step-by-step image processing pipeline

### 🔬 Advanced Visualization
- **TB Probability Overlay**: Red intensity shows TB probability in lung regions
- **Processing Steps**: View CLAHE preprocessing, lung segmentation, and feature extraction
- **Feature Analysis**: Detailed breakdown of GLCM + LBP feature extraction process

### 📊 Comprehensive Results
- **Prediction**: Clear Normal/TB classification
- **Confidence Scores**: Percentage probabilities for both classes
- **Visual Feedback**: Progress bars and color-coded result boxes
- **Model Performance**: Built-in display of 92.86% accuracy metrics

## Usage Tips

1. **Image Quality**: Use clear, well-contrasted chest X-ray images for best results
2. **File Formats**: Supports PNG, JPG, and JPEG formats
3. **Processing Time**: Analysis typically takes 2-3 seconds per image
4. **Interpretation**: Red overlay intensity indicates TB probability - brighter red means higher TB likelihood

## Troubleshooting

### Model Files Missing
If you see "Model files not found", run the Jupyter notebook first to train and save the models:
```bash
jupyter notebook pcd-final-project.ipynb
```

### Package Installation Issues
For Windows users with OpenCV issues:
```bash
pip install opencv-python-headless
```

### Memory Issues
If you encounter memory issues with large images, the app automatically resizes images to 512x512 pixels for processing.

## Technical Architecture

The Streamlit app implements the complete TB detection pipeline:
1. **Image Upload & Validation**
2. **CLAHE Preprocessing** 
3. **Otsu Lung Segmentation**
4. **GLCM + LBP Feature Extraction** (107 features)
5. **SVM Classification** with probability estimation
6. **Visualization Generation** with TB probability overlay

## Performance Specifications

- **Model Accuracy**: 92.86%
- **TB Detection Precision**: 95%
- **Normal Detection Recall**: 96%
- **Feature Processing**: ~2-3 seconds per image
- **Supported Image Size**: Any size (auto-resized to 512x512)
- **Memory Usage**: ~200MB for model loading