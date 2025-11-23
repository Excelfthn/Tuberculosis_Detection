# TB Detection System - Demo & Testing Guide

## 🚀 Quick Start

### Option 1: Simple Command Line
```bash
streamlit run app.py
```

### Option 2: Windows Batch File  
Double-click `run_app.bat` (Windows users)

### Option 3: PowerShell Script
```powershell
.\run_app.ps1
```

## 🎮 How to Use the Web Interface

### 1. **Launch the Application**
- Open your browser to `http://localhost:8501`
- You'll see the TB Detection System homepage

### 2. **Upload an Image**
- Click "Browse files" or drag & drop a chest X-ray image
- Supported formats: PNG, JPG, JPEG
- The image will be displayed immediately

### 3. **Analyze the Image**
- Click the "🔬 Analyze Image" button
- Wait 2-3 seconds for processing
- View the results in real-time

### 4. **Interpret Results**
- **Green Box**: Normal (No TB detected)
- **Red Box**: TB detected
- **Confidence Scores**: Percentage probabilities
- **Progress Bars**: Visual probability distribution

### 5. **Explore Visualizations**
- **Tab 1 - TB Probability Overlay**: Red intensity shows TB regions
- **Tab 2 - Processing Steps**: See CLAHE, segmentation, lung extraction
- **Tab 3 - Feature Analysis**: Technical details about feature extraction

## 🧪 Testing with Sample Images

### Test with Normal Images:
Try uploading images from:
```
TB_Chest_Radiography_Database/Normal/Normal-1.png
TB_Chest_Radiography_Database/Normal/Normal-100.png
```

### Test with TB Images:
Try uploading images from:
```
TB_Chest_Radiography_Database/Tuberculosis/Tuberculosis-1.png
TB_Chest_Radiography_Database/Tuberculosis/Tuberculosis-50.png
```

## 🎨 Understanding the Visualizations

### Red Overlay Interpretation
- **Bright Red**: High TB probability (70-100%)
- **Medium Red**: Moderate TB probability (40-70%)  
- **Light Red**: Low TB probability (10-40%)
- **No Red**: Very low TB probability (<10%)

### Processing Steps Explanation
1. **Original**: Raw chest X-ray image
2. **CLAHE Enhanced**: Contrast improvement for better feature extraction
3. **Lung Segmentation**: Otsu thresholding + morphological operations
4. **Lung Regions**: Isolated lung areas for analysis
5. **Feature Extraction**: GLCM (48) + LBP (59) = 107 features
6. **Classification**: SVM with RBF kernel prediction

## 📊 Performance Metrics

The system achieves:
- **Overall Accuracy**: 92.86%
- **TB Detection Precision**: 95% (fewer false positives)
- **Normal Detection Recall**: 96% (catches almost all normal cases)
- **Processing Time**: 2-3 seconds per image
- **Feature Vector**: 107 dimensions (GLCM + LBP)

## 🔧 Troubleshooting

### Common Issues & Solutions

#### 1. "Model files not found"
**Problem**: The trained models are missing
**Solution**: 
```bash
# Run the Jupyter notebook to train models
jupyter notebook pcd-final-project.ipynb
# Execute all cells to create trained_models/ directory
```

#### 2. "Module not found" errors
**Problem**: Missing Python packages
**Solution**:
```bash
pip install -r requirements.txt
# or individually:
pip install streamlit opencv-python scikit-learn scikit-image matplotlib numpy pillow
```

#### 3. Image upload fails
**Problem**: Unsupported image format or corrupted file
**Solution**: 
- Use PNG, JPG, or JPEG formats only
- Ensure image is a valid chest X-ray
- Check file size (recommended < 10MB)

#### 4. Slow processing
**Problem**: Large image files or limited system resources
**Solution**:
- Use images smaller than 2048x2048 pixels
- Close other applications to free up memory
- The app automatically resizes to 512x512 for processing

#### 5. Browser doesn't open automatically
**Problem**: Streamlit doesn't launch browser
**Solution**:
- Manually navigate to `http://localhost:8501`
- Try different browsers (Chrome, Firefox, Edge)
- Check if port 8501 is available

## 🎯 Best Practices

### For Optimal Results:
1. **Image Quality**: Use high-contrast, clear chest X-rays
2. **Proper Positioning**: Standard PA (posterior-anterior) chest X-rays work best
3. **File Format**: PNG format often provides best quality
4. **Image Size**: Any size works (auto-resized to 512x512)

### For Testing & Validation:
1. **Compare Multiple Images**: Test both normal and TB cases
2. **Check Confidence Scores**: Higher confidence (>80%) indicates more reliable predictions
3. **Visual Verification**: Use the red overlay to see which lung regions drive the prediction
4. **Clinical Context**: Always consult medical professionals for actual diagnosis

## 📈 Technical Architecture

### Image Processing Pipeline:
```
Raw Image → Resize (512x512) → Grayscale → CLAHE → Median Blur → Normalize
    ↓
Otsu Threshold → Morphology → Small Object Removal → Lung Mask
    ↓
Masked Lung Image → GLCM Features (48) + LBP Features (59) → Combined Vector (107)
    ↓
StandardScaler → PCA → SVM Classification → Probability Estimation → Result + Visualization
```

### Model Details:
- **Algorithm**: Support Vector Machine (SVM)
- **Kernel**: Radial Basis Function (RBF)
- **Parameters**: C=10, gamma=0.01
- **Features**: Gray-Level Co-occurrence Matrix + Local Binary Patterns
- **Preprocessing**: CLAHE enhancement, Otsu segmentation
- **Dimensionality**: 107 features → PCA reduced

## 🎭 Demo Scenarios

### Scenario 1: Normal Case Detection
1. Upload `Normal-1.png`
2. Expected: Green result box, Normal ~85-95%
3. Visualization: Minimal or no red overlay

### Scenario 2: TB Case Detection  
1. Upload `Tuberculosis-1.png`
2. Expected: Red result box, TB ~70-95%
3. Visualization: Bright red overlay in affected lung regions

### Scenario 3: Borderline Case
1. Upload various images to see confidence variations
2. Note how confidence scores change
3. Observe overlay intensity differences

## 📞 Support & Development

### For Issues:
- Check console logs in browser (F12 → Console)
- Verify all dependencies are installed
- Ensure model files exist in `trained_models/`

### For Customization:
- Modify `app.py` for UI changes
- Adjust model parameters in the notebook
- Update visualization colors and thresholds

### Performance Optimization:
- Use SSD storage for faster model loading
- Increase system RAM for better performance
- Consider GPU acceleration for large-scale deployment