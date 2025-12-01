"""
Tuberculosis Detection System - Streamlit Web Application
==========================================================
A professional web interface for TB detection from chest X-ray images.

This application implements the same pipeline as pcd-final-project.ipynb:
1. Image Preprocessing (CLAHE, median blur, normalization)
2. Lung Segmentation (Otsu thresholding + morphological operations)
3. Feature Extraction (GLCM + LBP = 107 features)
4. Classification (SVM with RBF kernel)

Reference: pcd-final-project.ipynb
"""

import streamlit as st
import cv2
import numpy as np
import pickle
import os
from PIL import Image
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import threshold_otsu
from skimage.morphology import closing, opening, disk, remove_small_objects

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="TB Detection System",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM STYLING
# =============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #4a69bd;
        margin-bottom: 2rem;
    }
    
    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #4a69bd;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e8e8e8;
    }
    
    .result-card {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .result-normal {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 5px solid #28a745;
    }
    
    .result-tb {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 5px solid #dc3545;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    
    .info-panel {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    
    .warning-panel {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
    
    .footer {
        text-align: center;
        color: #6c757d;
        padding: 2rem 0;
        margin-top: 3rem;
        border-top: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTS (matching notebook: pcd-final-project.ipynb)
# =============================================================================
IMG_SIZE = 512
MODEL_DIR = "trained_models"

# =============================================================================
# MODEL LOADING
# =============================================================================
@st.cache_resource
def load_models():
    """
    Load trained models from pcd-final-project.ipynb
    
    Expected files in trained_models/:
    - svm_model.pkl: SVM classifier with RBF kernel
    - knn_model.pkl: KNN classifier (k=7, distance-weighted)
    - scaler.pkl: StandardScaler for feature normalization
    - pca.pkl: PCA transformer (95% variance retention)
    """
    required_files = ['svm_model.pkl', 'knn_model.pkl', 'scaler.pkl', 'pca.pkl']
    
    for file in required_files:
        if not os.path.exists(os.path.join(MODEL_DIR, file)):
            return None, None, None, None, f"Missing: {file}"
    
    try:
        with open(os.path.join(MODEL_DIR, 'svm_model.pkl'), 'rb') as f:
            svm_model = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'knn_model.pkl'), 'rb') as f:
            knn_model = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'pca.pkl'), 'rb') as f:
            pca = pickle.load(f)
        return svm_model, knn_model, scaler, pca, None
    except Exception as e:
        return None, None, None, None, str(e)

# =============================================================================
# IMAGE PROCESSING FUNCTIONS (exact match to notebook)
# =============================================================================
def preprocess_image(img_bgr):
    """
    Preprocess chest X-ray image.
    Reference: pcd-final-project.ipynb Cell 2
    
    Pipeline:
    1. Resize to 512x512
    2. Convert to grayscale
    3. CLAHE enhancement (clipLimit=2.0, tileGridSize=8x8)
    4. Median blur (kernel=3)
    5. Normalize to [0,1]
    """
    img_resized = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.medianBlur(enhanced, 3)
    norm = denoised.astype(np.float32) / 255.0
    return norm


def segment_lungs(img):
    """
    Segment lung regions using Otsu thresholding.
    Reference: pcd-final-project.ipynb Cell 4
    
    Pipeline:
    1. Otsu thresholding
    2. Morphological closing (disk=5)
    3. Morphological opening (disk=5)
    4. Remove small objects (<300 pixels)
    """
    img_uint8 = (img * 255).astype(np.uint8)
    thresh = threshold_otsu(img_uint8)
    mask = img_uint8 > thresh
    mask = closing(mask, disk(5))
    mask = opening(mask, disk(5))
    mask = remove_small_objects(mask, 300)
    return mask


def extract_glcm_features(img):
    """
    Extract GLCM texture features.
    Reference: pcd-final-project.ipynb Cell 6
    
    Parameters:
    - Distances: [1, 2, 4]
    - Angles: [0, π/4, π/2, 3π/4]
    - Properties: contrast, correlation, energy, homogeneity
    
    Output: 48 features (4 properties × 3 distances × 4 angles)
    """
    img_uint8 = (img * 255).astype(np.uint8)
    distances = [1, 2, 4]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(
        img_uint8,
        distances=distances,
        angles=angles,
        levels=256,
        symmetric=True,
        normed=True
    )
    features = []
    for prop in ['contrast', 'correlation', 'energy', 'homogeneity']:
        vals = graycoprops(glcm, prop)
        features.extend(vals.ravel())
    return np.array(features, dtype=np.float32)


def extract_lbp_features(img, P=16, R=2):
    """
    Extract Local Binary Pattern features.
    Reference: pcd-final-project.ipynb Cell 8
    
    Parameters:
    - P: 16 sampling points
    - R: 2 radius
    - Method: uniform
    
    Output: 59 histogram bins
    """
    img_uint8 = (img * 255).astype(np.uint8)
    lbp = local_binary_pattern(img_uint8, P=P, R=R, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=59, range=(0, 59), density=True)
    return hist.astype(np.float32)


def extract_all_features(lung_img):
    """
    Extract combined GLCM + LBP features.
    Reference: pcd-final-project.ipynb Cell 9
    
    Output: 107 features (48 GLCM + 59 LBP)
    """
    glcm_feat = extract_glcm_features(lung_img)
    lbp_feat = extract_lbp_features(lung_img)
    return np.concatenate([glcm_feat, lbp_feat], axis=0)


def process_uploaded_image(image):
    """
    Complete processing pipeline for uploaded image.
    Reference: pcd-final-project.ipynb Cell 10
    """
    img_array = np.array(image)
    
    if len(img_array.shape) == 3:
        if img_array.shape[2] == 4:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        else:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    
    img_prep = preprocess_image(img_bgr)
    mask = segment_lungs(img_prep)
    lung_only = img_prep * mask
    features = extract_all_features(lung_only)
    
    return features, img_prep, mask, lung_only, img_bgr


def create_overlay(img_bgr, mask, tb_prob):
    """Create TB probability visualization overlay."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    overlay = np.zeros_like(img_rgb)
    overlay[:, :, 0] = mask * tb_prob * 255
    alpha = 0.3
    result = cv2.addWeighted(img_rgb, 1-alpha, overlay.astype(np.uint8), alpha, 0)
    return result

# =============================================================================
# SIDEBAR
# =============================================================================
def render_sidebar():
    """Render sidebar with instructions and model info."""
    st.sidebar.markdown("## 📖 Instructions")
    st.sidebar.markdown("""
    1. Upload a chest X-ray image
    2. Click **Analyze Image**
    3. View detection results and visualizations
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📊 Model Performance")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.sidebar.metric("SVM Accuracy", "92.62%")
        st.sidebar.metric("KNN Accuracy", "92.14%")
    with col2:
        st.sidebar.metric("TB Precision", "95%")
        st.sidebar.metric("Normal Recall", "95%")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🔧 Technical Details")
    st.sidebar.markdown("""
    **Pipeline:**
    - Preprocessing: CLAHE + Median Blur
    - Segmentation: Otsu + Morphology
    - Features: GLCM (48) + LBP (59)
    - Classifier: SVM (RBF kernel)
    - PCA: 95% variance retained
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="text-align: center; font-size: 0.8rem; color: #888;">
        Reference: pcd-final-project.ipynb
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    render_sidebar()
    
    st.markdown('<h1 class="main-header">🫁 Tuberculosis Detection System</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-panel">
        <strong>🎯 About This System</strong><br>
        Advanced TB detection using computer vision and machine learning. 
        Upload a chest X-ray image to get instant analysis with <strong>92.62% accuracy</strong>.
    </div>
    """, unsafe_allow_html=True)
    
    svm_model, knn_model, scaler, pca, error = load_models()
    
    if error:
        st.error(f"❌ Model Loading Error: {error}")
        st.markdown("""
        <div class="warning-panel">
            <strong>⚠️ Setup Required</strong><br>
            Models not found. Please run <code>pcd-final-project.ipynb</code> to train and save the models.
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    
    st.success("✅ Models loaded successfully (SVM + KNN)")
    
    col_upload, col_results = st.columns([1, 1])
    
    with col_upload:
        st.markdown('<p class="section-header">📤 Upload Image</p>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Select a chest X-ray image",
            type=['png', 'jpg', 'jpeg'],
            help="Supported formats: PNG, JPG, JPEG"
        )
        
        model_choice = st.radio(
            "Select Classifier",
            options=["SVM", "KNN"],
            horizontal=True,
            help="SVM: 92.62% accuracy | KNN: 92.14% accuracy"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded X-ray", use_column_width=True)
            
            analyze_btn = st.button("🔬 Analyze Image", type="primary", use_container_width=True)
            
            if analyze_btn:
                with st.spinner("Processing..."):
                    features, img_prep, mask, lung_only, img_bgr = process_uploaded_image(image)
                    
                    if features is not None and len(features) == 107:
                        features_scaled = scaler.transform([features])
                        features_pca = pca.transform(features_scaled)
                        
                        model = svm_model if model_choice == "SVM" else knn_model
                        prediction = model.predict(features_pca)[0]
                        probs = model.predict_proba(features_pca)[0]
                        normal_prob, tb_prob = probs[0], probs[1]
                        
                        st.session_state.update({
                            'prediction': prediction,
                            'normal_prob': normal_prob,
                            'tb_prob': tb_prob,
                            'model_used': model_choice,
                            'img_bgr': img_bgr,
                            'img_prep': img_prep,
                            'mask': mask,
                            'lung_only': lung_only
                        })
                    else:
                        st.error("Feature extraction failed. Please try another image.")
    
    with col_results:
        st.markdown('<p class="section-header">📋 Analysis Results</p>', unsafe_allow_html=True)
        
        if 'prediction' in st.session_state:
            pred = st.session_state.prediction
            normal_prob = st.session_state.normal_prob
            tb_prob = st.session_state.tb_prob
            model_used = st.session_state.get('model_used', 'SVM')
            
            result_class = "result-tb" if pred == 1 else "result-normal"
            result_text = "Tuberculosis Detected" if pred == 1 else "Normal"
            result_icon = "🔴" if pred == 1 else "🟢"
            
            st.markdown(f"""
            <div class="result-card {result_class}">
                <h2 style="margin:0;">{result_icon} {result_text}</h2>
                <p style="margin-top:0.5rem; font-size:0.9rem; color:#666;">Model: {model_used}</p>
                <p style="margin-top:1rem; font-size:1.1rem;">
                    <strong>Normal:</strong> {normal_prob:.1%}<br>
                    <strong>Tuberculosis:</strong> {tb_prob:.1%}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.progress(float(normal_prob), text=f"Normal: {normal_prob:.1%}")
            st.progress(float(tb_prob), text=f"TB: {tb_prob:.1%}")
        else:
            st.info("Upload an image and click **Analyze Image** to see results.")
    
    if 'img_bgr' in st.session_state:
        st.markdown("---")
        st.markdown('<p class="section-header">🎨 Visualizations</p>', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🔍 TB Probability Overlay", "⚙️ Processing Steps"])
        
        with tab1:
            overlay = create_overlay(
                st.session_state.img_bgr,
                st.session_state.mask,
                st.session_state.tb_prob
            )
            
            c1, c2 = st.columns(2)
            with c1:
                st.image(
                    cv2.cvtColor(st.session_state.img_bgr, cv2.COLOR_BGR2RGB),
                    caption="Original",
                    use_column_width=True
                )
            with c2:
                st.image(overlay, caption="TB Probability Overlay", use_column_width=True)
            
            st.info("🔴 Red intensity indicates TB probability in lung regions.")
        
        with tab2:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.image(st.session_state.img_prep, caption="1. Preprocessed", use_column_width=True, clamp=True)
            with c2:
                st.image(st.session_state.mask, caption="2. Lung Mask", use_column_width=True, clamp=True)
            with c3:
                st.image(st.session_state.lung_only, caption="3. Segmented", use_column_width=True, clamp=True)
    
    st.markdown("""
    <div class="footer">
        <strong>🏥 TB Detection System</strong> | SVM Accuracy: 92.62% | KNN Accuracy: 92.14%<br>
        <em>⚠️ For research purposes only. Consult healthcare professionals for diagnosis.</em>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
