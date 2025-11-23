# TB Detection Streamlit App Launcher (PowerShell)

Write-Host "====================================" -ForegroundColor Cyan
Write-Host "   TB Detection Streamlit App      " -ForegroundColor Cyan  
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python detected: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python from https://python.org" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""

# Check if streamlit is installed
try {
    python -c "import streamlit" 2>$null
    Write-Host "Streamlit found!" -ForegroundColor Green
} catch {
    Write-Host "Streamlit not found. Installing dependencies..." -ForegroundColor Yellow
    Write-Host ""
    pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to install dependencies" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Check if trained models exist
if (-not (Test-Path "trained_models\svm_model.pkl")) {
    Write-Host ""
    Write-Host "WARNING: Trained models not found!" -ForegroundColor Yellow
    Write-Host "Please run the Jupyter notebook first to train the model." -ForegroundColor Yellow
    Write-Host "File: pcd-final-project.ipynb" -ForegroundColor Yellow
    Write-Host ""
    $choice = Read-Host "Continue anyway? (y/n)"
    if ($choice -ne "y" -and $choice -ne "Y") {
        exit 1
    }
}

Write-Host ""
Write-Host "Starting TB Detection Web Application..." -ForegroundColor Green
Write-Host "The app will open in your default browser at http://localhost:8501" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the application" -ForegroundColor Yellow
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""

# Start Streamlit app
streamlit run app.py

Read-Host "Press Enter to exit"