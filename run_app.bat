@echo off
echo ====================================
echo   TB Detection Streamlit App
echo ====================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://python.org
    pause
    exit /b 1
)

echo Python detected successfully!
echo.

REM Check if streamlit is installed
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo Streamlit not found. Installing dependencies...
    echo.
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
)

REM Check if trained models exist
if not exist "trained_models\svm_model.pkl" (
    echo.
    echo WARNING: Trained models not found!
    echo Please run the Jupyter notebook first to train the model.
    echo File: pcd-final-project.ipynb
    echo.
    set /p choice="Continue anyway? (y/n): "
    if /i "%choice%" neq "y" exit /b 1
)

echo.
echo Starting TB Detection Web Application...
echo The app will open in your default browser at http://localhost:8501
echo.
echo Press Ctrl+C to stop the application
echo ====================================
echo.

streamlit run app.py

pause