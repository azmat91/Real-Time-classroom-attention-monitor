@echo off
echo ========================================
echo SCB-05 Classroom Analyzer Setup
echo ========================================
echo.

echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing requirements...
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics
pip install deep-sort-realtime
pip install opencv-python pillow numpy pyyaml pandas matplotlib tqdm

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo To run the application:
echo 1. Place your model files in:
echo    - models/train/weights/best.pt
echo    - models/train2/weights/best.pt
echo 2. Run: run.bat
echo.
pause