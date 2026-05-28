@echo off
echo ============================================================
echo Creating Python Virtual Environment 'env' (Windows)
echo ============================================================
python -m venv env
if %errorlevel% neq 0 (
    echo Error: Failed to create virtual environment. 
    echo Please verify that Python is installed and added to your environment variables (PATH).
    pause
    exit /b %errorlevel%
)

echo.
echo ============================================================
echo Activating Virtual Environment...
echo ============================================================
call env\Scripts\activate

echo.
echo ============================================================
echo Upgrading Pip...
echo ============================================================
python -m pip install --upgrade pip

echo.
echo ============================================================
echo Installing Dependencies from requirements.txt...
echo ============================================================
pip install -r requirements.txt

echo.
echo ============================================================
echo Setup Complete!
echo ============================================================
echo To activate this environment in the future, run: env\Scripts\activate
echo You can run the code now using: python main.py --model llama-3.1-8b
echo.
pause
