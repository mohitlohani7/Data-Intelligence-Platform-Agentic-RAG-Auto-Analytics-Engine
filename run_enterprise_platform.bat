@echo off
title Data Engine & Intelligence Platform
echo ========================================================
echo 🏢 DATA ENGINE PLATFORM STARTED
echo ========================================================
echo.

echo [1/2] Initializing FastAPI Backend Microservice...
:: Run natively so it inherits your Python environment on port 8080
start /B python -m uvicorn backend.main:app --host 127.0.0.1 --port 8080

echo Waiting 5 seconds for backend to stabilize...
timeout /t 5 /nobreak > NUL

echo [2/2] Booting Premium Streamlit Frontend UI...
streamlit run frontend\app.py

echo.
echo Stop the backend server manually by pressing CTRL+C when finished.
pause
