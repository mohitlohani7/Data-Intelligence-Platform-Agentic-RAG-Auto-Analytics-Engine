@echo off
echo =========================================
echo   Enterprise RAG Platform - Launcher
echo =========================================

REM Kill any existing python processes
taskkill /im python.exe /f >nul 2>&1
timeout /t 1 /nobreak >nul

REM Start Backend (FastAPI on port 8080)
echo [1/2] Starting Backend API on http://127.0.0.1:8080 ...
start "RAG Backend" cmd /k "cd /d "%~dp0" && .\venv\Scripts\python.exe backend\main.py"

REM Wait for backend to initialize
timeout /t 5 /nobreak >nul

REM Start Frontend (Streamlit on port 8501)
echo [2/2] Starting Streamlit Frontend on http://localhost:8501 ...
start "RAG Frontend" cmd /k "cd /d "%~dp0" && .\venv\Scripts\streamlit.exe run frontend\app.py"

echo.
echo =========================================
echo   Backend:  http://127.0.0.1:8080
echo   Frontend: http://localhost:8501
echo =========================================
pause
