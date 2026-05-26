@echo off
:: ZENTRA — Windows 11 Setup Script
:: รันไฟล์นี้ครั้งเดียวเพื่อตั้งค่าทั้งหมด
:: ดับเบิลคลิก หรือ รันใน PowerShell: .\setup.bat

title ZENTRA Setup

echo.
echo ========================================================
echo   ZENTRA — Zone Environment Network Thermal Risk Analysis
echo   Setup Script for Windows 11
echo ========================================================
echo.

:: ── ตรวจสอบ Python ────────────────────────────────────────────
echo [1/6] Checking Python version...
python --version 2>NUL
if errorlevel 1 (
    echo [ERROR] Python not found! Please install Python 3.11 from:
    echo         https://www.python.org/downloads/
    pause
    exit /b 1
)

:: ── ตรวจสอบ pip ───────────────────────────────────────────────
echo [2/6] Upgrading pip...
python -m pip install --upgrade pip --quiet

:: ── ตรวจสอบ NVIDIA GPU ────────────────────────────────────────
echo [3/6] Checking GPU...
nvidia-smi >NUL 2>&1
if errorlevel 1 (
    echo      No NVIDIA GPU detected — will use CPU mode
    echo      (Training will be slower, detection still works)
    set GPU_MODE=cpu
) else (
    echo      NVIDIA GPU detected!
    set GPU_MODE=gpu
)

:: ── ติดตั้ง dependencies ───────────────────────────────────────
echo [4/6] Installing dependencies...
pip install -r requirements.txt --quiet

if "%GPU_MODE%"=="gpu" (
    echo      Installing PyTorch with CUDA 12.4 support...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --quiet
) else (
    echo      Installing PyTorch CPU version...
    pip install torch torchvision --quiet
)

:: ── สร้าง .env ────────────────────────────────────────────────
echo [5/6] Setting up .env file...
if not exist .env (
    copy .env.example .env
    echo      Created .env from template
    echo      [ACTION NEEDED] Open .env and fill in your API keys!
) else (
    echo      .env already exists — skipping
)

:: ── สร้าง folder structure ────────────────────────────────────
echo [6/6] Creating folders...
if not exist data\collected\ppe_violations mkdir data\collected\ppe_violations
if not exist data\collected\zone_intrusions mkdir data\collected\zone_intrusions
if not exist data\collected\fall_events     mkdir data\collected\fall_events
if not exist data\collected\normal          mkdir data\collected\normal
if not exist models  mkdir models
if not exist logs    mkdir logs
if not exist reports mkdir reports
if not exist runs    mkdir runs

echo.
echo ========================================================
echo   Setup Complete!
echo ========================================================
echo.
echo   NEXT STEPS:
echo   1. Open .env and fill in:
echo      - ROBOFLOW_API_KEY
echo      - LINE_OA_CHANNEL_ACCESS_TOKEN
echo      - LINE_OA_GROUP_SUPERVISOR / SAFETY / EMERGENCY
echo.
echo   2. Install Docker Desktop from:
echo      https://www.docker.com/products/docker-desktop
echo.
echo   3. Start inference server:
echo      docker compose up inference -d
echo.
echo   4. Run ZENTRA:
echo      python main.py
echo.
echo   5. Get LINE Group IDs:
echo      python get_line_group_id.py
echo.
pause
