@echo off
echo Setting up PySpark project environment...

REM Check Python installation
python --version
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python 3.10.x is required.
    pause
    exit /b
)

REM Check Java installation
java -version
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: Java 11 or Java 8 is required for PySpark.
    pause
    exit /b
)

REM Create virtual environment
python -m venv venv

REM Activate virtual environment
call venv\Scripts\activate

REM Upgrade pip
python -m pip install --upgrade pip

REM Install project dependencies
pip install pyspark==3.5.1
pip install numpy==1.26.4
pip install matplotlib==3.8.2
pip install seaborn==0.13.0
pip install scikit-learn==1.3.2

echo Environment setup completed
pause
