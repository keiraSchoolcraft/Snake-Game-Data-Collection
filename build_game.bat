@echo off
echo Snake Game Packaging Tool
echo ------------------------

REM Check if Python is installed
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH. Please install Python first.
    pause
    exit /b 1
)

REM Install required packages if needed
echo Installing required packages...
pip install -r requirements.txt > nul 2>&1

REM Run cx_Freeze
echo Building executable...
python setup.py build

if %errorlevel% neq 0 (
    echo Build failed. Please check the error message above.
    pause
    exit /b 1
)

echo.
echo Build successful! Your executable is in the build directory.
echo.

REM Optional: Create a ZIP file
set /p choice=Do you want to create a ZIP file for distribution? (Y/N): 
if /i "%choice%"=="Y" (
    echo Creating ZIP file...
    powershell -Command "& {Compress-Archive -Path build -DestinationPath SnakeGame.zip -Force}"
    echo ZIP file created: SnakeGame.zip
)

echo.
echo You can now distribute the contents of the build folder.
echo Users can run your game without having Python installed.
echo.

pause