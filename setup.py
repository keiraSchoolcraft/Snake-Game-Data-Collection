import sys
import os
from cx_Freeze import setup, Executable

# Dependencies are automatically detected, but it might need fine tuning.
build_exe_options = {
    "packages": [
        "os", 
        "pygame", 
        "random", 
        "time", 
        "numpy", 
        "sys",
        "gspread",
        "google.oauth2",
        "datetime",
        "socket",
        "uuid"
    ],
    "excludes": ["tkinter"],
    "include_files": [
        "config.py",
        "service_account.json",  # Make sure this file exists in your project directory
    ],
    "include_msvcr": True,
    "zip_include_packages": ["*"],
    "zip_exclude_packages": []
}

base = None
if sys.platform == "win32":
    base = "Win32GUI"  # Use this for a windowed application

executables = [
    Executable(
        "main.py",  # Your main Python script
        base=base,
        target_name="SnakeGame.exe",
        icon="snake_icon.ico",  # Optional: add an icon
    )
]

setup(
    name="SnakeGame",
    version="1.0",
    description="16x16 Snake Game with Google Sheets Logging",
    options={"build_exe": build_exe_options},
    executables=executables
)