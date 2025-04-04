import os
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

# Collect all necessary data files, including the service account JSON
a = Analysis(
    ['main.py'],
    pathex=[os.path.abspath('.')],
    binaries=[],
    datas=[
        ('service_account.json', '.'),  # Keep the service account file at the root level
        ('config.py', '.'),
        ('snake_game.py', '.'),
        ('sheets_logger.py', '.')
    ],
    hiddenimports=[
        'gspread', 
        'google.oauth2.service_account', 
        'google.auth', 
        'google.auth.transport.requests',
        'numpy',
        'pygame',
        'google.auth.crypt',
        'google.auth.crypt.base',
        'google.auth.crypt.rsa',
        'google.auth.transport.urllib3',
        'google.auth.transport.requests',
        'google.auth._default',
        'google.auth._service_account_info',
        'requests'
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False
)

# Don't manually extend datas - this was likely causing the error

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SnakeGame',
    debug=True,  # Enable debug for troubleshooting
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True  # Set to True for debugging to see output
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SnakeGame'
)