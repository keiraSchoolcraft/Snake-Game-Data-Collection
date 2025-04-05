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

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Single-file build - include all binaries, data, and zip files in the EXE itself
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,          # Include binaries in the EXE
    a.zipfiles,          # Include zipfiles in the EXE
    a.datas,             # Include data in the EXE
    [],
    name='SnakeGame',
    debug=True,          # Enable debug for troubleshooting
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None, # Store temporary files in memory
    console=False,        # Set to True for debugging, False for final release
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# Note: The COLLECT section is removed entirely for a single-file build