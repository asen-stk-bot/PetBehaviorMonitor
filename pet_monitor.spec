# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller 打包配置
构建命令（在项目根目录）：
    pyinstaller pet_monitor.spec

生成的可执行文件位于 dist/PetBehaviorMonitor/
"""
from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

# ultralytics 会按需加载多个子模块
hidden = collect_submodules('ultralytics')

a = Analysis(
    ['main.py'],
    pathex=['src'],
    binaries=[],
    datas=[
        ('data', 'data'),
    ],
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'scipy'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PetBehaviorMonitor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='PetBehaviorMonitor',
)