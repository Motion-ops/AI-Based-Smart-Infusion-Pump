# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = [('models', 'models')]
binaries = []
hiddenimports = []
tmp_ret = collect_all('sklearn')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('joblib')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


# [ADDED 2026-10-20] native_app.py imports telemetry_publisher via a
# runtime sys.path.insert(..., "remote_dashboard") trick (see its own
# comment), not a normal top-level package import - PyInstaller's
# static analyzer can't follow that at build time on its own. Adding
# the folder to pathex lets the analyzer actually find and bundle it;
# the explicit hiddenimport is belt-and-suspenders in case the
# try/except ImportError wrapping around that import statement makes
# the analyzer skip it entirely (observed behavior varies by
# PyInstaller version). Without both, the exe would still BUILD fine
# (native_app.py's own try/except falls back to
# RemoteTelemetryPublisher=None) but the remote dashboard feature
# would silently never work in the packaged app - worth getting right
# now rather than re-debugging "it works from source but not the exe"
# later.
hiddenimports += ['telemetry_publisher']

a = Analysis(
    ['native_app.py'],
    pathex=['remote_dashboard'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SmartInfusionAI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SmartInfusionAI',
)
