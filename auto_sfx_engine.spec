# -*- mode: python ; coding: utf-8 -*-
import sys
import os
from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None

datas = []
binaries = []
hiddenimports = [
    'soundfile',
    '_soundfile_data',
    'scipy',
    'scipy.fft',
    'scipy.signal',
    'scipy.special',
    'scipy.special.cython_special',
    'scipy.spatial.transform._rotation_groups',
    'tokenizers',
    'tokenizers.models',
    'tokenizers.decoders',
    'tokenizers.processors',
    'tokenizers.normalizers',
    'tokenizers.pre_tokenizers',
    'librosa',
    'cv2',
    'numpy',
    'onnxruntime',
]

for pkg in ['onnxruntime', 'tokenizers', 'librosa', '_soundfile_data']:
    try:
        pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
        datas += pkg_datas
        binaries += pkg_binaries
        hiddenimports += pkg_hidden
    except Exception as e:
        print(f"Hook note for {pkg}: {e}")

# Deduplicate hidden imports
hiddenimports = sorted(list(set(hiddenimports)))

script_path = os.path.abspath('engine/auto_sfx_engine.py')
if not os.path.exists(script_path):
    script_path = os.path.abspath('auto_sfx_engine.py')

a = Analysis(
    [script_path],
    pathex=['.', 'engine'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'torch',
        'torchvision',
        'torchaudio',
        'notebook',
        'IPython',
        'PIL._tkinter_finder',
        'pytest',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Strip heavy CUDA / TensorRT dependencies that bloat the binary by 1.6 GB
cuda_bloat = [
    'cublas', 'cufft', 'curand', 'cusolver', 'cusparse', 'cudnn', 'nvrtc',
    'onnxruntime_providers_cuda', 'onnxruntime_providers_tensorrt',
    'torch\\lib', 'torch/lib'
]
a.binaries = [
    (dest, src, kind) for dest, src, kind in a.binaries
    if not any(pat in dest.lower() or pat in src.lower() for pat in cuda_bloat)
]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

output_name = os.environ.get('SFX_EXE_NAME', 'auto_sfx_engine')

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name=output_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
