import os
import sys
import subprocess
import logging
import shutil
from unittest.mock import MagicMock

sys.modules['onnxruntime'] = MagicMock()

# 1. IMMEDIATE PATCH: Fix NumPy 2.0 compatibility BEFORE importing anything else
import numpy as np
if not hasattr(np, 'float_'): np.float_ = np.float64
if not hasattr(np, 'int_'): np.int_ = np.int64
if not hasattr(np, 'complex_'): np.complex_ = np.complex128
np.NaN = np.nan



_LOGGER = logging.getLogger(__name__)

def bypass_and_install_chroma():
    """Installs ChromaDB using an executable temporary build directory."""
    try:
        import chromadb
        _LOGGER.debug("ChromaDB is already installed.")
        return
    except ImportError:
        _LOGGER.info("Starting ChromaDB installation...")
        
        # 1. Setup a custom TMPDIR that is EXECUATABLE (not /tmp)
        build_dir = os.path.join(os.path.dirname(__file__), "temp_build")
        os.makedirs(build_dir, exist_ok=True)
        
        # 2. Environment for C++17 Compilation
        custom_env = os.environ.copy()
        custom_env["CXXFLAGS"] = "-std=c++17"
        custom_env["FORCE_CMAKE"] = "1"
        custom_env["TMPDIR"] = build_dir  # <-- This is the fix!
        
        # 3. Setup dummy onnxruntime bypass
        dummy_dir = os.path.join(os.path.dirname(__file__), "dummy_onnx")
        os.makedirs(dummy_dir, exist_ok=True)
        with open(os.path.join(dummy_dir, "setup.py"), "w") as f:
            f.write("from setuptools import setup\nsetup(name='onnxruntime', version='1.99.0')")
        subprocess.run([sys.executable, "-m", "pip", "install", dummy_dir], check=True, env=custom_env)
        
        # 4. Install dependencies
        try:
            # Install hnswlib first
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "chroma-hnswlib==0.7.6"],
                check=True, env=custom_env
            )
            
            # Install ChromaDB
            process = subprocess.Popen(
                [sys.executable, "-m", "pip", "install", "chromadb==0.5.5"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=custom_env
            )
            
            for line in process.stdout:
                if line.strip(): _LOGGER.info(f"[pip] {line.strip()}")
            process.wait()
            
            if process.returncode == 0:
                _LOGGER.info("ChromaDB 0.5.5 installed successfully!")
            else:
                _LOGGER.error("Installation failed.")
                
        finally:
            shutil.rmtree(dummy_dir, ignore_errors=True)
            shutil.rmtree(build_dir, ignore_errors=True)