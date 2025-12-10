"""
Setup script for VSA that only installs the Python/Triton components
without building the CUDA extension. Use this if you don't have an H100
or encounter CUDA compilation errors.

Usage: pip install -e . --config-settings="--build-option=--triton-only"
Or simply: python setup_triton_only.py install
"""

from setuptools import find_packages, setup

# Package metadata
PACKAGE_NAME = "vsa"
VERSION = "0.0.3"
AUTHOR = "Hao AI Lab"
DESCRIPTION = "Video Sparse Attention Kernel Used in FastVideo (Triton-only)"
URL = "https://github.com/hao-ai-lab/FastVideo/tree/main/csrc/attn/video_sparse_attn"

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author=AUTHOR,
    description=DESCRIPTION,
    url=URL,
    packages=["vsa"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
    python_requires='>=3.10',
    install_requires=[
        "torch>=2.5.0",
        "triton>=2.0.0",
    ]
)
