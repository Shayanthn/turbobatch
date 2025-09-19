#!/usr/bin/env python3
"""
Dynamic Batcher for Transformers - Setup Script
A high-performance dynamic batching library for transformer models
"""

import os

from setuptools import find_packages, setup


# Read the README file for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()


# Read requirements from requirements.txt
def read_requirements():
    """Read core requirements from requirements.txt"""
    requirements = []
    try:
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#"):
                    if any(pkg in line for pkg in ["torch", "transformers", "numpy"]):
                        requirements.append(line)
    except FileNotFoundError:
        # Fallback requirements if file doesn't exist
        requirements = ["torch>=1.9.0", "transformers>=4.20.0", "numpy>=1.21.0"]
    return requirements


setup(
    name="turbobatch",
    version="1.0.0",
    author="Shayan Taherkhani",
    author_email="shayan.taherkhani@studio.unibo.it",
    description="🚀 High-Performance Dynamic Batching for Transformer Models",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/Shayanthn/turbobatch",
    packages=find_packages(),
    py_modules=["turbobatch"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "mypy>=0.991",
        ],
        "monitoring": [
            "psutil>=5.9.0",
            "GPUtil>=1.4.0",
        ],
        "examples": [
            "tqdm>=4.64.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "jupyter>=1.0.0",
            "datasets>=2.0.0",
        ],
        "acceleration": [
            "accelerate>=0.20.0",
        ],
        "all": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "mypy>=0.991",
            "psutil>=5.9.0",
            "GPUtil>=1.4.0",
            "tqdm>=4.64.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "jupyter>=1.0.0",
            "datasets>=2.0.0",
            "accelerate>=0.20.0",
        ],
    },
    keywords="transformers, batching, inference, optimization, pytorch, huggingface",
    project_urls={
        "Bug Reports": "https://github.com/Shayanthn/turbobatch/issues",
        "Source": "https://github.com/Shayanthn/turbobatch",
        "Documentation": "https://github.com/Shayanthn/turbobatch#readme",
        "Personal Website": "https://shayantaherkhani.ir",
    },
    include_package_data=True,
    zip_safe=False,
)
