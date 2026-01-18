"""
Setup script for TORCS Racing AI package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="torcs-ai",
    version="2.0.0",
    author="GitHub Copilot Enhanced",
    author_email="copilot@github.com",
    description="Advanced Machine Learning Racing AI for TORCS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/github-copilot/torcs-ai",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Games/Entertainment :: Simulation",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "plotly>=5.0.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
        "psutil>=5.8.0",
        "gym>=0.21.0",
        "stable-baselines3>=1.5.0",
        "opencv-python>=4.5.0",
        "pillow>=8.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=4.0",
            "mypy>=0.900",
        ],
        "gpu": [
            "torch>=2.0.0+cu118",  # CUDA 11.8
        ],
    },
    entry_points={
        "console_scripts": [
            "torcs-ai=torcs_ai.main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)