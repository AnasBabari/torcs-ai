"""
Setup script for TORCS Racing AI package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="torcs-ai",
    version="2.1.0a1",
    author="Anas Babari",
    description="Research tooling for native Windows TORCS racing agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AnasBabari/torcs-ai",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Games/Entertainment :: Simulation",
    ],
    python_requires=">=3.11",
    install_requires=[
        "torch>=2.2,<3",
        "numpy>=1.24,<3",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=5.0",
            "ruff>=0.6",
            "mypy>=1.10",
        ],
        "analysis": [
            "pandas>=2.0,<3",
            "scikit-learn>=1.3,<2",
            "psutil>=5.9,<7",
        ],
        "viz": [
            "matplotlib>=3.7,<4",
            "plotly>=5.18,<7",
            "tqdm>=4.66,<5",
        ],
        "rl": [
            "gymnasium>=0.29,<1.1",
            "stable-baselines3>=2.3,<3",
        ],
        "gpu": [
            # CUDA wheels are selected by the user's PyTorch index/platform;
            # keep the project requirement PEP 508-valid for package builds.
            "torch>=2.2,<3",
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
