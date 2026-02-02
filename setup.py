"""
Setup script for tokusan package.

Tokusan is a Python library that provides Japanese-friendly LIME
explanations for text classification models.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tokusan",
    version="0.1.0",
    author="Noyu Ritsuji",
    author_email="",
    description="Japanese-friendly LIME explanations for text classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/noyuri2z/tokusan",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Natural Language :: Japanese",
        "Natural Language :: English",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.6.0",
        "scikit-learn>=0.24.0",
    ],
    extras_require={
        "japanese": [
            "sudachipy>=0.6.0",
            "sudachidict_core>=20220729",
        ],
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.900",
        ],
        "viz": [
            "matplotlib>=3.3.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipython>=7.0.0",
        ],
    },
    keywords=[
        "lime",
        "explainability",
        "interpretability",
        "machine learning",
        "nlp",
        "japanese",
        "text classification",
        "xai",
    ],
)
