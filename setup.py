from setuptools import setup, find_packages

setup(
    name="trust_soc",
    version="1.0.0",
    description="Human-Centric Trust Calibration and Adaptive Explainability in AI-Driven Security Operations",
    author="Pierluigi Dell'Acqua, Francesco La Rosa, Massimo Villari",
    author_email="pidellacqua@unime.it",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.22.0",
        "pandas>=1.4.0",
        "pyyaml>=6.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "tqdm>=4.64.0",
    ],
    extras_require={
        "full": [
            "scipy>=1.9.0",
            "scikit-learn>=1.1.0",
            "pytest>=7.0.0",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
    ],
)
