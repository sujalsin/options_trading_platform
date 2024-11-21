from setuptools import setup, find_packages

setup(
    name="optionslib",
    version="0.1.0",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "pybind11>=2.7.0",
        "pytest>=6.2.0",
        "jupyter>=1.0.0",
        "ipython>=7.25.0",
        "statsmodels>=0.12.0",
        "cvxopt>=1.2.0",
        "arch>=4.19.0",
        "yfinance>=0.1.63",
        "ta-lib>=0.4.0",
    ],
    python_requires=">=3.8",
)
