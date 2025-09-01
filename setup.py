# setup.py (needed for pip install -e .)
from setuptools import setup, find_packages

setup(
    name="trader",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.11",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "lightgbm>=4.0.0",
        "xgboost>=2.0.0",
        "statsmodels>=0.14.0",
        "scipy>=1.11.0",
        "ta>=0.10.2",
        "tsfresh>=0.20.0",
        "yfinance>=0.2.28",
        "requests>=2.31.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "typer>=0.9.0",
        "rich>=13.5.0",
        "mlflow>=2.8.0",
        "joblib>=1.3.0",
        "pyarrow>=13.0.0",
        "tqdm>=4.66.0",
    ],
    entry_points={
        "console_scripts": [
            "trader=trader.cli:app",
        ],
    },
)