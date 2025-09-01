# trader/__init__.py
"""
Hierarchical Stock Selection and Trading Pipeline

A production-ready system for multi-stage stock selection and trading.
"""

__version__ = "0.1.0"

from . import data
from . import features
from . import stages
from . import models
from . import utils

__all__ = [
    "data",
    "features", 
    "stages",
    "models",
    "utils",
]

# trader/__main__.py
"""
Entry point for running the trader module as a script.
This allows running: python -m trader
"""

from .trader.cli import app

if __name__ == "__main__":
    app()
