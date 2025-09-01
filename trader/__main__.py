# trader/__main__.py
"""
Entry point for running the trader module as a script.
This allows running: python -m trader
"""

from .cli import app

if __name__ == "__main__":
    app()