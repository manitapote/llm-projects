"""
search.py
=========
Entry point for BERT grid search over
learning rate, epochs and batch size.
"""

import logging
from src.utils import load_config
from search.grid_search import run_grid_search

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

if __name__ == "__main__":
    config = load_config()
    run_grid_search(config)
