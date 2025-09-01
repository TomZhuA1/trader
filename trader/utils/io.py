# trader/utils/io.py
import os
import pickle
import json
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)

def ensure_dir(path: str) -> Path:
    """Ensure directory exists."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_parquet(df: pd.DataFrame, path: str, **kwargs) -> None:
    """Save DataFrame to parquet."""
    ensure_dir(os.path.dirname(path))
    df.to_parquet(path, **kwargs)
    logger.info(f"Saved {len(df)} rows to {path}")

def load_parquet(path: str) -> pd.DataFrame:
    """Load DataFrame from parquet."""
    return pd.read_parquet(path)

def save_pickle(obj: Any, path: str) -> None:
    """Save object to pickle."""
    ensure_dir(os.path.dirname(path))
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    logger.info(f"Saved pickle to {path}")

def load_pickle(path: str) -> Any:
    """Load object from pickle."""
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_json(data: Dict, path: str) -> None:
    """Save dictionary to JSON."""
    ensure_dir(os.path.dirname(path))
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Saved JSON to {path}")

def load_json(path: str) -> Dict:
    """Load dictionary from JSON."""
    with open(path, 'r') as f:
        return json.load(f)

def cache_data(cache_dir: str, key: str, compute_fn, force: bool = False) -> Any:
    """Cache computation results."""
    cache_path = os.path.join(cache_dir, f"{key}.pkl")
    
    if not force and os.path.exists(cache_path):
        logger.info(f"Loading cached {key}")
        return load_pickle(cache_path)
    
    logger.info(f"Computing {key}")
    result = compute_fn()
    save_pickle(result, cache_path)
    return result