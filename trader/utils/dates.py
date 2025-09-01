# trader/utils/dates.py
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from datetime import datetime, timedelta

def get_trading_days(start: str, end: str, exchange: str = "NYSE") -> pd.DatetimeIndex:
    """Get trading days between start and end dates."""
    return pd.bdate_range(start=start, end=end, freq='B')

def align_dates(df: pd.DataFrame, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    """Align dataframe to trading days, forward filling gaps."""
    df = df.reindex(trading_days)
    df = df.fillna(method='ffill', limit=3)  # Forward fill up to 3 days
    return df

def create_train_test_splits(
    dates: pd.DatetimeIndex, 
    n_splits: int = 5, 
    test_size: int = 252,  # 1 year
    embargo_days: int = 21
) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Create time series splits with embargo period."""
    splits = []
    total_days = len(dates)
    fold_size = (total_days - test_size) // n_splits
    
    for i in range(n_splits):
        train_end_idx = fold_size * (i + 1)
        test_start_idx = train_end_idx + embargo_days
        test_end_idx = min(test_start_idx + test_size, total_days)
        
        if test_end_idx <= test_start_idx:
            break
            
        train_dates = dates[:train_end_idx]
        test_dates = dates[test_start_idx:test_end_idx]
        splits.append((train_dates, test_dates))
    
    return splits

def get_forward_returns(
    prices: pd.DataFrame, 
    horizons: List[int] = [1, 5, 21]
) -> pd.DataFrame:
    """Calculate forward returns for multiple horizons."""
    forward_returns = pd.DataFrame(index=prices.index)
    
    for h in horizons:
        forward_returns[f'ret_{h}d'] = prices.pct_change(h).shift(-h)
    
    return forward_returns