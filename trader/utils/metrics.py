# trader/utils/metrics.py
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from scipy import stats

def calculate_returns_metrics(returns: pd.Series, periods: int = 252) -> Dict[str, float]:
    """Calculate common return metrics."""
    total_return = (1 + returns).prod() - 1
    n_years = len(returns) / periods
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    
    volatility = returns.std() * np.sqrt(periods)
    sharpe = (returns.mean() * periods) / volatility if volatility > 0 else 0
    
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(periods) if len(downside_returns) > 0 else 0
    sortino = (returns.mean() * periods) / downside_vol if downside_vol > 0 else 0
    
    return {
        'total_return': total_return,
        'cagr': cagr,
        'volatility': volatility,
        'sharpe': sharpe,
        'sortino': sortino
    }

def calculate_drawdown(returns: pd.Series) -> Tuple[pd.Series, float, int]:
    """Calculate drawdown series, max drawdown, and duration."""
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    
    max_dd = drawdown.min()
    
    # Calculate max drawdown duration
    dd_start = None
    max_duration = 0
    current_duration = 0
    
    for i, dd in enumerate(drawdown):
        if dd < 0:
            if dd_start is None:
                dd_start = i
            current_duration = i - dd_start
        else:
            if current_duration > max_duration:
                max_duration = current_duration
            dd_start = None
            current_duration = 0
    
    return drawdown, max_dd, max_duration

def calculate_hit_rate(returns: pd.Series) -> float:
    """Calculate percentage of positive returns."""
    return (returns > 0).mean()

def calculate_avg_win_loss(returns: pd.Series) -> Tuple[float, float, float]:
    """Calculate average win, average loss, and win/loss ratio."""
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
    
    return avg_win, avg_loss, win_loss_ratio

def calculate_calmar_ratio(returns: pd.Series, periods: int = 252) -> float:
    """Calculate Calmar ratio (CAGR / Max DD)."""
    metrics = calculate_returns_metrics(returns, periods)
    _, max_dd, _ = calculate_drawdown(returns)
    
    return metrics['cagr'] / abs(max_dd) if max_dd != 0 else 0

def calculate_information_coefficient(predictions: pd.Series, actuals: pd.Series) -> float:
    """Calculate rank correlation between predictions and actuals."""
    return stats.spearmanr(predictions, actuals)[0]

def calculate_precision_at_k(predictions: pd.Series, actuals: pd.Series, k: int) -> float:
    """Calculate precision at k for top predictions."""
    top_k_preds = predictions.nlargest(k).index
    top_k_actuals = actuals.nlargest(k).index
    
    return len(set(top_k_preds) & set(top_k_actuals)) / k