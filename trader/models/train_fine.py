# trader/models/train_fine.py
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class FineTuner:
    """Train fine selection models."""
    
    def __init__(self, config: dict):
        self.config = config
        self.models = {}
        self.feature_selectors = {}
        
    def train_horizon(
        self,
        features: pd.DataFrame,
        targets: Dict[str, pd.Series],
        horizon: int
    ) -> Tuple[Any, Dict]:
        """Train model for specific horizon."""
        # Prepare data
        X, y = self._prepare_data(features, targets, horizon)
        # Select model type
        model = self._create_model()
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.config['cv']['n_splits'])
        
        scores = []
        for train_idx, val_idx in tscv.split(X):
            # Apply embargo
            embargo = self.config['cv']['embargo_days']
            if embargo > 0:
                train_idx = train_idx[:-embargo]
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train
            model.fit(X_train, y_train)
            
            # Evaluate
            pred = model.predict(X_val)
            scores.append(self._evaluate(y_val, pred))
        
        # Train final model
        model.fit(X, y)
        
        # Average scores
        avg_scores = {
            key: np.mean([s[key] for s in scores])
            for key in scores[0].keys()
        }
        
        self.models[horizon] = model
        
        return model, avg_scores
    
    def _prepare_data(
        self,
        features: pd.DataFrame,
        targets: Dict[str, pd.Series],
        horizon: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data."""
        # Combine targets
        all_targets = []
        for symbol, target_series in targets.items():
            all_targets.append(target_series)
        
        y = pd.concat(all_targets).values
        
        # Remove symbol column if present
        X = features.drop(['symbol'], axis=1, errors='ignore').values
        
        # Remove NaN
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        
        return X[mask], y[mask]
    
    def _create_model(self):
        """Create model based on config."""
        model_config = self.config['models'][0]
        
        if model_config['type'] == 'lasso':
            return Lasso(alpha=model_config.get('alpha', 0.001))
        elif model_config['type'] == 'random_forest':
            return RandomForestRegressor(
                n_estimators=model_config.get('n_estimators', 100),
                max_depth=model_config.get('max_depth', 10),
                random_state=42
            )
        elif model_config['type'] == 'lightgbm':
            return LGBMRegressor(
                n_estimators=model_config.get('n_estimators', 1000),
                learning_rate=model_config.get('learning_rate', 0.01),
                max_depth=model_config.get('max_depth', 5),
                random_state=42,
                verbosity=-1
            )
        else:
            raise ValueError(f"Unknown model type: {model_config['type']}")
    
    def _evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Evaluate predictions."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        from scipy.stats import spearmanr
        
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'ic': spearmanr(y_true, y_pred)[0]
        }