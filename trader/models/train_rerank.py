# trader/models/train_rerank.py
import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
from typing import Dict, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class RerankTrainer:
    """Train reranking models."""
    
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        
    def train(
        self,
        features: pd.DataFrame,
        targets: np.ndarray
    ) -> Tuple[Any, Dict]:
        """Train reranking model."""
        # Prepare data
        X = features.drop(['symbol'], axis=1, errors='ignore').values
        y = targets
        
        # Remove NaN
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y).any(axis=1))
        X, y = X[mask], y[mask]
        
        # Create model
        base_estimator = LGBMRegressor(
            **self.config['model']['params'],
            random_state=42,
            verbosity=-1
        )
        
        if self.config['model']['type'] == 'multi_output':
            self.model = MultiOutputRegressor(base_estimator)
            self.model.fit(X, y)
        else:
            # Train independent models
            self.model = []
            for i in range(y.shape[1]):
                model = LGBMRegressor(
                    **self.config['model']['params'],
                    random_state=42,
                    verbosity=-1
                )
                model.fit(X, y[:, i])
                self.model.append(model)
        
        # Evaluate
        predictions = self.predict(X)
        metrics = self._evaluate(y, predictions)
        
        return self.model, metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        if isinstance(self.model, MultiOutputRegressor):
            return self.model.predict(X)
        else:
            predictions = []
            for model in self.model:
                predictions.append(model.predict(X))
            return np.column_stack(predictions)
    
    def _evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Evaluate predictions."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        return {
            'mse': mean_squared_error(y_true.flatten(), y_pred.flatten()),
            'mae': mean_absolute_error(y_true.flatten(), y_pred.flatten()),
            'correlation': np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]
        }
