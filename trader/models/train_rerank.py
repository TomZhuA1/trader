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

# trader/models/train_signals.py
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class SignalTrainer:
    """Train signal generation models."""
    
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        
    def train(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[Any, Dict]:
        """Train signal classifier."""
        # Handle class imbalance
        base_model = LGBMClassifier(
            **self.config['model']['params'],
            random_state=42,
            verbosity=-1
        )
        
        # Calibrate probabilities
        self.model = CalibratedClassifierCV(
            base_model,
            cv=3,
            method='isotonic'
        )
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        
        for train_idx, val_idx in tscv.split(features):
            X_train, X_val = features[train_idx], features[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]
            
            # Fit model
            temp_model = CalibratedClassifierCV(
                LGBMClassifier(**self.config['model']['params'], random_state=42, verbosity=-1),
                cv=2,
                method='isotonic'
            )
            temp_model.fit(X_train, y_train)
            
            # Evaluate
            pred_proba = temp_model.predict_proba(X_val)[:, 1]
            pred = temp_model.predict(X_val)
            
            scores.append({
                'accuracy': (pred == y_val).mean(),
                'buy_precision': ((pred == 1) & (y_val == 1)).sum() / max((pred == 1).sum(), 1),
                'sell_precision': ((pred == -1) & (y_val == -1)).sum() / max((pred == -1).sum(), 1)
            })
        
        # Train final model
        self.model.fit(features, labels)
        
        # Average scores
        avg_scores = {
            key: np.mean([s[key] for s in scores])
            for key in scores[0].keys()
        }
        
        return self.model, avg_scores