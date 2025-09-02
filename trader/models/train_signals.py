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