# trader/models/predict.py
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from ..utils import load_pickle

logger = logging.getLogger(__name__)

class Predictor:
    """Unified prediction interface."""
    
    def __init__(self, model_dir: str = "data/models"):
        self.model_dir = model_dir
        self.models = {}
        self._load_models()
    
    def _load_models(self):
        """Load saved models."""
        import os
        
        # Load fine models
        for horizon in [1, 5, 21]:
            model_path = os.path.join(self.model_dir, f"fine_model_{horizon}d.pkl")
            if os.path.exists(model_path):
                self.models[f'fine_{horizon}'] = load_pickle(model_path)
        
        # Load rerank model
        rerank_path = os.path.join(self.model_dir, "rerank_model.pkl")
        if os.path.exists(rerank_path):
            self.models['rerank'] = load_pickle(rerank_path)
        
        # Load signal model
        signal_path = os.path.join(self.model_dir, "signal_model.pkl")
        if os.path.exists(signal_path):
            self.models['signal'] = load_pickle(signal_path)
    
    def predict_fine(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate fine predictions."""
        predictions = pd.DataFrame(index=features.index)
        
        for horizon in [1, 5, 21]:
            model_key = f'fine_{horizon}'
            if model_key in self.models:
                X = features.drop(['symbol'], axis=1, errors='ignore').values
                predictions[f'pred_{horizon}d'] = self.models[model_key].predict(X)
        
        return predictions
    
    def predict_rerank(self, features: pd.DataFrame) -> np.ndarray:
        """Generate rerank predictions."""
        if 'rerank' not in self.models:
            raise ValueError("Rerank model not loaded")
        
        X = features.drop(['symbol'], axis=1, errors='ignore').values
        return self.models['rerank'].predict(X)
    
    def predict_signals(self, features: np.ndarray) -> np.ndarray:
        """Generate signal predictions."""
        if 'signal' not in self.models:
            raise ValueError("Signal model not loaded")
        
        return self.models['signal'].predict_proba(features)[:, 1]