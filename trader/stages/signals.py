# trader/stages/signals.py
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class SignalGenerator:
    """Generate buy/sell signals from predictions."""
    
    def __init__(self, config: dict):
        self.config = config
        self.buy_threshold = config['labeling']['buy_threshold']
        self.sell_threshold = config['labeling']['sell_threshold']
        self.top_n = config['decision']['top_n']
        self.max_hold_days = config['decision']['max_hold_days']
        self.model = None
    
    def train(
        self,
        price_data: Dict[str, pd.DataFrame],
        rerank_predictions: pd.DataFrame
    ) -> Dict:
        """Train signal generation model."""
        # Create features
        X = self._create_features(price_data, rerank_predictions)
        
        # Create labels
        y = self._create_labels(price_data)
        
        # Train classifier
        base_model = LGBMClassifier(**self.config['model']['params'])
        self.model = CalibratedClassifierCV(base_model, cv=3, method='isotonic')
        
        self.model.fit(X, y)
        
        # Calculate metrics
        predictions = self.model.predict_proba(X)[:, 1]  # P(BUY)
        
        metrics = {
            'accuracy': (self.model.predict(X) == y).mean(),
            'buy_precision': ((predictions > 0.5) & (y == 1)).sum() / (predictions > 0.5).sum()
        }
        
        return metrics
    
    def generate_signals(
        self,
        price_data: Dict[str, pd.DataFrame],
        rerank_predictions: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate trading signals."""
        # Create features
        X = self._create_features(price_data, rerank_predictions)
        
        # Get probabilities
        buy_probs = self.model.predict_proba(X)[:, 1]
        
        # Create signals dataframe
        signals = pd.DataFrame({
            'symbol': list(price_data.keys()),
            'buy_prob': buy_probs,
            'signal': 'HOLD'
        })
        
        # Assign signals based on thresholds
        signals.loc[buy_probs > 0.6, 'signal'] = 'BUY'
        signals.loc[buy_probs < 0.3, 'signal'] = 'SELL'
        
        # Select top N by buy probability
        top_buys = signals.nlargest(self.top_n, 'buy_prob')
        
        return top_buys
    
    def _create_features(
        self,
        price_data: Dict[str, pd.DataFrame],
        predictions: pd.DataFrame
    ) -> np.ndarray:
        """Create features for signal generation."""
        features = []
        
        for symbol in price_data.keys():
            feature_vec = []
            
            # Add prediction features
            symbol_pred = predictions[predictions.get('symbol', '') == symbol]
            if not symbol_pred.empty:
                # Add prediction vector
                for col in predictions.columns:
                    if col.startswith('pred_'):
                        feature_vec.append(symbol_pred[col].values[0])
            
            # Add recent price/volume features
            df = price_data[symbol]
            feature_vec.extend([
                df['close'].pct_change(5).iloc[-1],
                df['close'].pct_change(20).iloc[-1],
                df['volume'].iloc[-5:].mean() / df['volume'].iloc[-20:].mean(),
                df['close'].iloc[-1] / df['close'].rolling(50).mean().iloc[-1] - 1
            ])
            
            features.append(feature_vec)
        
        return np.array(features)
    
    def _create_labels(self, price_data: Dict[str, pd.DataFrame]) -> np.ndarray:
        """Create labels based on forward returns."""
        labels = []
        
        for symbol, df in price_data.items():
            forward_return = df['close'].pct_change(self.config['labeling']['horizon_days']).shift(
                -self.config['labeling']['horizon_days']
            ).iloc[-1]
            
            if forward_return >= self.buy_threshold:
                labels.append(1)  # BUY
            elif forward_return <= self.sell_threshold:
                labels.append(-1)  # SELL
            else:
                labels.append(0)  # HOLD
        
        return np.array(labels)