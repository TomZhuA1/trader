import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
from typing import Dict, List, Optional
import logging
from ..features import TimeSeriesFeatures
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class Reranker:
    """Reranking stage with multi-horizon predictions."""
    
    def __init__(self, config: dict):
        self.config = config
        self.n_days = config.get('n_days', 30)
        self.aggregation_method = config.get('aggregation', {}).get('method', 'cum_return')
        self.top_k = config.get('aggregation', {}).get('top_k', 500)
        self.model = None
        self.ts_calculator = TimeSeriesFeatures(windows=config['features']['windows'])
    
    def train(
        self, 
        price_data: Dict[str, pd.DataFrame],
        fine_predictions: pd.DataFrame
    ) -> Dict:
        """Train reranking model."""
        # Calculate additional features
        features = self._calculate_features(price_data, fine_predictions)
        
        # Calculate 30-day forward returns
        targets = self._calculate_multi_horizon_targets(price_data)
        
        # Align data
        X, y = self._align_data(features, targets)
        
        # Train model
        base_estimator = LGBMRegressor(**self.config['model']['params'])
        
        if self.config['model']['type'] == 'multi_output':
            self.model = MultiOutputRegressor(base_estimator)
        else:
            # Train independent models
            self.model = []
            for i in range(self.n_days):
                model = LGBMRegressor(**self.config['model']['params'])
                model.fit(X, y[:, i])
                self.model.append(model)
        
        if isinstance(self.model, MultiOutputRegressor):
            self.model.fit(X, y)
        
        # Calculate training metrics
        predictions = self.predict_raw(features)
        metrics = self._calculate_metrics(predictions, targets)
        
        return metrics
    
    def predict(
        self, 
        price_data: Dict[str, pd.DataFrame],
        fine_predictions: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate reranked predictions."""
        # Calculate features
        features = self._calculate_features(price_data, fine_predictions)
        
        # Get raw predictions
        predictions = self.predict_raw(features)
        
        # Aggregate predictions
        aggregated = self._aggregate_predictions(predictions)
        
        # Rank and select top K
        aggregated['rank'] = aggregated['score'].rank(ascending=False)
        top_predictions = aggregated.nsmallest(self.top_k, 'rank')
        
        return top_predictions
    
    def predict_raw(self, features: pd.DataFrame) -> np.ndarray:
        """Get raw multi-horizon predictions."""
        if self.model is None:
            raise ValueError("Model not trained")
        
        X = features.values
        
        if isinstance(self.model, MultiOutputRegressor):
            return self.model.predict(X)
        else:
            predictions = []
            for model in self.model:
                predictions.append(model.predict(X))
            return np.column_stack(predictions)
    
    def _calculate_features(
        self, 
        price_data: Dict[str, pd.DataFrame],
        fine_predictions: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate reranking features."""
        features = []
        
        for symbol, df in price_data.items():
            # Time series features
            returns = df['close'].pct_change()
            ts_features = self.ts_calculator.calculate_all(returns)
            
            # Add fine predictions if available
            symbol_preds = fine_predictions[fine_predictions.get('symbol', '') == symbol]
            if not symbol_preds.empty:
                for col in ['pred_1d', 'pred_5d', 'pred_21d']:
                    if col in symbol_preds.columns:
                        ts_features[col] = symbol_preds[col].values[0]
            
            ts_features['symbol'] = symbol
            features.append(ts_features)
        
        return pd.concat(features, ignore_index=True)
    
    def _calculate_multi_horizon_targets(
        self, 
        price_data: Dict[str, pd.DataFrame]
    ) -> np.ndarray:
        """Calculate targets for each day in the horizon."""
        all_targets = []
        
        for symbol, df in price_data.items():
            targets = []
            for h in range(1, self.n_days + 1):
                forward_return = df['close'].pct_change(h).shift(-h)
                targets.append(forward_return.iloc[-1] if len(forward_return) > 0 else 0)
            all_targets.append(targets)
        
        return np.array(all_targets)
    
    def _aggregate_predictions(self, predictions: np.ndarray) -> pd.DataFrame:
        """Aggregate multi-horizon predictions into scores."""
        df = pd.DataFrame()
        
        if self.aggregation_method == 'cum_return':
            # Compound daily returns
            df['score'] = np.prod(1 + predictions, axis=1) - 1
        elif self.aggregation_method == 'sharpe_like':
            # Sharpe-like ratio
            mean_return = np.mean(predictions, axis=1)
            std_return = np.std(predictions, axis=1)
            df['score'] = mean_return / (std_return + 1e-6)
        else:
            # Simple average
            df['score'] = np.mean(predictions, axis=1)
        
        return df
    
    def _align_data(self, features: pd.DataFrame, targets: np.ndarray) -> Tuple:
        """Align features and targets."""
        if 'symbol' in features.columns:
            features = features.drop('symbol', axis=1)
        
        X = features.values
        y = targets
        
        # Remove NaN
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y).any(axis=1))
        
        return X[mask], y[mask]
    
    def _calculate_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict:
        """Calculate evaluation metrics."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        metrics = {
            'mse': mean_squared_error(targets.flatten(), predictions.flatten()),
            'mae': mean_absolute_error(targets.flatten(), predictions.flatten()),
            'correlation': np.corrcoef(targets.flatten(), predictions.flatten())[0, 1]
        }
        
        return metrics