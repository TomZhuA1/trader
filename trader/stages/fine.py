# trader/stages/fine.py
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import List, Dict, Tuple, Optional
import logging
from ..features import TechnicalIndicators, create_feature_pipeline
from ..utils import calculate_information_coefficient, calculate_precision_at_k

logger = logging.getLogger(__name__)

class FineSelector:
    """Fine selection stage using ML models."""
    
    def __init__(self, config: dict):
        self.config = config
        self.horizons = config.get('horizons', [1, 5, 21])
        self.rank_weights = config.get('rank_weights', [0.1, 0.3, 0.6])
        self.top_k = config.get('ranking', {}).get('top_k', 2000)
        self.models = {}
        self.pipelines = {}
        self.selected_features = {}
    
    def train(
        self, 
        price_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, any]:
        """Train models for each horizon."""
        results = {}
        
        # Calculate features
        features_df = self._calculate_features(price_data)
        
        for horizon in self.horizons:
            logger.info(f"Training model for {horizon}-day horizon")
            
            # Calculate targets
            targets = self._calculate_targets(price_data, horizon)
            
            # Align features and targets
            X, y = self._align_data(features_df, targets)
            
            # Train model
            model, pipeline, metrics = self._train_horizon_model(X, y, horizon)
            
            self.models[horizon] = model
            self.pipelines[horizon] = pipeline
            results[f'horizon_{horizon}'] = metrics
        
        return results
    
    def predict(
        self, 
        price_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Generate predictions for all horizons."""
        # Calculate features
        features_df = self._calculate_features(price_data)
        
        predictions = pd.DataFrame()
        
        for horizon in self.horizons:
            if horizon not in self.models:
                logger.warning(f"No model found for horizon {horizon}")
                continue
            
            # Transform features
            X = self.pipelines[horizon].transform(features_df.values)
            
            # Predict
            preds = self.models[horizon].predict(X)
            
            predictions[f'pred_{horizon}d'] = preds
        
        # Calculate combined score
        predictions['score'] = sum(
            predictions[f'pred_{h}d'] * w 
            for h, w in zip(self.horizons, self.rank_weights)
        )
        
        # Rank
        predictions['rank'] = predictions['score'].rank(ascending=False)
        
        # Select top K
        top_predictions = predictions.nsmallest(self.top_k, 'rank')
        
        return top_predictions
    
    def _calculate_features(self, price_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate features for all symbols."""
        indicator_calc = TechnicalIndicators(windows=self.config['features']['windows'])
        
        feature_dfs = []
        for symbol, df in price_data.items():
            features = indicator_calc.calculate_all(df)
            features['symbol'] = symbol
            feature_dfs.append(features)
        
        return pd.concat(feature_dfs, ignore_index=True)
    
    def _calculate_targets(
        self, 
        price_data: Dict[str, pd.DataFrame], 
        horizon: int
    ) -> pd.Series:
        """Calculate forward returns for given horizon."""
        targets = []
        
        for symbol, df in price_data.items():
            forward_returns = df['close'].pct_change(horizon).shift(-horizon)
            forward_returns.name = 'target'
            targets.append(forward_returns)
        
        return pd.concat(targets)
    
    def _align_data(
        self, 
        features: pd.DataFrame, 
        targets: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Align features and targets, removing NaN values."""
        # Drop symbol column if present
        if 'symbol' in features.columns:
            features = features.drop('symbol', axis=1)
        
        # Align indices
        common_idx = features.index.intersection(targets.index)
        X = features.loc[common_idx].values
        y = targets.loc[common_idx].values
        
        # Remove NaN
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        
        return X[mask], y[mask]
    
    def _train_horizon_model(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        horizon: int
    ) -> Tuple:
        """Train model for specific horizon."""
        from sklearn.ensemble import RandomForestRegressor
        from lightgbm import LGBMRegressor
        
        # Create pipeline
        pipeline = create_feature_pipeline(
            feature_selection_method=self.config['feature_selection']['method'],
            n_features=self.config['feature_selection']['n_features'],
            winsorize_pct=self.config['preprocessing']['winsorize_pct']
        )
        
        # Fit pipeline
        X_transformed = pipeline.fit_transform(X, y)
        
        # Train model
        model_config = self.config['models'][0]  # Use first model config
        if model_config['type'] == 'lightgbm':
            model = LGBMRegressor(**model_config)
        else:
            model = RandomForestRegressor(**model_config)
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=self.config['cv']['n_splits'])
        
        # Cross-validation
        scores = []
        for train_idx, val_idx in tscv.split(X_transformed):
            X_train, X_val = X_transformed[train_idx], X_transformed[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            
            scores.append({
                'mse': mean_squared_error(y_val, pred),
                'mae': mean_absolute_error(y_val, pred),
                'ic': calculate_information_coefficient(pd.Series(pred), pd.Series(y_val))
            })
        
        # Fit final model on all data
        model.fit(X_transformed, y)
        
        # Average metrics
        avg_metrics = {
            key: np.mean([s[key] for s in scores])
            for key in scores[0].keys()
        }
        
        return model, pipeline, avg_metrics