# trader/features/pipelines.py
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from typing import List, Optional, Tuple
import logging
from typing import Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)

def create_feature_pipeline(
    feature_selection_method: str = "lasso",
    n_features: int = 100,
    scale: bool = True,
    winsorize_pct: float = 0.01
) -> Pipeline:
    """Create feature preprocessing pipeline."""
    steps = []
    
    # Imputation
    steps.append(('imputer', SimpleImputer(strategy='median')))
    
    # Winsorization (as custom transformer)
    if winsorize_pct > 0:
        steps.append(('winsorizer', Winsorizer(pct=winsorize_pct)))
    
    # Scaling
    if scale:
        steps.append(('scaler', RobustScaler()))
    
    # Feature selection
    if feature_selection_method == "lasso":
        steps.append(('selector', LassoSelector(n_features=n_features)))
    elif feature_selection_method == "rfe":
        estimator = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        steps.append(('selector', RFE(estimator, n_features_to_select=n_features)))
    elif feature_selection_method == "kbest":
        steps.append(('selector', SelectKBest(f_regression, k=n_features)))
    
    return Pipeline(steps)

class Winsorizer:
    """Custom transformer for winsorization."""
    
    def __init__(self, pct: float = 0.01):
        self.pct = pct
        self.lower_bounds = None
        self.upper_bounds = None
    
    def fit(self, X, y=None):
        """Fit winsorization bounds."""
        self.lower_bounds = np.percentile(X, self.pct * 100, axis=0)
        self.upper_bounds = np.percentile(X, (1 - self.pct) * 100, axis=0)
        return self
    
    def transform(self, X):
        """Apply winsorization."""
        X_transformed = X.copy()
        for i in range(X.shape[1]):
            X_transformed[:, i] = np.clip(
                X_transformed[:, i],
                self.lower_bounds[i],
                self.upper_bounds[i]
            )
        return X_transformed
    
    def fit_transform(self, X, y=None):
        """Fit and transform."""
        return self.fit(X, y).transform(X)

class LassoSelector:
    """Feature selector using Lasso."""
    
    def __init__(self, n_features: int = 100, alpha: float = 0.001):
        self.n_features = n_features
        self.alpha = alpha
        self.selected_features = None
        self.lasso = None
    
    def fit(self, X, y):
        """Fit Lasso and select features."""
        self.lasso = Lasso(alpha=self.alpha, random_state=42)
        self.lasso.fit(X, y)
        
        # Get feature importances (absolute coefficients)
        importances = np.abs(self.lasso.coef_)
        
        # Select top n features
        n_select = min(self.n_features, np.sum(importances > 0))
        top_indices = np.argsort(importances)[-n_select:]
        
        self.selected_features = top_indices
        return self
    
    def transform(self, X):
        """Select features."""
        if self.selected_features is None:
            raise ValueError("Selector not fitted")
        return X[:, self.selected_features]
    
    def fit_transform(self, X, y):
        """Fit and transform."""
        return self.fit(X, y).transform(X)

def create_feature_dataframe(
    price_data: Dict[str, pd.DataFrame],
    indicator_calculator: 'TechnicalIndicators',
    ts_calculator: Optional['TimeSeriesFeatures'] = None,
    tsfresh_calculator: Optional['TSFreshFeatures'] = None
) -> pd.DataFrame:
    """Create feature dataframe from price data."""
    all_features = []
    
    for symbol, df in price_data.items():
        # Calculate technical indicators
        tech_features = indicator_calculator.calculate_all(df)
        tech_features['symbol'] = symbol
        
        # Calculate time series features if provided
        if ts_calculator is not None:
            returns = df['close'].pct_change()
            ts_features = ts_calculator.calculate_all(returns)
            # Broadcast to all rows
            for col in ts_features.columns:
                tech_features[col] = ts_features[col].values[0]
        
        # Calculate tsfresh features if provided
        if tsfresh_calculator is not None:
            returns = df['close'].pct_change()
            tsfresh_features = tsfresh_calculator.extract(returns, symbol)
            if not tsfresh_features.empty:
                # Broadcast to all rows
                for col in tsfresh_features.columns:
                    tech_features[col] = tsfresh_features[col].values[0]
        
        all_features.append(tech_features)
    
    # Combine all features
    feature_df = pd.concat(all_features, ignore_index=True)
    
    return feature_df