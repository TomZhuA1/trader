# trader/features/timeseries_feats.py
import pandas as pd
import numpy as np
from scipy import stats, signal
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, pacf
from typing import List, Dict, Optional, Tuple
import warnings
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class TimeSeriesFeatures:
    """Calculate advanced time series features."""
    
    def __init__(self, windows: List[int] = [60, 120]):
        self.windows = windows
    
    def calculate_all(self, returns: pd.Series) -> pd.DataFrame:
        """Calculate all time series features."""
        features = {}
        
        for window in self.windows:
            window_features = self._calculate_window_features(returns, window)
            for key, value in window_features.items():
                features[f'{key}_{window}d'] = value
        
        return pd.DataFrame([features])
    
    def _calculate_window_features(self, returns: pd.Series, window: int) -> Dict:
        """Calculate features for a specific window."""
        features = {}
        
        # Use the last 'window' days of returns
        window_returns = returns.iloc[-window:] if len(returns) >= window else returns
        
        if len(window_returns) < 20:  # Not enough data
            return features
        
        # Moments
        features['mean'] = window_returns.mean()
        features['std'] = window_returns.std()
        features['variance'] = window_returns.var()
        features['skewness'] = stats.skew(window_returns.dropna())
        features['kurtosis'] = stats.kurtosis(window_returns.dropna())
        
        # Autocorrelation features
        acf_features = self._calculate_acf_features(window_returns)
        features.update(acf_features)
        
        # Hurst exponent
        features['hurst'] = self._calculate_hurst_exponent(window_returns)
        
        # Entropy
        features['entropy'] = self._calculate_entropy(window_returns)
        
        # Trend features
        trend_features = self._calculate_trend_features(window_returns)
        features.update(trend_features)
        
        # Drawdown features
        dd_features = self._calculate_drawdown_features(window_returns)
        features.update(dd_features)
        
        # Up/down ratio
        features['up_down_ratio'] = (window_returns > 0).sum() / len(window_returns)
        
        return features
    
    def _calculate_acf_features(self, returns: pd.Series, max_lag: int = 20) -> Dict:
        """Calculate autocorrelation-based features."""
        features = {}
        
        try:
            # Calculate ACF
            acf_values = acf(returns.dropna(), nlags=min(max_lag, len(returns)//4))
            
            # Summary statistics of ACF
            features['acf_sum'] = np.sum(np.abs(acf_values[1:]))
            features['acf_mean'] = np.mean(np.abs(acf_values[1:]))
            features['acf_max'] = np.max(np.abs(acf_values[1:]))
            
            # First zero crossing
            zero_crossings = np.where(np.diff(np.sign(acf_values)))[0]
            features['acf_first_zero'] = zero_crossings[0] if len(zero_crossings) > 0 else max_lag
            
            # Ljung-Box test statistic
            lb_test = acorr_ljungbox(returns.dropna(), lags=min(10, len(returns)//4))
            features['ljung_box_stat'] = lb_test.iloc[0]['lb_stat']
            
        except Exception as e:
            logger.warning(f"ACF calculation failed: {e}")
            
        return features
    
    def _calculate_hurst_exponent(self, returns: pd.Series) -> float:
        """Calculate Hurst exponent using R/S analysis."""
        try:
            ts = returns.dropna().values
            if len(ts) < 20:
                return 0.5
            
            lags = range(2, min(100, len(ts)//2))
            tau = []
            
            for lag in lags:
                # Calculate R/S for this lag
                chunks = [ts[i:i+lag] for i in range(0, len(ts)-lag+1, lag)]
                rs_values = []
                
                for chunk in chunks:
                    if len(chunk) < 2:
                        continue
                    mean = np.mean(chunk)
                    deviations = chunk - mean
                    Z = np.cumsum(deviations)
                    R = np.max(Z) - np.min(Z)
                    S = np.std(chunk, ddof=1)
                    if S > 0:
                        rs_values.append(R / S)
                
                if rs_values:
                    tau.append(np.mean(rs_values))
            
            if len(tau) > 10:
                # Fit log-log regression
                log_lags = np.log(list(lags[:len(tau)]))
                log_tau = np.log(tau)
                hurst, _ = np.polyfit(log_lags, log_tau, 1)
                return hurst
            
        except Exception as e:
            logger.warning(f"Hurst calculation failed: {e}")
        
        return 0.5  # Default to random walk
    
    def _calculate_entropy(self, returns: pd.Series) -> float:
        """Calculate Shannon entropy of returns."""
        try:
            # Discretize returns into bins
            n_bins = min(10, len(returns) // 10)
            hist, _ = np.histogram(returns.dropna(), bins=n_bins)
            
            # Calculate probabilities
            probs = hist / np.sum(hist)
            probs = probs[probs > 0]  # Remove zeros
            
            # Shannon entropy
            entropy = -np.sum(probs * np.log2(probs))
            
            return entropy
            
        except Exception as e:
            logger.warning(f"Entropy calculation failed: {e}")
            return 0.0
    
    def _calculate_trend_features(self, returns: pd.Series) -> Dict:
        """Calculate trend-related features."""
        features = {}
        
        try:
            # Cumulative returns
            cum_returns = (1 + returns).cumprod()
            
            # Linear regression on log prices
            if (cum_returns > 0).all():
                log_prices = np.log(cum_returns)
                x = np.arange(len(log_prices))
                
                # Fit linear trend
                slope, intercept = np.polyfit(x, log_prices, 1)
                
                # Calculate R-squared
                y_pred = slope * x + intercept
                ss_res = np.sum((log_prices - y_pred) ** 2)
                ss_tot = np.sum((log_prices - np.mean(log_prices)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                features['trend_slope'] = slope
                features['trend_r2'] = r_squared
            else:
                features['trend_slope'] = 0
                features['trend_r2'] = 0
                
        except Exception as e:
            logger.warning(f"Trend calculation failed: {e}")
            features['trend_slope'] = 0
            features['trend_r2'] = 0
        
        return features
    
    def _calculate_drawdown_features(self, returns: pd.Series) -> Dict:
        """Calculate drawdown-related features."""
        features = {}
        
        try:
            # Cumulative returns
            cum_returns = (1 + returns).cumprod()
            
            # Calculate drawdown
            running_max = cum_returns.expanding().max()
            drawdown = (cum_returns - running_max) / running_max
            
            features['max_drawdown'] = drawdown.min()
            features['avg_drawdown'] = drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0
            features['drawdown_duration'] = (drawdown < 0).sum() / len(drawdown)
            
        except Exception as e:
            logger.warning(f"Drawdown calculation failed: {e}")
            features['max_drawdown'] = 0
            features['avg_drawdown'] = 0
            features['drawdown_duration'] = 0
        
        return features

# trader/features/tsfresh_feats.py
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import MinimalFCParameters, EfficientFCParameters

logger = logging.getLogger(__name__)

class TSFreshFeatures:
    """Extract features using tsfresh library."""
    
    def __init__(self, feature_set: str = "minimal", max_features: int = 50):
        self.feature_set = feature_set
        self.max_features = max_features
        
        # Choose parameter set
        if feature_set == "minimal":
            self.fc_parameters = MinimalFCParameters()
        elif feature_set == "efficient":
            self.fc_parameters = EfficientFCParameters()
        else:
            self.fc_parameters = MinimalFCParameters()
    
    def extract(self, returns: pd.Series, symbol: str) -> pd.DataFrame:
        """Extract tsfresh features from returns series."""
        try:
            # Prepare data for tsfresh
            df = pd.DataFrame({
                'id': symbol,
                'time': range(len(returns)),
                'value': returns.values
            })
            
            # Extract features
            features = extract_features(
                df, 
                column_id='id', 
                column_sort='time',
                default_fc_parameters=self.fc_parameters,
                disable_progressbar=True,
                n_jobs=1
            )
            
            # Impute missing values
            impute(features)
            
            # Select top features if too many
            if len(features.columns) > self.max_features:
                # Use variance as a simple selection criterion
                variances = features.var()
                top_features = variances.nlargest(self.max_features).index
                features = features[top_features]
            
            return features
            
        except Exception as e:
            logger.warning(f"TSFresh extraction failed for {symbol}: {e}")
            return pd.DataFrame()