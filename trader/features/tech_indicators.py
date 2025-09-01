# trader/features/tech_indicators.py
import pandas as pd
import numpy as np
import ta
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """Calculate technical indicators for stock data."""
    
    def __init__(self, windows: List[int] = [5, 10, 14, 20, 50]):
        self.windows = windows
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators."""
        features = pd.DataFrame(index=df.index)
        
        # Ensure we have required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required):
            logger.error(f"Missing required columns. Have: {df.columns.tolist()}")
            return features
        
        # Momentum indicators
        features.update(self._calculate_momentum(df))
        
        # Trend indicators
        features.update(self._calculate_trend(df))
        
        # Volatility indicators
        features.update(self._calculate_volatility(df))
        
        # Volume indicators
        features.update(self._calculate_volume(df))
        
        # Relative indicators
        features.update(self._calculate_relative(df))
        
        return features
    
    def _calculate_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum indicators."""
        features = pd.DataFrame(index=df.index)
        
        for window in self.windows:
            # RSI
            features[f'rsi_{window}'] = ta.momentum.RSIIndicator(
                close=df['close'], window=window
            ).rsi()
            
            # Rate of Change
            features[f'roc_{window}'] = ta.momentum.ROCIndicator(
                close=df['close'], window=window
            ).roc()
            
            # Stochastic
            stoch = ta.momentum.StochasticOscillator(
                high=df['high'], low=df['low'], close=df['close'],
                window=window, smooth_window=3
            )
            features[f'stoch_k_{window}'] = stoch.stoch()
            features[f'stoch_d_{window}'] = stoch.stoch_signal()
        
        # CCI (Commodity Channel Index)
        features['cci'] = ta.trend.CCIIndicator(
            high=df['high'], low=df['low'], close=df['close']
        ).cci()
        
        # Williams %R
        features['williams_r'] = ta.momentum.WilliamsRIndicator(
            high=df['high'], low=df['low'], close=df['close']
        ).williams_r()
        
        # Momentum
        for window in [10, 20]:
            features[f'momentum_{window}'] = df['close'].diff(window)
        
        return features
    
    def _calculate_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend indicators."""
        features = pd.DataFrame(index=df.index)
        
        # Moving averages
        for window in self.windows:
            features[f'sma_{window}'] = ta.trend.SMAIndicator(
                close=df['close'], window=window
            ).sma_indicator()
            
            features[f'ema_{window}'] = ta.trend.EMAIndicator(
                close=df['close'], window=window
            ).ema_indicator()
            
            # Price relative to MA
            features[f'close_to_sma_{window}'] = df['close'] / features[f'sma_{window}'] - 1
            features[f'close_to_ema_{window}'] = df['close'] / features[f'ema_{window}'] - 1
        
        # MACD
        macd = ta.trend.MACD(close=df['close'])
        features['macd'] = macd.macd()
        features['macd_signal'] = macd.macd_signal()
        features['macd_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close=df['close'])
        features['bb_high'] = bb.bollinger_hband()
        features['bb_low'] = bb.bollinger_lband()
        features['bb_mid'] = bb.bollinger_mavg()
        features['bb_width'] = bb.bollinger_wband()
        features['bb_pct'] = bb.bollinger_pband()
        
        # ADX
        adx = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close'])
        features['adx'] = adx.adx()
        features['adx_pos'] = adx.adx_pos()
        features['adx_neg'] = adx.adx_neg()
        
        # Ichimoku
        ichimoku = ta.trend.IchimokuIndicator(high=df['high'], low=df['low'])
        features['ichimoku_a'] = ichimoku.ichimoku_a()
        features['ichimoku_b'] = ichimoku.ichimoku_b()
        features['ichimoku_base'] = ichimoku.ichimoku_base_line()
        features['ichimoku_conv'] = ichimoku.ichimoku_conversion_line()
        
        return features
    
    def _calculate_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility indicators."""
        features = pd.DataFrame(index=df.index)
        
        # ATR
        for window in [14, 20]:
            features[f'atr_{window}'] = ta.volatility.AverageTrueRange(
                high=df['high'], low=df['low'], close=df['close'], window=window
            ).average_true_range()
        
        # Rolling standard deviation
        for window in self.windows:
            features[f'volatility_{window}'] = df['close'].pct_change().rolling(window).std()
        
        # Donchian Channel
        dc = ta.volatility.DonchianChannel(high=df['high'], low=df['low'], close=df['close'])
        features['donchian_high'] = dc.donchian_channel_hband()
        features['donchian_low'] = dc.donchian_channel_lband()
        features['donchian_width'] = dc.donchian_channel_wband()
        features['donchian_pct'] = dc.donchian_channel_pband()
        
        # Keltner Channel
        kc = ta.volatility.KeltnerChannel(high=df['high'], low=df['low'], close=df['close'])
        features['keltner_high'] = kc.keltner_channel_hband()
        features['keltner_low'] = kc.keltner_channel_lband()
        features['keltner_width'] = kc.keltner_channel_wband()
        
        return features
    
    def _calculate_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume indicators."""
        features = pd.DataFrame(index=df.index)
        
        # On Balance Volume
        features['obv'] = ta.volume.OnBalanceVolumeIndicator(
            close=df['close'], volume=df['volume']
        ).on_balance_volume()
        
        # Money Flow Index
        features['mfi'] = ta.volume.MFIIndicator(
            high=df['high'], low=df['low'], close=df['close'], volume=df['volume']
        ).money_flow_index()
        
        # Volume Price Trend
        features['vpt'] = ta.volume.VolumePriceTrendIndicator(
            close=df['close'], volume=df['volume']
        ).volume_price_trend()
        
        # Accumulation/Distribution
        features['ad'] = ta.volume.AccDistIndexIndicator(
            high=df['high'], low=df['low'], close=df['close'], volume=df['volume']
        ).acc_dist_index()
        
        # Volume Rate of Change
        for window in [10, 20]:
            features[f'volume_roc_{window}'] = (
                df['volume'] / df['volume'].shift(window) - 1
            )
        
        # Volume moving averages
        for window in self.windows:
            features[f'volume_sma_{window}'] = df['volume'].rolling(window).mean()
            features[f'volume_ratio_{window}'] = df['volume'] / features[f'volume_sma_{window}']
        
        return features
    
    def _calculate_relative(self, df: pd.DataFrame, spy_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Calculate relative strength and other relative indicators."""
        features = pd.DataFrame(index=df.index)
        
        # Distance from 52-week high/low
        features['dist_52w_high'] = df['close'] / df['close'].rolling(252).max() - 1
        features['dist_52w_low'] = df['close'] / df['close'].rolling(252).min() - 1
        
        # Relative position in range
        for window in [20, 50, 252]:
            roll_max = df['high'].rolling(window).max()
            roll_min = df['low'].rolling(window).min()
            features[f'rel_position_{window}'] = (
                (df['close'] - roll_min) / (roll_max - roll_min)
            ).fillna(0.5)
        
        # Relative strength vs SPY (if provided)
        if spy_df is not None and 'close' in spy_df.columns:
            # Align indices
            spy_close = spy_df['close'].reindex(df.index, method='ffill')
            
            for window in [20, 50]:
                stock_ret = df['close'].pct_change(window)
                spy_ret = spy_close.pct_change(window)
                features[f'rel_strength_spy_{window}'] = stock_ret - spy_ret
                features[f'rel_ratio_spy_{window}'] = (
                    df['close'] / df['close'].shift(window)
                ) / (
                    spy_close / spy_close.shift(window)
                ) - 1
        
        return features