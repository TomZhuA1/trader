# trader/stages/coarse.py
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class CoarseFilter:
    """Coarse filtering stage for initial universe selection."""
    
    def __init__(self, config: dict):
        self.config = config
        self.price_min = config.get('price_min', 5.0)
        self.adv_usd_min = config.get('adv_usd_min', 2_000_000)
        self.adv_percentile = config.get('adv_percentile', 60)
        self.history_days_min = config.get('history_days_min', 756)
        self.exchanges = config.get('exchanges', ['NYSE', 'NASDAQ', 'AMEX'])
        self.exclude_types = config.get('exclude_types', ['ETF', 'ETN', 'PREFERRED', 'OTC'])
        self.max_missing_pct = config.get('max_missing_pct', 0.10)
    
    def filter(
        self, 
        price_data: Dict[str, pd.DataFrame], 
        reference_data: pd.DataFrame
    ) -> List[str]:
        """Apply coarse filters to select initial universe."""
        passed_symbols = []
        
        for symbol in price_data.keys():
            if self._check_symbol(symbol, price_data[symbol], reference_data):
                passed_symbols.append(symbol)
        
        logger.info(f"Coarse filter: {len(passed_symbols)}/{len(price_data)} symbols passed")
        
        return passed_symbols
    
    def _check_symbol(
        self, 
        symbol: str, 
        df: pd.DataFrame, 
        reference_data: pd.DataFrame
    ) -> bool:
        """Check if symbol passes all coarse filters."""
        
        # Check exchange
        if not reference_data.empty:
            symbol_ref = reference_data[reference_data['symbol'] == symbol]
            if not symbol_ref.empty:
                exchange = symbol_ref['exchange'].values[0]
                if exchange not in self.exchanges:
                    return False
                
                # Check if ETF/ETN
                if symbol_ref['is_etf'].values[0] and 'ETF' in self.exclude_types:
                    return False
        
        # Check price minimum
        if df['close'].mean() < self.price_min:
            return False
        
        # Check minimum history
        if len(df) < self.history_days_min:
            return False
        
        # Check average dollar volume
        dollar_volume = (df['close'] * df['volume']).rolling(30).mean()
        if dollar_volume.mean() < self.adv_usd_min:
            return False
        
        # Check missing data
        last_year = df.iloc[-252:] if len(df) >= 252 else df
        missing_pct = last_year['close'].isna().sum() / len(last_year)
        if missing_pct > self.max_missing_pct:
            return False
        
        return True