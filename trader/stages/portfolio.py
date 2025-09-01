# trader/stages/portfolio.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class Portfolio:
    """Portfolio management and position sizing."""
    
    def __init__(self, config: dict):
        self.config = config
        self.initial_capital = config.get('initial_capital', 1_000_000)
        self.position_sizing = config.get('position_sizing', 'equal_weight')
        self.max_position_pct = config.get('max_position_pct', 0.10)
        self.max_industry_pct = config.get('max_industry_pct', 0.30)
        self.positions = {}
        self.cash = self.initial_capital
        self.portfolio_value = self.initial_capital
    
    def rebalance(
        self,
        signals: pd.DataFrame,
        current_prices: Dict[str, float],
        reference_data: Optional[pd.DataFrame] = None
    ) -> Dict:
        """Rebalance portfolio based on signals."""
        # Close positions with SELL signals
        for symbol in list(self.positions.keys()):
            if symbol in signals[signals['signal'] == 'SELL']['symbol'].values:
                self._close_position(symbol, current_prices[symbol])
        
        # Open new positions with BUY signals
        buy_signals = signals[signals['signal'] == 'BUY']
        
        # Calculate position sizes
        position_sizes = self._calculate_position_sizes(
            buy_signals,
            current_prices,
            reference_data
        )
        
        # Execute trades
        for symbol, size in position_sizes.items():
            if symbol not in self.positions:
                self._open_position(symbol, size, current_prices[symbol])
        
        # Update portfolio value
        self._update_portfolio_value(current_prices)
        
        return {
            'positions': self.positions.copy(),
            'cash': self.cash,
            'portfolio_value': self.portfolio_value
        }
    
    def _calculate_position_sizes(
        self,
        buy_signals: pd.DataFrame,
        current_prices: Dict[str, float],
        reference_data: Optional[pd.DataFrame]
    ) -> Dict[str, float]:
        """Calculate position sizes based on strategy."""
        available_capital = self.cash
        n_positions = len(buy_signals)
        
        if n_positions == 0:
            return {}
        
        position_sizes = {}
        
        if self.position_sizing == 'equal_weight':
            # Equal weight across all positions
            position_value = min(
                available_capital / n_positions,
                self.portfolio_value * self.max_position_pct
            )
            
            for symbol in buy_signals['symbol']:
                if symbol in current_prices:
                    shares = int(position_value / current_prices[symbol])
                    position_sizes[symbol] = shares
        
        elif self.position_sizing == 'volatility_scaled':
            # Scale by inverse volatility (simplified)
            # Would need historical volatility data
            pass
        
        return position_sizes
    
    def _open_position(self, symbol: str, shares: int, price: float):
        """Open a new position."""
        cost = shares * price
        
        if cost <= self.cash:
            self.positions[symbol] = {
                'shares': shares,
                'entry_price': price,
                'current_price': price,
                'entry_date': pd.Timestamp.now()
            }
            self.cash -= cost
            logger.info(f"Opened position: {symbol} - {shares} shares @ ${price:.2f}")
    
    def _close_position(self, symbol: str, price: float):
        """Close an existing position."""
        if symbol in self.positions:
            position = self.positions[symbol]
            proceeds = position['shares'] * price
            self.cash += proceeds
            
            # Calculate return
            return_pct = (price / position['entry_price'] - 1) * 100
            
            logger.info(f"Closed position: {symbol} - Return: {return_pct:.2f}%")
            del self.positions[symbol]
    
    def _update_portfolio_value(self, current_prices: Dict[str, float]):
        """Update portfolio value."""
        positions_value = 0
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                position['current_price'] = current_prices[symbol]
                positions_value += position['shares'] * current_prices[symbol]
        
        self.portfolio_value = self.cash + positions_value
