# Hierarchical Stock Selection and Trading Pipeline

A production-ready, modular Python system implementing a hierarchical, coarse-to-fine stock selection and trading pipeline for U.S. equities.

## 🎯 Overview

This system implements a sophisticated multi-stage stock selection pipeline:

1. **Data Layer** - Fetches and caches historical price data
2. **Coarse Filtering** - Rule-based initial universe selection
3. **Fine Selection** - ML-based screening with 3 horizons (1D, 5D, 21D)
4. **Reranking** - Multi-horizon (30-day) predictions and aggregation
5. **Signal Generation** - Supervised buy/sell classification
6. **Portfolio & Backtesting** - Strategy execution and performance analysis

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd trader

# Install dependencies
pip install -r requirements.txt

# Or using Poetry
poetry install
```

### Docker Setup

```bash
# Build the Docker image
docker build -t trader .

# Run the container
docker run -it trader python -m trader.cli --help
```

## 📊 Usage

### Complete Pipeline Execution

```bash
# Run the entire pipeline
python -m trader.cli run-all --start 2015-01-01 --end 2025-08-01
```

### Stage-by-Stage Execution

```bash
# Stage 1: Fetch data
python -m trader.cli fetch --start 2015-01-01 --end 2025-08-01

# Stage 2: Coarse filtering
python -m trader.cli coarse

# Stage 3: Train fine selection models
python -m trader.cli fine-train

# Stage 4: Generate fine predictions
python -m trader.cli fine-predict

# Stage 5: Train reranking model
python -m trader.cli rerank-train

# Stage 6: Generate reranked predictions
python -m trader.cli rerank-predict

# Stage 7: Train signal model
python -m trader.cli signal-train

# Stage 8: Generate signals
python -m trader.cli signal-predict

# Stage 9: Run backtest
python -m trader.cli backtest --start 2020-01-01 --end 2025-08-01
```

## 🔧 Configuration

All parameters are configurable via YAML files in the `configs/` directory:

- `base.yaml` - General settings (data paths, logging, reproducibility)
- `coarse.yaml` - Coarse filtering parameters
- `fine.yaml` - Fine selection model configuration
- `rerank.yaml` - Reranking model settings
- `signals.yaml` - Signal generation parameters
- `backtest.yaml` - Backtesting and portfolio settings

### Key Configuration Parameters

```yaml
# configs/coarse.yaml
filters:
  price_min: 5.0              # Minimum stock price
  adv_usd_min: 2_000_000      # Minimum average dollar volume
  history_days_min: 756       # Minimum trading history (~3 years)
  exchanges: ["NYSE", "NASDAQ", "AMEX"]
  max_missing_pct: 0.10       # Maximum missing data tolerance

# configs/fine.yaml
horizons: [1, 5, 21]          # Prediction horizons (days)
rank_weights: [0.1, 0.3, 0.6] # Weights for combining horizons

# configs/backtest.yaml
costs:
  commission_bps: 2           # Commission in basis points
  slippage_bps: 8            # Slippage in basis points
```

## 📈 Features

### Technical Indicators
- **Momentum**: RSI, ROC, Stochastic, CCI, Williams %R
- **Trend**: SMA/EMA, MACD, Bollinger Bands, ADX, Ichimoku
- **Volatility**: ATR, Rolling STD, Donchian Channel
- **Volume**: OBV, MFI, Volume ROC, Accumulation/Distribution

### Time Series Features
- Statistical moments (variance, skewness, kurtosis)
- Autocorrelation features
- Hurst exponent
- Shannon entropy
- Trend analysis
- Drawdown metrics

### Machine Learning Models
- Lasso Regression
- Random Forest
- LightGBM
- Multi-output regression for 30-day forecasting
- Calibrated probability classifiers for signals

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=trader --cov-report=html

# Run specific test module
pytest tests/test_data_provider.py

# Run leakage checks
pytest tests/test_leakage_checks.py -v
```

## 📊 Performance Metrics

The system tracks comprehensive performance metrics:

- **Returns**: Total Return, CAGR, Volatility
- **Risk-Adjusted**: Sharpe Ratio, Sortino Ratio, Calmar Ratio
- **Drawdown**: Maximum Drawdown, Duration
- **Trading**: Hit Rate, Win/Loss Ratio, Turnover
- **ML Metrics**: MSE, MAE, Information Coefficient, Precision@K

## 🏗️ Architecture

```
Pipeline Flow:
┌─────────┐     ┌────────┐     ┌──────┐     ┌────────┐     ┌─────────┐
│  Data   │ --> │ Coarse │ --> │ Fine │ --> │ Rerank │ --> │ Signals │
│ Fetcher │     │ Filter │     │  ML  │     │   ML   │     │   ML    │
└─────────┘     └────────┘     └──────┘     └────────┘     └─────────┘
                                                                  |
                                                                  v
                                                           ┌──────────┐
                                                           │ Backtest │
                                                           └──────────┘
```

## 📝 Data Requirements

- **Historical Data**: Minimum 3 years of daily OHLCV data
- **Universe**: NYSE, NASDAQ, AMEX listed equities
- **Adjustments**: Split and dividend adjusted prices
- **Reference Data**: Exchange, sector, market cap information

## 🔒 Leakage Prevention

The system includes comprehensive leakage checks:

- Forward return shifting verification
- Train/test embargo enforcement
- Feature/target date alignment validation
- Scaler fitting on training data only
- Point-in-time data access

## 🚦 Production Considerations

- **Caching**: Parquet file caching for historical data
- **Rate Limiting**: Configurable delays for API calls
- **Parallel Processing**: Multi-core support via joblib
- **Reproducibility**: Fixed random seeds, deterministic operations
- **Monitoring**: Comprehensive logging at all stages

## 📚 Notebooks

Two Jupyter notebooks are included for analysis:

1. `01_explore_data.ipynb` - Data exploration and visualization
2. `02_model_diagnostics.ipynb` - Model performance analysis

## 🤝 Contributing

Please ensure all tests pass and maintain >80% code coverage for core logic:

```bash
# Format code
black trader/
isort trader/

# Check style
flake8 trader/

# Run tests
pytest --cov=trader
```

## 📄 License

[Your License Here]

## ⚠️ Disclaimer

This system is for educational and research purposes only. Past performance does not guarantee future results. Always perform your own due diligence before making investment decisions.

---

# Test File Example: tests/test_data_provider.py

```python
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from trader.data import YFinanceProvider, UniverseManager

class TestYFinanceProvider:
    """Test YFinance data provider."""
    
    @pytest.fixture
    def provider(self, tmp_path):
        """Create provider instance."""
        return YFinanceProvider(cache_dir=str(tmp_path))
    
    def test_list_symbols(self, provider):
        """Test symbol listing."""
        symbols = provider.list_symbols()
        
        assert isinstance(symbols, list)
        assert len(symbols) > 0
        assert 'AAPL' in symbols
        assert 'MSFT' in symbols
    
    def test_get_history(self, provider):
        """Test historical data fetching."""
        symbols = ['AAPL', 'MSFT']
        start = '2024-01-01'
        end = '2024-12-31'
        
        data = provider.get_history(symbols, start, end)
        
        assert isinstance(data, dict)
        assert 'AAPL' in data
        assert 'MSFT' in data
        
        # Check data structure
        aapl_data = data['AAPL']
        assert isinstance(aapl_data, pd.DataFrame)
        assert 'open' in aapl_data.columns
        assert 'high' in aapl_data.columns
        assert 'low' in aapl_data.columns
        assert 'close' in aapl_data.columns
        assert 'volume' in aapl_data.columns
    
    def test_caching(self, provider, tmp_path):
        """Test data caching."""
        symbols = ['AAPL']
        start = '2024-01-01'
        end = '2024-01-31'
        
        # First fetch
        data1 = provider.get_history(symbols, start, end)
        
        # Check cache file exists
        cache_files = list(tmp_path.glob("*.parquet"))
        assert len(cache_files) > 0
        
        # Second fetch should use cache
        data2 = provider.get_history(symbols, start, end)
        
        # Data should be identical
        pd.testing.assert_frame_equal(data1['AAPL'], data2['AAPL'])

class TestUniverseManager:
    """Test universe management."""
    
    @pytest.fixture
    def reference_data(self):
        """Create sample reference data."""
        return pd.DataFrame({
            'symbol': ['AAPL', 'MSFT', 'SPY', 'QQQ', 'PENNY'],
            'exchange': ['NASDAQ', 'NASDAQ', 'NYSE', 'NASDAQ', 'OTC'],
            'sector': ['Technology', 'Technology', 'ETF', 'ETF', 'Unknown'],
            'is_etf': [False, False, True, True, False],
            'market_cap': [3e12, 2.8e12, 5e11, 4e11, 1e6]
        })
    
    @pytest.fixture
    def universe(self, reference_data):
        """Create universe manager."""
        return UniverseManager(reference_data)
    
    def test_filter_by_exchange(self, universe):
        """Test exchange filtering."""
        symbols = universe.filter_symbols(exchanges=['NASDAQ'])
        
        assert 'AAPL' in symbols
        assert 'MSFT' in symbols
        assert 'SPY' not in symbols
        assert 'PENNY' not in symbols
    
    def test_exclude_etfs(self, universe):
        """Test ETF exclusion."""
        symbols = universe.filter_symbols(exclude_types=['ETF'])
        
        assert 'AAPL' in symbols
        assert 'MSFT' in symbols
        assert 'SPY' not in symbols
        assert 'QQQ' not in symbols
    
    def test_market_cap_filter(self, universe):
        """Test market cap filtering."""
        symbols = universe.filter_symbols(min_market_cap=1e9)
        
        assert 'AAPL' in symbols
        assert 'MSFT' in symbols
        assert 'PENNY' not in symbols

# Test File: tests/test_leakage_checks.py

import pytest
import pandas as pd
import numpy as np
from trader.utils.leakage_checks import LeakageChecker

class TestLeakageChecker:
    """Test leakage detection."""
    
    def test_no_future_data(self):
        """Test future data detection."""
        checker = LeakageChecker()
        
        # Create sample data
        dates = pd.date_range('2024-01-01', periods=100)
        features = pd.DataFrame(np.random.randn(100, 5), index=dates)
        targets = pd.Series(np.random.randn(100), index=dates)
        
        # Correct case - features before targets
        feature_dates = dates[:-1]
        target_dates = dates[1:]
        
        assert checker.check_no_future_data(
            features.iloc[:-1], 
            targets.iloc[1:],
            feature_dates,
            target_dates
        )
        
        # Incorrect case - features after targets
        assert not checker.check_no_future_data(
            features.iloc[1:],
            targets.iloc[:-1],
            dates[1:],
            dates[:-1]
        )
    
    def test_train_test_separation(self):
        """Test train/test separation."""
        checker = LeakageChecker()
        
        train_dates = pd.date_range('2024-01-01', '2024-06-30')
        test_dates = pd.date_range('2024-07-22', '2024-12-31')
        
        # With proper embargo (21 days)
        assert checker.check_train_test_separation(
            train_dates,
            test_dates,
            embargo_days=21
        )
        
        # Without proper embargo
        test_dates_bad = pd.date_range('2024-07-01', '2024-12-31')
        assert not checker.check_train_test_separation(
            train_dates,
            test_dates_bad,
            embargo_days=21
        )
```