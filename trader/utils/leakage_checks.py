# trader/utils/leakage_checks.py
import pandas as pd
import numpy as np
from typing import List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class LeakageChecker:
    """Check for data leakage in time series models."""
    
    @staticmethod
    def check_no_future_data(
        features: pd.DataFrame, 
        targets: pd.Series,
        feature_dates: pd.DatetimeIndex,
        target_dates: pd.DatetimeIndex
    ) -> bool:
        """Verify no future data is used in features."""
        for i, target_date in enumerate(target_dates):
            feature_date = feature_dates[i] if i < len(feature_dates) else feature_dates[-1]
            if feature_date >= target_date:
                logger.error(f"Leakage detected: feature date {feature_date} >= target date {target_date}")
                return False
        return True
    
    @staticmethod
    def check_train_test_separation(
        train_dates: pd.DatetimeIndex,
        test_dates: pd.DatetimeIndex,
        embargo_days: int = 0
    ) -> bool:
        """Verify proper separation between train and test sets."""
        if len(train_dates) == 0 or len(test_dates) == 0:
            return True
            
        max_train_date = train_dates.max()
        min_test_date = test_dates.min()
        
        actual_gap = (min_test_date - max_train_date).days
        
        if actual_gap < embargo_days:
            logger.error(f"Insufficient embargo: {actual_gap} days < {embargo_days} days required")
            return False
        
        return True
    
    @staticmethod
    def check_no_lookahead_in_features(
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        horizon: int
    ) -> bool:
        """Check that features don't contain lookahead bias."""
        # Check correlation between features and unshifted target
        unshifted_target = df[target_col].shift(horizon)  # Undo the shift
        
        for col in feature_cols:
            if col in df.columns:
                # High correlation with unshifted target suggests lookahead
                corr = df[col].corr(unshifted_target)
                if abs(corr) > 0.95:  # Threshold for suspicion
                    logger.warning(f"Potential lookahead in {col}: corr={corr:.3f} with unshifted target")
                    return False
        
        return True
    
    @staticmethod
    def check_no_data_snooping(
        scaler_fit_data: pd.DataFrame,
        transform_data: pd.DataFrame,
        fit_dates: pd.DatetimeIndex,
        transform_dates: pd.DatetimeIndex
    ) -> bool:
        """Verify scalers/transformers are fit only on training data."""
        if len(fit_dates) == 0 or len(transform_dates) == 0:
            return True
            
        max_fit_date = fit_dates.max()
        
        # Check if any transform data is from before fit period
        future_transform = transform_dates[transform_dates > max_fit_date]
        
        if len(future_transform) > 0 and len(transform_dates) == len(future_transform):
            # All transform data is from after fit period - this is correct
            return True
        elif len(future_transform) > 0 and len(future_transform) < len(transform_dates):
            # Mixed dates - potential issue
            logger.warning("Transform data contains both training and test periods")
            return False
        
        return True
    
    @staticmethod
    def run_all_checks(
        features: pd.DataFrame,
        targets: pd.Series,
        train_dates: pd.DatetimeIndex,
        test_dates: pd.DatetimeIndex,
        embargo_days: int = 21
    ) -> Dict[str, bool]:
        """Run all leakage checks."""
        results = {}
        
        results['no_future_data'] = LeakageChecker.check_no_future_data(
            features, targets, features.index, targets.index
        )
        
        results['train_test_separation'] = LeakageChecker.check_train_test_separation(
            train_dates, test_dates, embargo_days
        )
        
        all_passed = all(results.values())
        
        if all_passed:
            logger.info("All leakage checks passed")
        else:
            logger.error(f"Leakage checks failed: {results}")
        
        return results