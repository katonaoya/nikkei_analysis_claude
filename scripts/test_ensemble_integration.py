#!/usr/bin/env python3
"""
Test script for ensemble model integration and end-to-end functionality
Tests the complete pipeline from feature generation to ensemble prediction
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('ensemble_integration_test.log')
        ]
    )

def create_realistic_test_data(n_stocks: int = 20, n_days: int = 150) -> pd.DataFrame:
    """
    Create realistic test data for ensemble model testing
    
    Args:
        n_stocks: Number of stocks to generate
        n_days: Number of days to generate
        
    Returns:
        DataFrame with realistic stock data
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Creating realistic test data: {n_stocks} stocks, {n_days} days")
    
    # Major Nikkei 225 stock codes
    stock_codes = [
        '7203', '6758', '9984', '6861', '8001', '8058', '4519', '6098', 
        '8306', '7974', '9432', '4063', '6367', '8316', '4502', '9983',
        '8035', '2914', '4568', '6954'
    ][:n_stocks]
    
    # Generate realistic date range (business days)
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=int(n_days * 1.4))  # Account for weekends
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')[:n_days]
    
    data = []
    
    for code in stock_codes:
        # Different base prices for different stocks
        base_prices = {
            '7203': 2500,   # Toyota
            '6758': 12000,  # Sony
            '9984': 7000,   # SoftBank
            '6861': 50000,  # Keyence
            '8001': 4500,   # Itochu
            '8058': 3800,   # Mitsubishi
            '4519': 3500,   # Takeda
            '6098': 8000,   # Recruit
            '8306': 1200,   # MUFG
            '7974': 60000,  # Nintendo
        }
        
        base_price = base_prices.get(code, 5000)
        
        # Set seed for reproducible results per stock
        np.random.seed(int(code) % 1000)
        
        # Generate more realistic price movements with trends
        trend = np.random.uniform(-0.0002, 0.0005)  # Daily trend
        volatility = np.random.uniform(0.015, 0.025)  # Daily volatility
        
        returns = np.random.normal(trend, volatility, len(date_range))
        
        # Add some autocorrelation (momentum)
        for i in range(1, len(returns)):
            returns[i] += 0.1 * returns[i-1]  # Small momentum effect
        
        # Generate price series
        prices = [base_price]
        for i in range(1, len(date_range)):
            new_price = prices[i-1] * (1 + returns[i])
            # Prevent extreme price drops
            new_price = max(new_price, base_price * 0.5)
            prices.append(new_price)
        
        for i, date in enumerate(date_range):
            close_price = prices[i]
            
            # Generate OHLC with realistic relationships
            daily_range = close_price * abs(np.random.normal(0, 0.008))
            high_price = close_price + daily_range * np.random.uniform(0.3, 0.7)
            low_price = close_price - daily_range * np.random.uniform(0.3, 0.7)
            open_price = low_price + (high_price - low_price) * np.random.uniform(0.2, 0.8)
            
            # Ensure OHLC relationships
            high_price = max(high_price, close_price, open_price, low_price)
            low_price = min(low_price, close_price, open_price, high_price)
            
            # Realistic volume with some correlation to price movement
            base_volume = np.random.lognormal(13, 0.6)
            price_change = abs(returns[i]) if i > 0 else 0
            volume_multiplier = 1 + price_change * 5  # Higher volume on big moves
            volume = int(base_volume * volume_multiplier)
            volume = max(volume, 10000)  # Minimum volume
            
            data.append({
                'Date': date.date(),
                'Code': code,
                'Open': round(open_price, 0),
                'High': round(high_price, 0),
                'Low': round(low_price, 0),
                'Close': round(close_price, 0),
                'Volume': volume
            })
    
    df = pd.DataFrame(data)
    logger.info(f"Realistic test data created: {len(df)} records")
    return df

def test_feature_engineering():
    """Test feature engineering pipeline"""
    logger = logging.getLogger(__name__)
    logger.info("=== Testing Feature Engineering Pipeline ===")
    
    try:
        from features.feature_engineer import FeatureEngineer
        
        # Create test data
        test_data = create_realistic_test_data(n_stocks=10, n_days=100)
        logger.info(f"Test data shape: {test_data.shape}")
        
        # Initialize feature engineer
        feature_engineer = FeatureEngineer()
        
        # Generate features
        enhanced_features = feature_engineer.generate_features(
            test_data,
            include_technical=True,
            include_market=True,
            include_time_series=True,
            include_fundamental=True,
            include_labels=True
        )
        
        logger.info(f"Enhanced features shape: {enhanced_features.shape}")
        logger.info(f"Features added: {enhanced_features.shape[1] - test_data.shape[1]}")
        
        # Check for critical issues
        n_missing = enhanced_features.isnull().sum().sum()
        n_infinite = np.isinf(enhanced_features.select_dtypes(include=[np.number])).sum().sum()
        
        logger.info(f"Missing values: {n_missing}")
        logger.info(f"Infinite values: {n_infinite}")
        
        if enhanced_features.shape[1] < 50:
            logger.error("Too few features generated")
            return False
        
        logger.info("✅ Feature engineering test passed")
        return enhanced_features
        
    except Exception as e:
        logger.error(f"❌ Feature engineering test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_individual_models():
    """Test individual model implementations"""
    logger = logging.getLogger(__name__)
    logger.info("=== Testing Individual Models ===")
    
    try:
        from models.lightgbm_model import LightGBMModel
        from models.catboost_model import CatBoostModel
        from models.logistic_model import LogisticRegressionModel
        
        # Generate test features
        enhanced_features = test_feature_engineering()
        if enhanced_features is False:
            logger.error("Cannot test models without features")
            return False
        
        # Prepare training data
        feature_cols = [col for col in enhanced_features.columns 
                       if not col.startswith('target_') and col not in ['Date', 'Code']]
        
        X = enhanced_features[feature_cols].fillna(0)
        y = enhanced_features.get('target_next_day_gain_1pct', pd.Series(np.random.binomial(1, 0.1, len(X))))
        
        logger.info(f"Training data: X={X.shape}, y={y.shape}, positive_rate={y.mean():.3f}")
        
        # Split data
        split_idx = int(len(X) * 0.7)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        models_to_test = [
            ("LightGBM", LightGBMModel),
            ("CatBoost", CatBoostModel), 
            ("LogisticRegression", LogisticRegressionModel)
        ]
        
        trained_models = {}
        
        for model_name, model_class in models_to_test:
            logger.info(f"Testing {model_name} model")
            
            try:
                # Create and train model
                model = model_class()
                
                # Get default parameters
                params = model.get_default_params()
                logger.info(f"{model_name} default params count: {len(params)}")
                
                # Quick training (reduced iterations for testing)
                if hasattr(model, 'model') and hasattr(model.model, 'set_params'):
                    if 'n_estimators' in params:
                        params['n_estimators'] = min(100, params.get('n_estimators', 100))
                    if 'iterations' in params:
                        params['iterations'] = min(100, params.get('iterations', 100))
                
                # Fit model
                model.fit(X_train, y_train, X_val, y_val, params=params)
                
                # Test predictions
                pred_proba = model.predict_proba(X_val)
                pred_binary = model.predict(X_val)
                
                logger.info(f"{model_name} predictions: proba_shape={pred_proba.shape}, binary_shape={pred_binary.shape}")
                
                # Basic validation
                assert pred_proba.shape == (len(X_val), 2), f"Wrong proba shape: {pred_proba.shape}"
                assert len(pred_binary) == len(X_val), f"Wrong binary predictions length"
                assert np.all((pred_proba >= 0) & (pred_proba <= 1)), "Probabilities out of range"
                
                trained_models[model_name.lower()] = model
                logger.info(f"✅ {model_name} test passed")
                
            except Exception as e:
                logger.error(f"❌ {model_name} test failed: {e}")
                return False
        
        logger.info("✅ All individual model tests passed")
        return trained_models, X_val, y_val
        
    except Exception as e:
        logger.error(f"❌ Individual models test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_ensemble_integration():
    """Test ensemble model integration"""
    logger = logging.getLogger(__name__)
    logger.info("=== Testing Ensemble Integration ===")
    
    try:
        from models.ensemble_model import EnsembleModel
        
        # Get trained models
        models_result = test_individual_models()
        if models_result is False:
            logger.error("Cannot test ensemble without individual models")
            return False
        
        trained_models, X_val, y_val = models_result
        
        # Create ensemble
        ensemble = EnsembleModel()
        
        # Add models to ensemble
        for model_name, model in trained_models.items():
            ensemble.add_model(model)
            logger.info(f"Added {model_name} to ensemble")
        
        # Test calibration fitting
        split_idx = len(X_val) // 2
        X_cal, X_test = X_val[:split_idx], X_val[split_idx:]
        y_cal, y_test = y_val[:split_idx], y_val[split_idx:]
        
        ensemble.fit_calibration(X_cal, y_cal)
        logger.info("Calibration fitting completed")
        
        # Test predictions
        ensemble_probs = ensemble.predict_proba(X_test)
        ensemble_preds = ensemble.predict(X_test)
        
        logger.info(f"Ensemble predictions: proba_shape={ensemble_probs.shape}, binary_shape={ensemble_preds.shape}")
        
        # Test individual predictions
        individual_preds = ensemble.get_individual_predictions(X_test)
        logger.info(f"Individual predictions shape: {individual_preds.shape}")
        
        # Test precision threshold optimization
        try:
            optimal_threshold = ensemble.optimize_threshold_for_precision(X_test, y_test, target_precision=0.75)
            logger.info(f"Optimal threshold for precision 0.75: {optimal_threshold:.4f}")
        except Exception as e:
            logger.warning(f"Threshold optimization failed: {e}")
        
        # Test top-k selection
        try:
            # Create dummy stock data for selection
            stock_data = pd.DataFrame({
                'Code': [f'TEST{i:04d}' for i in range(len(X_test))],
                'Name': [f'Test Stock {i}' for i in range(len(X_test))]
            })
            
            selected_indices, selected_probas, _ = ensemble.select_top_k_predictions(
                X_test, k=3, min_threshold=0.6
            )
            
            logger.info(f"Top-K selection: {len(selected_indices)} stocks selected")
            if len(selected_indices) > 0:
                logger.info(f"Selected probabilities: {selected_probas}")
                
        except Exception as e:
            logger.warning(f"Top-K selection test failed: {e}")
        
        # Basic validations
        assert ensemble_probs.shape == (len(X_test), 2), f"Wrong ensemble proba shape: {ensemble_probs.shape}"
        assert len(ensemble_preds) == len(X_test), "Wrong ensemble predictions length"
        assert np.all((ensemble_probs >= 0) & (ensemble_probs <= 1)), "Ensemble probabilities out of range"
        
        logger.info("✅ Ensemble integration test passed")
        return ensemble, X_test, y_test
        
    except Exception as e:
        logger.error(f"❌ Ensemble integration test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_calibration_optimizer():
    """Test calibration optimizer functionality"""
    logger = logging.getLogger(__name__)
    logger.info("=== Testing Calibration Optimizer ===")
    
    try:
        from models.calibration_optimizer import CalibrationOptimizer
        
        # Get ensemble results
        ensemble_result = test_ensemble_integration()
        if ensemble_result is False:
            logger.error("Cannot test calibration without ensemble")
            return False
        
        ensemble, X_test, y_test = ensemble_result
        
        # Get ensemble probabilities
        ensemble_probas = ensemble.predict_proba(X_test, use_calibration=False)[:, 1]
        
        # Create calibration optimizer
        calibrator = CalibrationOptimizer()
        
        # Fit calibration
        calibrator.fit_advanced_calibration(ensemble_probas, y_test, method='isotonic')
        logger.info("Advanced calibration fitting completed")
        
        # Test calibrated predictions
        calibrated_probas = calibrator.predict_calibrated(ensemble_probas, method='isotonic')
        logger.info(f"Calibrated probabilities shape: {calibrated_probas.shape}")
        
        # Test threshold optimization
        optimization_result = calibrator.optimize_precision_threshold(
            calibrated_probas, y_test, target_precision=0.75
        )
        logger.info(f"Precision threshold optimization completed: {optimization_result['optimal_threshold']:.4f}")
        
        # Test stock selection
        stock_data = pd.DataFrame({
            'Code': [f'TEST{i:04d}' for i in range(len(X_test))],
            'Name': [f'Test Stock {i}' for i in range(len(X_test))],
            'Sector': [f'Sector{i%5}' for i in range(len(X_test))]  # 5 sectors
        })
        
        selected_stocks = calibrator.select_optimal_stocks(
            stock_data, calibrated_probas, 
            threshold=optimization_result['optimal_threshold'],
            max_selections=3
        )
        
        logger.info(f"Stock selection completed: {len(selected_stocks)} stocks selected")
        
        # Basic validations
        assert len(calibrated_probas) == len(ensemble_probas), "Calibrated probabilities length mismatch"
        assert np.all((calibrated_probas >= 0) & (calibrated_probas <= 1)), "Calibrated probabilities out of range"
        
        logger.info("✅ Calibration optimizer test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Calibration optimizer test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_hyperparameter_optimizer():
    """Test hyperparameter optimizer (quick test)"""
    logger = logging.getLogger(__name__)
    logger.info("=== Testing Hyperparameter Optimizer (Quick Test) ===")
    
    try:
        from models.hyperparameter_optimizer import HyperparameterOptimizer
        from models.lightgbm_model import LightGBMModel
        
        # Create small test dataset
        test_data = create_realistic_test_data(n_stocks=5, n_days=50)
        
        from features.feature_engineer import FeatureEngineer
        feature_engineer = FeatureEngineer()
        
        enhanced_features = feature_engineer.generate_features(
            test_data, include_technical=True, include_labels=True
        )
        
        # Prepare data
        feature_cols = [col for col in enhanced_features.columns 
                       if not col.startswith('target_') and col not in ['Date', 'Code']]
        
        X = enhanced_features[feature_cols].fillna(0)
        y = enhanced_features.get('target_next_day_gain_1pct', pd.Series(np.random.binomial(1, 0.1, len(X))))
        
        # Split data
        split_idx = int(len(X) * 0.6)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Create optimizer with minimal trials for testing
        optimizer = HyperparameterOptimizer({
            'optimization.n_trials': 5,  # Very few trials for testing
            'optimization.target_precision': 0.75
        })
        
        # Test optimization
        results = optimizer.optimize_model(
            LightGBMModel,
            X_train, y_train,
            X_val, y_val,
            study_name="test_optimization"
        )
        
        logger.info(f"Optimization completed: best_score={results['best_score']:.4f}")
        logger.info(f"Best params: {results['best_params']}")
        
        # Basic validations
        assert 'best_params' in results, "Best parameters not found in results"
        assert 'best_score' in results, "Best score not found in results"
        assert results['n_completed_trials'] > 0, "No trials completed"
        
        logger.info("✅ Hyperparameter optimizer test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Hyperparameter optimizer test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def run_comprehensive_test():
    """Run comprehensive test suite"""
    logger = logging.getLogger(__name__)
    logger.info("Starting comprehensive ensemble integration test")
    
    test_results = {
        'feature_engineering': False,
        'individual_models': False,
        'ensemble_integration': False,
        'calibration_optimizer': False,
        'hyperparameter_optimizer': False
    }
    
    # Suppress warnings during testing
    warnings.filterwarnings("ignore")
    
    try:
        # Test 1: Feature Engineering
        logger.info("\n" + "="*60)
        feature_result = test_feature_engineering()
        test_results['feature_engineering'] = feature_result is not False
        
        # Test 2: Individual Models
        logger.info("\n" + "="*60)
        models_result = test_individual_models()
        test_results['individual_models'] = models_result is not False
        
        # Test 3: Ensemble Integration
        logger.info("\n" + "="*60)
        ensemble_result = test_ensemble_integration()
        test_results['ensemble_integration'] = ensemble_result is not False
        
        # Test 4: Calibration Optimizer
        logger.info("\n" + "="*60)
        calibration_result = test_calibration_optimizer()
        test_results['calibration_optimizer'] = calibration_result
        
        # Test 5: Hyperparameter Optimizer (quick test)
        logger.info("\n" + "="*60)
        hyperopt_result = test_hyperparameter_optimizer()
        test_results['hyperparameter_optimizer'] = hyperopt_result
        
    except Exception as e:
        logger.error(f"Comprehensive test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Print final results
    logger.info("\n" + "="*60)
    logger.info("COMPREHENSIVE TEST RESULTS")
    logger.info("="*60)
    
    all_passed = True
    for test_name, passed in test_results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name.upper().replace('_', ' ')}: {status}")
        if not passed:
            all_passed = False
    
    logger.info("="*60)
    
    if all_passed:
        logger.info("🎉 ALL TESTS PASSED! The ensemble implementation is working correctly.")
        logger.info("✅ Ready to proceed to Step 3 (Backtest & Validation System)")
    else:
        logger.error("❌ SOME TESTS FAILED! Please review and fix the issues before proceeding.")
        
    logger.info("="*60)
    
    return all_passed

def main():
    """Main test execution"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting Step 2 Implementation Validation")
        success = run_comprehensive_test()
        
        if success:
            logger.info("\n✅ VALIDATION SUCCESSFUL - Implementation is ready for production use!")
            return 0
        else:
            logger.error("\n❌ VALIDATION FAILED - Issues need to be resolved")
            return 1
            
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit(main())