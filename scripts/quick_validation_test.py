#!/usr/bin/env python3
"""
Quick validation test for Step 2 implementation
Focus on core functionality validation
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import warnings

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

def setup_logging():
    """Setup simple logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def create_minimal_test_data() -> pd.DataFrame:
    """Create minimal test data"""
    dates = pd.date_range('2024-01-01', periods=50, freq='D')
    data = []
    
    for i, date in enumerate(dates):
        base_price = 1000 + i * 10
        data.append({
            'Date': date.date(),
            'Code': '1000',
            'Open': base_price,
            'High': base_price * 1.02,
            'Low': base_price * 0.98,
            'Close': base_price * (1 + np.random.normal(0, 0.01)),
            'Volume': 100000
        })
    
    return pd.DataFrame(data)

def test_basic_model_functionality():
    """Test basic model functionality without complex features"""
    print("=== Testing Basic Model Functionality ===")
    
    try:
        # Test individual models
        from models.lightgbm_model import LightGBMModel
        from models.catboost_model import CatBoostModel
        from models.logistic_model import LogisticRegressionModel
        
        # Create simple test data
        X = np.random.randn(100, 10)  # 100 samples, 10 features
        y = np.random.binomial(1, 0.3, 100)  # Imbalanced binary target
        
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        y_series = pd.Series(y)
        
        # Split data
        X_train, X_val = X_df[:70], X_df[70:]
        y_train, y_val = y_series[:70], y_series[70:]
        
        models = [
            ("LightGBM", LightGBMModel()),
            ("CatBoost", CatBoostModel()),
            ("LogisticRegression", LogisticRegressionModel())
        ]
        
        trained_models = {}
        
        for model_name, model in models:
            print(f"Testing {model_name}...")
            
            # Get default params and reduce complexity for testing
            params = model.get_default_params()
            if 'n_estimators' in params:
                params['n_estimators'] = 50
            if 'iterations' in params:
                params['iterations'] = 50
            
            # Train model
            model.fit(X_train, y_train, X_val, y_val, params=params)
            
            # Test predictions
            pred_proba = model.predict_proba(X_val)
            pred_binary = model.predict(X_val)
            
            print(f"  {model_name}: proba_shape={pred_proba.shape}, predictions={pred_binary.sum()}/{len(pred_binary)}")
            
            # Basic validations
            assert pred_proba.shape == (len(X_val), 2), f"Wrong proba shape for {model_name}"
            assert len(pred_binary) == len(X_val), f"Wrong predictions length for {model_name}"
            assert np.all((pred_proba >= 0) & (pred_proba <= 1)), f"Invalid probabilities for {model_name}"
            
            trained_models[model_name.lower()] = model
            print(f"  ✅ {model_name} passed")
        
        print("✅ Basic model functionality test passed")
        return trained_models, X_val, y_val
        
    except Exception as e:
        print(f"❌ Basic model functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_ensemble_basic():
    """Test basic ensemble functionality"""
    print("\n=== Testing Basic Ensemble Functionality ===")
    
    try:
        from models.ensemble_model import EnsembleModel
        
        # Get trained models
        model_result = test_basic_model_functionality()
        if model_result is None:
            print("❌ Cannot test ensemble without trained models")
            return None
        
        trained_models, X_val, y_val = model_result
        
        # Create ensemble
        ensemble = EnsembleModel()
        
        # Add models
        for model_name, model in trained_models.items():
            ensemble.add_model(model)
            print(f"  Added {model_name} to ensemble")
        
        # Test basic predictions
        ensemble_probs = ensemble.predict_proba(X_val)
        ensemble_preds = ensemble.predict(X_val)
        
        print(f"  Ensemble predictions: shape={ensemble_probs.shape}, positive_preds={ensemble_preds.sum()}")
        
        # Basic validations
        assert ensemble_probs.shape == (len(X_val), 2), "Wrong ensemble proba shape"
        assert len(ensemble_preds) == len(X_val), "Wrong ensemble predictions length"
        assert np.all((ensemble_probs >= 0) & (ensemble_probs <= 1)), "Invalid ensemble probabilities"
        
        print("✅ Basic ensemble functionality test passed")
        return ensemble, X_val, y_val
        
    except Exception as e:
        print(f"❌ Basic ensemble functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_calibration_basic():
    """Test basic calibration functionality"""
    print("\n=== Testing Basic Calibration Functionality ===")
    
    try:
        from models.calibration_optimizer import CalibrationOptimizer
        
        # Get ensemble
        ensemble_result = test_ensemble_basic()
        if ensemble_result is None:
            print("❌ Cannot test calibration without ensemble")
            return False
        
        ensemble, X_val, y_val = ensemble_result
        
        # Split validation data for calibration
        split_idx = len(X_val) // 2
        X_cal, X_test = X_val[:split_idx], X_val[split_idx:]
        y_cal, y_test = y_val[:split_idx], y_val[split_idx:]
        
        # Test calibration
        ensemble_probas = ensemble.predict_proba(X_cal, use_calibration=False)[:, 1]
        
        calibrator = CalibrationOptimizer()
        calibrator.fit_advanced_calibration(ensemble_probas, y_cal, method='isotonic')
        
        # Test calibrated predictions
        calibrated_probas = calibrator.predict_calibrated(ensemble_probas, method='isotonic')
        
        print(f"  Calibration: original_range=[{ensemble_probas.min():.3f}, {ensemble_probas.max():.3f}]")
        print(f"  Calibrated range: [{calibrated_probas.min():.3f}, {calibrated_probas.max():.3f}]")
        
        # Basic validations
        assert len(calibrated_probas) == len(ensemble_probas), "Calibrated probabilities length mismatch"
        assert np.all((calibrated_probas >= 0) & (calibrated_probas <= 1)), "Invalid calibrated probabilities"
        
        print("✅ Basic calibration functionality test passed")
        return True
        
    except Exception as e:
        print(f"❌ Basic calibration functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_hyperopt_basic():
    """Test basic hyperparameter optimization"""
    print("\n=== Testing Basic Hyperparameter Optimization ===")
    
    try:
        from models.hyperparameter_optimizer import HyperparameterOptimizer
        from models.lightgbm_model import LightGBMModel
        
        # Create small dataset for quick optimization
        X = np.random.randn(50, 5)  # Small dataset
        y = np.random.binomial(1, 0.3, 50)
        
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        y_series = pd.Series(y)
        
        X_train, X_val = X_df[:35], X_df[35:]
        y_train, y_val = y_series[:35], y_series[35:]
        
        # Create optimizer with minimal trials
        optimizer = HyperparameterOptimizer({
            'optimization.n_trials': 3,  # Very few trials for quick testing
            'optimization.target_precision': 0.75
        })
        
        # Test optimization
        results = optimizer.optimize_model(
            LightGBMModel,
            X_train, y_train,
            X_val, y_val,
            study_name="quick_test"
        )
        
        print(f"  Optimization completed: n_trials={results['n_completed_trials']}")
        print(f"  Best score: {results['best_score']:.4f}")
        
        # Basic validations
        assert 'best_params' in results, "No best parameters found"
        assert results['n_completed_trials'] > 0, "No trials completed"
        
        print("✅ Basic hyperparameter optimization test passed")
        return True
        
    except Exception as e:
        print(f"❌ Basic hyperparameter optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_quick_validation():
    """Run quick validation tests"""
    print("Starting Quick Validation Test for Step 2 Implementation")
    print("=" * 60)
    
    # Suppress warnings
    warnings.filterwarnings("ignore")
    
    test_results = []
    
    # Test 1: Basic Model Functionality
    model_result = test_basic_model_functionality()
    test_results.append(("Model Functionality", model_result is not None))
    
    # Test 2: Basic Ensemble
    ensemble_result = test_ensemble_basic()
    test_results.append(("Ensemble Integration", ensemble_result is not None))
    
    # Test 3: Basic Calibration
    calibration_result = test_calibration_basic()
    test_results.append(("Calibration Functionality", calibration_result))
    
    # Test 4: Basic Hyperparameter Optimization
    hyperopt_result = test_hyperopt_basic()
    test_results.append(("Hyperparameter Optimization", hyperopt_result))
    
    # Print results
    print("\n" + "=" * 60)
    print("QUICK VALIDATION RESULTS")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in test_results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:<30}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("🎉 ALL CORE TESTS PASSED!")
        print("✅ Step 2 implementation is functionally working")
        print("✅ Ready to proceed to Step 3 (Backtest & Validation)")
    else:
        print("❌ SOME CORE TESTS FAILED!")
        print("⚠️  Issues need investigation before proceeding")
    
    print("=" * 60)
    
    return all_passed

def main():
    """Main execution"""
    setup_logging()
    
    try:
        success = run_quick_validation()
        return 0 if success else 1
    except Exception as e:
        print(f"Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())