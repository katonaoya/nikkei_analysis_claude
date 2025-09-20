#!/usr/bin/env python
"""
Feature generation script for stock analysis
"""

import os
import sys
from pathlib import Path
import argparse
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.features import FeatureEngineer
from src.ml import ModelTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('feature_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def generate_features(
    file_pattern: str = "*nikkei225_historical*",
    output_filename: str = None,
    include_technical: bool = True,
    include_market: bool = True,
    include_labels: bool = True
):
    """Generate features from stock data"""
    
    logger.info("Starting feature generation process...")
    
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    try:
        # Run feature generation pipeline
        features_df, output_path, summary = engineer.run_full_pipeline(
            file_pattern=file_pattern,
            output_filename=output_filename,
            include_technical=include_technical,
            include_market=include_market,
            include_labels=include_labels
        )
        
        # Display summary
        print("\n" + "="*60)
        print("📊 FEATURE GENERATION SUMMARY")
        print("="*60)
        print(f"📄 Total records: {summary['total_records']:,}")
        print(f"🔧 Total features: {summary['total_features']}")
        print(f"📈 Unique stocks: {summary['unique_stocks']}")
        print(f"📅 Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        print(f"💾 Output file: {output_path}")
        
        # Feature breakdown
        print(f"\n📋 GENERATED FEATURES:")
        print("-" * 40)
        
        feature_categories = {
            'Technical Indicators': [col for col in summary['feature_columns'] 
                                   if any(indicator in col.lower() for indicator in 
                                         ['ma_', 'rsi', 'bb_', 'macd', 'ema'])],
            'Market Features': [col for col in summary['feature_columns']
                              if any(market in col.lower() for market in
                                    ['volatility', 'trend', 'breadth', 'relative', 'sector'])],
            'Volume Features': [col for col in summary['feature_columns']
                              if any(vol in col.lower() for vol in
                                    ['volume', 'obv'])],
            'Labels': [col for col in summary['feature_columns']
                      if any(label in col.lower() for label in
                            ['return_', 'target_', 'label_', 'class_'])]
        }
        
        for category, features in feature_categories.items():
            if features:
                print(f"  {category}: {len(features)} features")
                for feature in features[:5]:  # Show first 5
                    print(f"    - {feature}")
                if len(features) > 5:
                    print(f"    ... and {len(features) - 5} more")
                print()
        
        # Data quality info
        missing_count = sum(1 for count in summary['missing_values'].values() if count > 0)
        if missing_count > 0:
            print(f"⚠️  Features with missing values: {missing_count}")
        else:
            print("✅ No missing values detected")
        
        print("\n" + "="*60)
        logger.info("Feature generation completed successfully!")
        
        return features_df, output_path, summary
        
    except Exception as e:
        logger.error(f"Feature generation failed: {e}")
        raise


def train_models(
    features_file: str,
    target_column: str = "next_day_return",
    model_type: str = "regression",
    models_to_train: list = None
):
    """Train machine learning models on generated features"""
    
    logger.info("Starting model training process...")
    
    # Initialize model trainer
    trainer = ModelTrainer()
    
    try:
        # Run training pipeline
        results = trainer.run_training_pipeline(
            features_file=features_file,
            target_column=target_column,
            model_type=model_type,
            models_to_train=models_to_train
        )
        
        # Display results
        print("\n" + "="*60)
        print("🤖 MODEL TRAINING SUMMARY")
        print("="*60)
        print(f"📊 Model type: {results['model_type'].title()}")
        print(f"🎯 Target column: {results['target_column']}")
        print(f"🔧 Feature count: {results['feature_count']}")
        print(f"📝 Training samples: {results['training_samples']:,}")
        
        print(f"\n🏆 TRAINED MODELS:")
        print("-" * 30)
        for model_name in results['models_trained']:
            print(f"✅ {model_name}")
            
            # Show scores if available
            if model_name in results['scores']:
                scores = results['scores'][model_name]
                if model_type == "regression":
                    if 'cv_mse_mean' in scores:
                        print(f"   CV MSE: {scores['cv_mse_mean']:.6f} (±{scores['cv_mse_std']:.6f})")
                else:
                    if 'cv_accuracy_mean' in scores:
                        print(f"   CV Accuracy: {scores['cv_accuracy_mean']:.3f} (±{scores['cv_accuracy_std']:.3f})")
        
        print(f"\n💾 SAVED MODELS:")
        print("-" * 30)
        for model_name, path in results['saved_paths'].items():
            print(f"📄 {model_name}: {path.name}")
        
        print("\n" + "="*60)
        logger.info("Model training completed successfully!")
        
        return results
        
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Generate features and train models for stock analysis")
    
    # Feature generation arguments
    parser.add_argument(
        "--file-pattern",
        type=str,
        default="*nikkei225_historical*",
        help="Pattern to match input data files"
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        help="Custom output filename for features"
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Skip feature generation (use existing features)"
    )
    parser.add_argument(
        "--features-file",
        type=str,
        help="Existing features file to use (if skipping generation)"
    )
    
    # Model training arguments
    parser.add_argument(
        "--train-models",
        action="store_true",
        help="Train models after feature generation"
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="next_day_return",
        help="Target column for model training"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["regression", "classification"],
        default="regression",
        help="Type of models to train"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="Specific models to train (e.g., random_forest xgboost)"
    )
    
    # Feature selection arguments
    parser.add_argument(
        "--no-technical",
        action="store_true",
        help="Skip technical indicators"
    )
    parser.add_argument(
        "--no-market",
        action="store_true",
        help="Skip market features"
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Skip label generation"
    )
    
    args = parser.parse_args()
    
    try:
        features_file = args.features_file
        
        # Generate features (unless skipped)
        if not args.skip_features:
            print("🔧 Starting feature generation...")
            
            features_df, output_path, summary = generate_features(
                file_pattern=args.file_pattern,
                output_filename=args.output_filename,
                include_technical=not args.no_technical,
                include_market=not args.no_market,
                include_labels=not args.no_labels
            )
            
            features_file = output_path.name
        
        elif not features_file:
            raise ValueError("Must provide --features-file when using --skip-features")
        
        # Train models (if requested)
        if args.train_models:
            print("\n🤖 Starting model training...")
            
            results = train_models(
                features_file=features_file,
                target_column=args.target_column,
                model_type=args.model_type,
                models_to_train=args.models
            )
        
        print("\n🎉 Process completed successfully!")
        
    except Exception as e:
        logger.error(f"Process failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())