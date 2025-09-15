#!/usr/bin/env python
"""
統合実運用システム - 1コマンドでデータ取得から予測まで実行
"""

import os
import sys
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import logging
import json
from datetime import datetime, date, timedelta
from typing import Optional, Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TradingPipeline:
    """統合トレーディングパイプライン"""
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path("data")
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.models_dir = self.data_dir / "models"
        self.predictions_dir = self.data_dir / "predictions"
        self.reports_dir = self.data_dir / "reports"
        
        # Create directories
        for dir_path in [self.processed_dir, self.models_dir, self.predictions_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.scripts_dir = project_root / "scripts"
    
    def run_command(self, command: List[str], description: str) -> Dict:
        """コマンド実行とログ記録"""
        logger.info(f"🔧 {description}")
        logger.info(f"   Command: {' '.join(command)}")
        
        try:
            start_time = datetime.now()
            result = subprocess.run(
                command,
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=600  # 10分タイムアウト
            )
            duration = datetime.now() - start_time
            
            if result.returncode == 0:
                logger.info(f"✅ {description} completed in {duration}")
                return {
                    'success': True,
                    'duration': duration,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
            else:
                logger.error(f"❌ {description} failed with return code {result.returncode}")
                logger.error(f"   Error: {result.stderr}")
                return {
                    'success': False,
                    'duration': duration,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'return_code': result.returncode
                }
        
        except subprocess.TimeoutExpired:
            logger.error(f"❌ {description} timed out after 10 minutes")
            return {
                'success': False,
                'error': 'timeout',
                'duration': timedelta(minutes=10)
            }
        except Exception as e:
            logger.error(f"❌ {description} failed with exception: {e}")
            return {
                'success': False,
                'error': str(e),
                'duration': datetime.now() - start_time
            }
    
    def step1_data_collection(
        self, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None,
        batch_size: int = 5,
        delay: float = 1.0
    ) -> Dict:
        """ステップ1: データ収集"""
        
        # デフォルト日付設定（過去30日）
        if not end_date:
            end_date = date.today().strftime("%Y-%m-%d")
        if not start_date:
            start_date = (date.today() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        command = [
            "python", 
            str(self.scripts_dir / "collect_historical_data.py"),
            "--start-date", start_date,
            "--end-date", end_date,
            "--batch-size", str(batch_size),
            "--delay", str(delay)
        ]
        
        result = self.run_command(command, f"データ収集 ({start_date} to {end_date})")
        
        # 成功した場合、最新のデータファイルを取得
        if result['success']:
            data_files = list(self.raw_dir.glob("nikkei225_historical_*.parquet"))
            if data_files:
                latest_file = max(data_files, key=lambda f: f.stat().st_mtime)
                result['data_file'] = latest_file.name
                logger.info(f"📊 Latest data file: {latest_file.name}")
        
        return result
    
    def step2_feature_generation(self, output_filename: Optional[str] = None) -> Dict:
        """ステップ2: 特徴量生成"""
        
        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"features_pipeline_{timestamp}.parquet"
        
        command = [
            "python",
            str(self.scripts_dir / "simple_feature_generation.py"),
            "--output-filename", output_filename
        ]
        
        result = self.run_command(command, "特徴量生成")
        
        if result['success']:
            result['features_file'] = output_filename
            logger.info(f"🔧 Features file: {output_filename}")
        
        return result
    
    def step3_model_training(
        self, 
        features_file: str, 
        models: List[str] = None,
        train_regression: bool = True,
        train_classification: bool = True
    ) -> Dict:
        """ステップ3: モデル訓練"""
        
        results = {}
        
        # 回帰モデル訓練
        if train_regression:
            command = [
                "python",
                str(self.scripts_dir / "simple_model_training.py"),
                "--features-file", features_file,
                "--target", "Next_Day_Return",
                "--model-type", "regression",
                "--save-models"
            ]
            
            reg_result = self.run_command(command, "回帰モデル訓練")
            results['regression'] = reg_result
            
            # 訓練されたモデルファイルを取得
            if reg_result['success']:
                model_files = list(self.models_dir.glob("*Next_Day_Return*.joblib"))
                if model_files:
                    latest_reg_model = max(model_files, key=lambda f: f.stat().st_mtime)
                    results['regression']['model_file'] = latest_reg_model.name
        
        # 分類モデル訓練
        if train_classification:
            command = [
                "python",
                str(self.scripts_dir / "simple_model_training.py"),
                "--features-file", features_file,
                "--target", "Binary_Direction",
                "--model-type", "classification",
                "--save-models"
            ]
            
            clf_result = self.run_command(command, "分類モデル訓練")
            results['classification'] = clf_result
            
            # 訓練されたモデルファイルを取得
            if clf_result['success']:
                model_files = list(self.models_dir.glob("*Binary_Direction*.joblib"))
                if model_files:
                    latest_clf_model = max(model_files, key=lambda f: f.stat().st_mtime)
                    results['classification']['model_file'] = latest_clf_model.name
        
        return results
    
    def step4_prediction(self, model_files: Dict[str, str]) -> Dict:
        """ステップ4: 予測実行"""
        
        results = {}
        
        # 回帰予測
        if 'regression' in model_files and model_files['regression']:
            command = [
                "python",
                str(self.scripts_dir / "simple_prediction.py"),
                "--model-file", model_files['regression'],
                "--save-predictions"
            ]
            
            reg_pred_result = self.run_command(command, "回帰予測実行")
            results['regression'] = reg_pred_result
        
        # 分類予測
        if 'classification' in model_files and model_files['classification']:
            command = [
                "python",
                str(self.scripts_dir / "simple_prediction.py"),
                "--model-file", model_files['classification'],
                "--save-predictions"
            ]
            
            clf_pred_result = self.run_command(command, "分類予測実行")
            results['classification'] = clf_pred_result
        
        return results
    
    def generate_summary_report(self, pipeline_results: Dict) -> Path:
        """総合レポート生成"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.reports_dir / f"pipeline_report_{timestamp}.json"
        
        # レポート内容作成
        report = {
            'timestamp': timestamp,
            'execution_date': date.today().isoformat(),
            'pipeline_results': pipeline_results,
            'summary': {
                'total_duration': sum(
                    [step.get('duration', timedelta(0)).total_seconds() for step in pipeline_results.values()
                     if isinstance(step, dict) and isinstance(step.get('duration'), timedelta)],
                    0
                ),
                'successful_steps': sum(
                    1 for step in pipeline_results.values()
                    if isinstance(step, dict) and step.get('success', False)
                ),
                'failed_steps': sum(
                    1 for step in pipeline_results.values()
                    if isinstance(step, dict) and not step.get('success', True)
                )
            }
        }
        
        # JSON形式で保存（datetime objectは文字列に変換）
        def json_serializer(obj):
            if isinstance(obj, timedelta):
                return obj.total_seconds()
            elif isinstance(obj, datetime):
                return obj.isoformat()
            return str(obj)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=json_serializer)
        
        logger.info(f"📋 Summary report saved: {report_file}")
        return report_file
    
    def run_full_pipeline(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        skip_data_collection: bool = False,
        train_regression: bool = True,
        train_classification: bool = True
    ) -> Dict:
        """フルパイプライン実行"""
        
        logger.info("🚀 トレーディングパイプライン開始")
        logger.info("="*60)
        
        pipeline_results = {}
        start_time = datetime.now()
        
        try:
            # Step 1: データ収集
            if not skip_data_collection:
                logger.info("📊 Step 1: データ収集")
                pipeline_results['data_collection'] = self.step1_data_collection(
                    start_date, end_date
                )
                
                if not pipeline_results['data_collection']['success']:
                    logger.error("❌ データ収集に失敗しました")
                    return pipeline_results
            
            # Step 2: 特徴量生成
            logger.info("🔧 Step 2: 特徴量生成")
            pipeline_results['feature_generation'] = self.step2_feature_generation()
            
            if not pipeline_results['feature_generation']['success']:
                logger.error("❌ 特徴量生成に失敗しました")
                return pipeline_results
            
            features_file = pipeline_results['feature_generation']['features_file']
            
            # Step 3: モデル訓練
            logger.info("🤖 Step 3: モデル訓練")
            pipeline_results['model_training'] = self.step3_model_training(
                features_file, 
                train_regression=train_regression,
                train_classification=train_classification
            )
            
            # 訓練されたモデルファイルを取得
            model_files = {}
            if train_regression and 'regression' in pipeline_results['model_training']:
                if pipeline_results['model_training']['regression'].get('success'):
                    model_files['regression'] = pipeline_results['model_training']['regression'].get('model_file')
            
            if train_classification and 'classification' in pipeline_results['model_training']:
                if pipeline_results['model_training']['classification'].get('success'):
                    model_files['classification'] = pipeline_results['model_training']['classification'].get('model_file')
            
            # Step 4: 予測実行
            if model_files:
                logger.info("🔮 Step 4: 予測実行")
                pipeline_results['predictions'] = self.step4_prediction(model_files)
            else:
                logger.warning("⚠️ 訓練されたモデルがないため、予測をスキップします")
            
            # 総実行時間
            total_duration = datetime.now() - start_time
            pipeline_results['total_duration'] = total_duration
            
            # サマリーレポート生成
            report_file = self.generate_summary_report(pipeline_results)
            
            logger.info("="*60)
            logger.info(f"✅ パイプライン完了! 総実行時間: {total_duration}")
            logger.info(f"📋 詳細レポート: {report_file}")
            
            return pipeline_results
            
        except Exception as e:
            logger.error(f"❌ パイプライン実行中にエラーが発生しました: {e}")
            pipeline_results['error'] = str(e)
            pipeline_results['total_duration'] = datetime.now() - start_time
            return pipeline_results


def print_pipeline_summary(results: Dict):
    """パイプライン結果のサマリー表示"""
    
    print("\n" + "="*60)
    print("📊 TRADING PIPELINE EXECUTION SUMMARY")
    print("="*60)
    
    # 全体のサマリー
    total_duration = results.get('total_duration', timedelta(0))
    print(f"⏱️  総実行時間: {total_duration}")
    print(f"📅 実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 各ステップの結果
    steps = [
        ('data_collection', 'データ収集'),
        ('feature_generation', '特徴量生成'),
        ('model_training', 'モデル訓練'),
        ('predictions', '予測実行')
    ]
    
    print(f"\n📋 実行ステップ:")
    print("-" * 40)
    
    for step_key, step_name in steps:
        if step_key in results:
            step_result = results[step_key]
            if isinstance(step_result, dict):
                if step_result.get('success'):
                    print(f"✅ {step_name}: 成功")
                else:
                    print(f"❌ {step_name}: 失敗")
            else:
                print(f"📝 {step_name}: 実行済み")
        else:
            print(f"⏭️  {step_name}: スキップ")
    
    # 成功したモデルがあれば表示
    if 'model_training' in results:
        training_results = results['model_training']
        if isinstance(training_results, dict):
            print(f"\n🤖 訓練されたモデル:")
            print("-" * 30)
            
            for model_type in ['regression', 'classification']:
                if model_type in training_results:
                    model_result = training_results[model_type]
                    if model_result.get('success') and 'model_file' in model_result:
                        print(f"  ✅ {model_type}: {model_result['model_file']}")
                    else:
                        print(f"  ❌ {model_type}: 訓練失敗")
    
    print("\n" + "="*60)


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="統合トレーディングパイプライン実行")
    
    # データ収集関連
    parser.add_argument(
        "--start-date",
        type=str,
        help="データ取得開始日 (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", 
        type=str,
        help="データ取得終了日 (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--skip-data-collection",
        action="store_true",
        help="データ収集をスキップ"
    )
    
    # モデル関連
    parser.add_argument(
        "--no-regression",
        action="store_true", 
        help="回帰モデルの訓練をスキップ"
    )
    parser.add_argument(
        "--no-classification",
        action="store_true",
        help="分類モデルの訓練をスキップ"
    )
    
    # その他
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="詳細出力を抑制"
    )
    
    args = parser.parse_args()
    
    # ログレベル調整
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    try:
        # パイプライン実行
        pipeline = TradingPipeline()
        
        results = pipeline.run_full_pipeline(
            start_date=args.start_date,
            end_date=args.end_date,
            skip_data_collection=args.skip_data_collection,
            train_regression=not args.no_regression,
            train_classification=not args.no_classification
        )
        
        # 結果表示
        if not args.quiet:
            print_pipeline_summary(results)
        
        # 成功/失敗の判定
        if 'error' in results:
            return 1
        
        # 主要ステップの成功確認
        critical_steps = ['feature_generation']
        for step in critical_steps:
            if step in results and isinstance(results[step], dict):
                if not results[step].get('success', False):
                    return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"パイプライン実行エラー: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())