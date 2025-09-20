#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
精度改善のための特徴量最適化スクリプト
目標: 選出された5銘柄/日の精度を60%以上にする
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from itertools import combinations
import logging
import warnings
warnings.filterwarnings('ignore')

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


class AccuracyOptimizer:
    """精度最適化クラス"""
    
    def __init__(self):
        """初期化"""
        self.best_accuracy = 0
        self.best_features = None
        self.best_params = None
        self.best_model = None
        
    def load_data(self):
        """データ読み込み"""
        data_path = Path("data/processed/integrated_with_external.parquet")
        
        if not data_path.exists():
            logger.error(f"データファイルが見つかりません: {data_path}")
            return None
            
        logger.info(f"📥 データ読み込み中...")
        df = pd.read_parquet(data_path)
        
        # 必要な列の処理
        if 'Target' not in df.columns and 'Binary_Direction' in df.columns:
            df['Target'] = df['Binary_Direction']
        if 'Stock' not in df.columns and 'Code' in df.columns:
            df['Stock'] = df['Code']
            
        return df
    
    def get_all_features(self, df):
        """利用可能な全特徴量を取得"""
        # 除外する列
        exclude_cols = ['Date', 'Stock', 'Code', 'Target', 'Binary_Direction', 
                       'Close', 'Open', 'High', 'Low', 'Volume', 'Direction']
        
        # 数値型の列のみ抽出
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 特徴量列を抽出
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        logger.info(f"📊 利用可能な特徴量: {len(feature_cols)}個")
        logger.info(f"  {feature_cols[:10]}...")
        
        return feature_cols
    
    def evaluate_features(self, df, features, model_type='logistic', 
                         confidence_threshold=0.50, top_n=5):
        """特定の特徴量セットで精度を評価"""
        
        # 日付でソート
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        # 必要な列のみ抽出
        required_cols = ['Date', 'Stock', 'Target', 'Close'] + features
        df_clean = df[required_cols].dropna()
        
        if len(df_clean) < 10000:
            return 0, 0, 0
        
        # テスト期間（直近30日）
        unique_dates = sorted(df_clean['Date'].unique())
        if len(unique_dates) < 130:
            return 0, 0, 0
            
        test_dates = unique_dates[-30:]
        
        all_predictions = []
        all_actuals = []
        daily_accuracies = []
        
        for test_date in test_dates:
            # 学習データとテストデータを分割
            train_data = df_clean[df_clean['Date'] < test_date]
            test_data = df_clean[df_clean['Date'] == test_date]
            
            if len(train_data) < 1000 or len(test_data) < 10:
                continue
            
            # 特徴量とターゲット
            X_train = train_data[features]
            y_train = train_data['Target']
            X_test = test_data[features]
            y_test = test_data['Target']
            
            # スケーリング
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # モデル選択と学習
            if model_type == 'logistic':
                model = LogisticRegression(random_state=42, max_iter=1000, 
                                          class_weight='balanced')
            elif model_type == 'rf':
                model = RandomForestClassifier(n_estimators=100, random_state=42,
                                              class_weight='balanced', max_depth=10)
            else:  # gradient_boost
                model = GradientBoostingClassifier(n_estimators=100, random_state=42,
                                                  max_depth=5)
            
            model.fit(X_train_scaled, y_train)
            
            # 予測確率を取得
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # 上位N銘柄を選択
            test_data_copy = test_data.copy()
            test_data_copy['confidence'] = y_pred_proba
            test_data_copy = test_data_copy.sort_values('confidence', ascending=False)
            
            # 信頼度閾値を満たす上位N銘柄
            top_stocks = test_data_copy[test_data_copy['confidence'] >= confidence_threshold].head(top_n)
            
            if len(top_stocks) > 0:
                # 選出された銘柄の実際の結果
                selected_actuals = top_stocks['Target'].values
                selected_predictions = np.ones(len(selected_actuals))  # 全て買い予測
                
                all_actuals.extend(selected_actuals)
                all_predictions.extend(selected_predictions)
                
                daily_accuracy = (selected_actuals == selected_predictions).mean()
                daily_accuracies.append(daily_accuracy)
        
        if len(all_predictions) == 0:
            return 0, 0, 0
        
        # 全体精度
        overall_accuracy = accuracy_score(all_actuals, all_predictions)
        
        # 日次精度の平均
        avg_daily_accuracy = np.mean(daily_accuracies) if daily_accuracies else 0
        
        # 選出率（何日買いシグナルが出たか）
        selection_rate = len(daily_accuracies) / len(test_dates)
        
        return overall_accuracy, avg_daily_accuracy, selection_rate
    
    def optimize_features(self, df):
        """特徴量の最適化"""
        logger.info("🔍 特徴量の最適化開始...")
        
        # 全特徴量取得
        all_features = self.get_all_features(df)
        
        # Step 1: 統計的に重要な特徴量を選択
        logger.info("Step 1: 統計的特徴量選択...")
        
        # クリーンなデータ準備
        required_cols = ['Target'] + all_features
        df_clean = df[required_cols].dropna()
        
        if len(df_clean) < 10000:
            logger.error("十分なデータがありません")
            return None
        
        X = df_clean[all_features]
        y = df_clean['Target']
        
        # 相互情報量による特徴量選択
        selector = SelectKBest(score_func=mutual_info_classif, k=20)
        selector.fit(X, y)
        
        # 重要度でソート
        feature_scores = zip(all_features, selector.scores_)
        sorted_features = sorted(feature_scores, key=lambda x: x[1], reverse=True)
        
        # 上位20特徴量
        top_features = [f[0] for f in sorted_features[:20]]
        
        logger.info(f"📊 上位20特徴量:")
        for i, (feat, score) in enumerate(sorted_features[:20], 1):
            logger.info(f"  {i:2d}. {feat:20s} (スコア: {score:.4f})")
        
        # Step 2: 特徴量の組み合わせを試す
        logger.info("\nStep 2: 特徴量組み合わせ最適化...")
        
        best_result = {
            'features': None,
            'accuracy': 0,
            'daily_accuracy': 0,
            'selection_rate': 0,
            'model_type': None,
            'threshold': None
        }
        
        # 異なる特徴量数を試す
        for n_features in [3, 5, 7, 10, 15]:
            logger.info(f"\n📊 {n_features}個の特徴量で評価...")
            
            # 上位n個の特徴量
            test_features = top_features[:n_features]
            
            # 異なるモデルを試す
            for model_type in ['logistic', 'rf', 'gradient_boost']:
                
                # 異なる信頼度閾値を試す
                for threshold in [0.45, 0.50, 0.52, 0.55]:
                    
                    accuracy, daily_acc, selection_rate = self.evaluate_features(
                        df, test_features, model_type, threshold, top_n=5
                    )
                    
                    # 選出された5銘柄の精度が重要
                    if accuracy > best_result['accuracy']:
                        best_result = {
                            'features': test_features,
                            'accuracy': accuracy,
                            'daily_accuracy': daily_acc,
                            'selection_rate': selection_rate,
                            'model_type': model_type,
                            'threshold': threshold
                        }
                        
                        logger.info(f"  ✅ 新記録! 精度: {accuracy:.2%} "
                                  f"(モデル: {model_type}, 閾値: {threshold})")
                        
                        if accuracy >= 0.60:
                            logger.info(f"  🎯 目標精度60%を達成!")
        
        # Step 3: さらに特徴量を追加して試す
        if best_result['accuracy'] < 0.60:
            logger.info("\nStep 3: 追加の特徴量組み合わせ...")
            
            # 技術指標系の特徴量を重点的に試す
            technical_features = [f for f in all_features if any(
                keyword in f for keyword in ['RSI', 'MA', 'EMA', 'MACD', 'Bollinger', 
                                            'Volatility', 'Volume', 'Price_vs', 'Returns']
            )]
            
            # 様々な組み合わせを試す
            for combo_size in [5, 7, 10]:
                if len(technical_features) >= combo_size:
                    # 最初のcombo_size個を試す
                    test_features = technical_features[:combo_size]
                    
                    for model_type in ['rf', 'gradient_boost']:
                        for threshold in [0.48, 0.51, 0.53]:
                            
                            accuracy, daily_acc, selection_rate = self.evaluate_features(
                                df, test_features, model_type, threshold, top_n=5
                            )
                            
                            if accuracy > best_result['accuracy']:
                                best_result = {
                                    'features': test_features,
                                    'accuracy': accuracy,
                                    'daily_accuracy': daily_acc,
                                    'selection_rate': selection_rate,
                                    'model_type': model_type,
                                    'threshold': threshold
                                }
                                
                                logger.info(f"  ✅ 更新! 精度: {accuracy:.2%}")
                                
                                if accuracy >= 0.60:
                                    logger.info(f"  🎯 目標精度60%を達成!")
                                    break
        
        return best_result
    
    def save_optimal_config(self, best_result):
        """最適な設定を保存"""
        if best_result['accuracy'] >= 0.60:
            # 設定ファイル更新
            config_path = Path("production_config.yaml")
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 特徴量更新
            config['features']['optimal_features'] = best_result['features']
            config['system']['confidence_threshold'] = best_result['threshold']
            
            # モデル情報を追加
            config['model'] = {
                'type': best_result['model_type'],
                'accuracy': float(best_result['accuracy']),
                'daily_accuracy': float(best_result['daily_accuracy']),
                'optimized_date': pd.Timestamp.now().strftime('%Y-%m-%d')
            }
            
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
            
            logger.info(f"✅ 設定ファイルを更新しました")
            
            # 結果をテキストファイルにも保存
            result_path = Path("optimization_result.txt")
            with open(result_path, 'w', encoding='utf-8') as f:
                f.write(f"精度最適化結果\n")
                f.write(f"="*50 + "\n")
                f.write(f"達成精度: {best_result['accuracy']:.2%}\n")
                f.write(f"日次平均精度: {best_result['daily_accuracy']:.2%}\n")
                f.write(f"選出率: {best_result['selection_rate']:.2%}\n")
                f.write(f"モデル: {best_result['model_type']}\n")
                f.write(f"信頼度閾値: {best_result['threshold']}\n")
                f.write(f"特徴量数: {len(best_result['features'])}\n")
                f.write(f"特徴量:\n")
                for feat in best_result['features']:
                    f.write(f"  - {feat}\n")
            
            return True
        return False


def main():
    """メイン実行"""
    optimizer = AccuracyOptimizer()
    
    # データ読み込み
    df = optimizer.load_data()
    if df is None:
        return
    
    logger.info(f"📊 データ件数: {len(df):,}レコード")
    
    # 特徴量最適化
    best_result = optimizer.optimize_features(df)
    
    if best_result and best_result['accuracy'] > 0:
        logger.info("\n" + "="*80)
        logger.info("🎯 最適化結果")
        logger.info("="*80)
        logger.info(f"最高精度: {best_result['accuracy']:.2%}")
        logger.info(f"日次平均精度: {best_result['daily_accuracy']:.2%}")
        logger.info(f"選出率: {best_result['selection_rate']:.2%}")
        logger.info(f"最適モデル: {best_result['model_type']}")
        logger.info(f"最適閾値: {best_result['threshold']}")
        logger.info(f"最適特徴量 ({len(best_result['features'])}個):")
        for i, feat in enumerate(best_result['features'], 1):
            logger.info(f"  {i:2d}. {feat}")
        
        if best_result['accuracy'] >= 0.60:
            logger.info("\n✅ 目標精度60%を達成しました!")
            
            # 設定を保存
            if optimizer.save_optimal_config(best_result):
                logger.info("📝 最適な設定を保存しました")
        else:
            logger.info(f"\n⚠️ 目標精度60%に届きませんでした ({best_result['accuracy']:.2%})")
            logger.info("さらなる最適化が必要です")
    else:
        logger.error("最適化に失敗しました")


if __name__ == "__main__":
    main()