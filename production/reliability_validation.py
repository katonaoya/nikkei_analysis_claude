#!/usr/bin/env python3
"""
実運用信頼性検証システム
83.33%精度の実運用再現性を徹底検証
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import lightgbm as lgb
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, confusion_matrix
from yahoo_market_data import YahooMarketData
import warnings
warnings.filterwarnings('ignore')

from loguru import logger

class ReliabilityValidator:
    """信頼性検証クラス"""
    
    def __init__(self):
        self.base_data_file = "data/processed/integrated_with_external.parquet"
        self.validation_results = {}
        
    def load_validation_data(self) -> pd.DataFrame:
        """検証用データ準備"""
        logger.info("🔍 信頼性検証データ準備中...")
        
        # ベースデータ
        df = pd.read_parquet(self.base_data_file)
        
        if 'date' in df.columns:
            df['Date'] = pd.to_datetime(df['date'])
        if 'code' in df.columns:
            df['Stock'] = df['code'].astype(str)
        
        # 高品質銘柄選択（実運用相当）
        stock_counts = df['Stock'].value_counts()
        reliable_stocks = stock_counts[stock_counts >= 400].head(150).index.tolist()
        df = df[df['Stock'].isin(reliable_stocks)].copy()
        
        logger.info(f"検証対象銘柄: {len(reliable_stocks)}銘柄")
        
        # マーケットデータ統合（成功パターン再現）
        market_data = YahooMarketData()
        data_dict = market_data.get_all_market_data(period="2y")
        
        if data_dict:
            market_features = market_data.calculate_market_features(data_dict)
            if not market_features.empty:
                # 日付統一（UTCで変換してからdateに変換）
                df['Date'] = pd.to_datetime(df['Date']).dt.date
                market_features['Date'] = pd.to_datetime(market_features['Date'], utc=True).dt.date
                
                try:
                    df = df.merge(market_features, on='Date', how='left')
                    market_cols = [col for col in market_features.columns if col != 'Date']
                    df[market_cols] = df[market_cols].fillna(method='ffill').fillna(0)
                    logger.success("✅ マーケットデータ統合完了")
                except:
                    logger.warning("マーケットデータ統合をスキップ")
        
        return df
    
    def create_validation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """成功パターンの特徴量再現"""
        logger.info("🔧 検証用特徴量生成中...")
        
        enhanced_df = df.copy()
        enhanced_df = enhanced_df.sort_values(['Stock', 'Date'])
        
        # ターゲット（元の条件：翌日1%以上上昇）
        enhanced_df['next_high'] = enhanced_df.groupby('Stock')['high'].shift(-1)
        enhanced_df['Target'] = (enhanced_df['next_high'] > enhanced_df['close'] * 1.01).astype(int)
        
        # 成功時の特徴量再現
        for stock, stock_df in enhanced_df.groupby('Stock'):
            stock_mask = enhanced_df['Stock'] == stock
            stock_data = enhanced_df[stock_mask].sort_values('Date')
            
            if len(stock_data) < 50:
                continue
            
            # RSI
            delta = stock_data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, 1)
            enhanced_df.loc[stock_mask, 'Enhanced_RSI'] = 100 - (100 / (1 + rs))
            
            # 移動平均
            enhanced_df.loc[stock_mask, 'MA_Cross_Signal'] = (
                stock_data['close'].rolling(5).mean() > stock_data['close'].rolling(20).mean()
            ).astype(int)
            
            # ボラティリティ
            returns = stock_data['close'].pct_change()
            enhanced_df.loc[stock_mask, 'Enhanced_Volatility'] = returns.rolling(20).std()
            
            # 出来高
            volume_ma = stock_data['volume'].rolling(20).mean()
            enhanced_df.loc[stock_mask, 'Enhanced_Volume_Ratio'] = stock_data['volume'] / volume_ma
        
        enhanced_df = enhanced_df.fillna(method='ffill').fillna(0)
        enhanced_df = enhanced_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        logger.success("✅ 検証用特徴量生成完了")
        return enhanced_df
    
    def time_series_validation(self, df: pd.DataFrame) -> dict:
        """時系列信頼性検証（実運用シミュレーション）"""
        logger.info("📊 時系列信頼性検証実行中...")
        
        df_sorted = df.sort_values(['Stock', 'Date'])
        unique_dates = sorted(df_sorted['Date'].unique())
        
        # 複数期間での検証
        validation_periods = [
            ('最新10日', unique_dates[-10:]),
            ('最新20日', unique_dates[-20:]),
            ('最新30日', unique_dates[-30:]),
            ('最新60日', unique_dates[-60:])
        ]
        
        results = {}
        
        # 成功パターンの特徴量
        feature_cols = [col for col in df.columns if col not in ['Date', 'Stock', 'Target', 'next_high']]
        feature_cols = [col for col in feature_cols if df[col].dtype in ['int64', 'float64']][:20]  # 上位20特徴量
        
        for period_name, test_dates in validation_periods:
            logger.info(f"  検証中: {period_name}")
            
            daily_precisions = []
            all_predictions = []
            all_actuals = []
            confidence_scores = []
            
            for test_date in test_dates:
                # 訓練データ（テスト日より前）
                train_data = df_sorted[df_sorted['Date'] < test_date]
                test_data = df_sorted[df_sorted['Date'] == test_date]
                
                train_clean = train_data.dropna(subset=['Target'] + feature_cols)
                test_clean = test_data.dropna(subset=['Target'] + feature_cols)
                
                if len(train_clean) < 500 or len(test_clean) < 3:
                    continue
                
                X_train = train_clean[feature_cols]
                y_train = train_clean['Target']
                X_test = test_clean[feature_cols]
                y_test = test_clean['Target']
                
                # 特徴量選択（成功パターン再現）
                selector = SelectKBest(score_func=f_classif, k=min(12, len(feature_cols)))
                X_train_selected = selector.fit_transform(X_train, y_train)
                X_test_selected = selector.transform(X_test)
                
                # スケーリング
                scaler = RobustScaler()
                X_train_scaled = scaler.fit_transform(X_train_selected)
                X_test_scaled = scaler.transform(X_test_selected)
                
                # 成功モデル再現
                model = lgb.LGBMClassifier(
                    n_estimators=150,
                    max_depth=4,
                    learning_rate=0.08,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    random_state=42,
                    verbose=-1
                )
                
                model.fit(X_train_scaled, y_train)
                probs = model.predict_proba(X_test_scaled)[:, 1]
                
                # 上位3銘柄選択（成功パターン）
                n_select = min(3, len(probs))
                top_indices = np.argsort(probs)[-n_select:]
                
                selected_actuals = y_test.iloc[top_indices].values
                selected_probs = probs[top_indices]
                
                all_predictions.extend([1] * len(selected_actuals))
                all_actuals.extend(selected_actuals)
                confidence_scores.extend(selected_probs)
                
                # 日別精度
                if len(selected_actuals) > 0:
                    daily_precision = sum(selected_actuals) / len(selected_actuals)
                    daily_precisions.append(daily_precision)
            
            # 期間別結果
            if all_predictions:
                overall_precision = sum(all_actuals) / len(all_actuals)
                precision_std = np.std(daily_precisions) if daily_precisions else 0
                avg_confidence = np.mean(confidence_scores)
                consistency = (np.array(daily_precisions) >= 0.5).mean() if daily_precisions else 0
                
                results[period_name] = {
                    'precision': overall_precision,
                    'precision_std': precision_std,
                    'avg_confidence': avg_confidence,
                    'consistency_rate': consistency,
                    'total_selections': len(all_predictions),
                    'daily_precisions': daily_precisions
                }
        
        return results
    
    def robustness_validation(self, df: pd.DataFrame) -> dict:
        """頑健性検証（市場環境別性能）"""
        logger.info("🛡️ 頑健性検証実行中...")
        
        df_sorted = df.sort_values(['Stock', 'Date'])
        
        # VIX水準別性能検証
        vix_scenarios = []
        if 'vix_close' in df.columns:
            df['vix_regime'] = pd.cut(df['vix_close'], 
                                    bins=[0, 20, 30, 100], 
                                    labels=['低VIX', '中VIX', '高VIX'])
            vix_scenarios = ['低VIX', '中VIX', '高VIX']
        
        # 市場トレンド別性能検証  
        market_scenarios = []
        if 'nikkei225_return_1d' in df.columns:
            df['market_trend'] = pd.cut(df['nikkei225_return_1d'], 
                                      bins=[-np.inf, -0.01, 0.01, np.inf], 
                                      labels=['下落', '横ばい', '上昇'])
            market_scenarios = ['下落', '横ばい', '上昇']
        
        scenarios = vix_scenarios + market_scenarios
        scenario_results = {}
        
        feature_cols = [col for col in df.columns if col not in ['Date', 'Stock', 'Target', 'next_high']]
        feature_cols = [col for col in feature_cols if df[col].dtype in ['int64', 'float64']][:15]
        
        for scenario in scenarios:
            if scenario in ['低VIX', '中VIX', '高VIX']:
                scenario_data = df[df['vix_regime'] == scenario]
            else:
                scenario_data = df[df['market_trend'] == scenario]
            
            if len(scenario_data) < 100:
                continue
            
            logger.info(f"  検証中: {scenario}環境")
            
            # 直近テスト
            unique_dates = sorted(scenario_data['Date'].unique())
            test_dates = unique_dates[-min(10, len(unique_dates)):]
            
            all_preds = []
            all_actuals = []
            
            for test_date in test_dates:
                train_data = scenario_data[scenario_data['Date'] < test_date]
                test_data = scenario_data[scenario_data['Date'] == test_date]
                
                train_clean = train_data.dropna(subset=['Target'] + feature_cols)
                test_clean = test_data.dropna(subset=['Target'] + feature_cols)
                
                if len(train_clean) < 100 or len(test_clean) < 2:
                    continue
                
                X_train = train_clean[feature_cols]
                y_train = train_clean['Target']
                X_test = test_clean[feature_cols]
                y_test = test_clean['Target']
                
                try:
                    scaler = RobustScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    model = lgb.LGBMClassifier(
                        n_estimators=100,
                        max_depth=4,
                        learning_rate=0.08,
                        random_state=42,
                        verbose=-1
                    )
                    
                    model.fit(X_train_scaled, y_train)
                    probs = model.predict_proba(X_test_scaled)[:, 1]
                    
                    # 上位2銘柄
                    n_select = min(2, len(probs))
                    top_indices = np.argsort(probs)[-n_select:]
                    
                    selected_actuals = y_test.iloc[top_indices].values
                    all_preds.extend([1] * len(selected_actuals))
                    all_actuals.extend(selected_actuals)
                    
                except Exception as e:
                    continue
            
            if all_preds:
                scenario_precision = sum(all_actuals) / len(all_actuals)
                scenario_results[scenario] = {
                    'precision': scenario_precision,
                    'sample_size': len(all_preds)
                }
        
        return scenario_results
    
    def statistical_validation(self, df: pd.DataFrame) -> dict:
        """統計的信頼性検証"""
        logger.info("📈 統計的信頼性検証実行中...")
        
        # ブートストラップ検証（リサンプリング）
        df_sorted = df.sort_values(['Stock', 'Date'])
        unique_dates = sorted(df_sorted['Date'].unique())
        test_dates = unique_dates[-20:]  # 最新20日
        
        bootstrap_precisions = []
        feature_cols = [col for col in df.columns if col not in ['Date', 'Stock', 'Target', 'next_high']]
        feature_cols = [col for col in feature_cols if df[col].dtype in ['int64', 'float64']][:15]
        
        # 10回ブートストラップ
        for bootstrap_iter in range(10):
            # ランダムサンプリング（復元抽出）
            sampled_dates = np.random.choice(test_dates, size=len(test_dates), replace=True)
            
            all_preds = []
            all_actuals = []
            
            for test_date in sampled_dates:
                train_data = df_sorted[df_sorted['Date'] < test_date]
                test_data = df_sorted[df_sorted['Date'] == test_date]
                
                train_clean = train_data.dropna(subset=['Target'] + feature_cols)
                test_clean = test_data.dropna(subset=['Target'] + feature_cols)
                
                if len(train_clean) < 500 or len(test_clean) < 2:
                    continue
                
                try:
                    X_train = train_clean[feature_cols]
                    y_train = train_clean['Target']
                    X_test = test_clean[feature_cols]
                    y_test = test_clean['Target']
                    
                    scaler = RobustScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    model = lgb.LGBMClassifier(
                        n_estimators=150,
                        max_depth=4,
                        learning_rate=0.08,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        random_state=42 + bootstrap_iter,
                        verbose=-1
                    )
                    
                    model.fit(X_train_scaled, y_train)
                    probs = model.predict_proba(X_test_scaled)[:, 1]
                    
                    n_select = min(3, len(probs))
                    top_indices = np.argsort(probs)[-n_select:]
                    
                    selected_actuals = y_test.iloc[top_indices].values
                    all_preds.extend([1] * len(selected_actuals))
                    all_actuals.extend(selected_actuals)
                    
                except Exception as e:
                    continue
            
            if all_preds:
                bootstrap_precision = sum(all_actuals) / len(all_actuals)
                bootstrap_precisions.append(bootstrap_precision)
        
        # 統計分析
        if bootstrap_precisions:
            mean_precision = np.mean(bootstrap_precisions)
            precision_std = np.std(bootstrap_precisions)
            confidence_95_lower = np.percentile(bootstrap_precisions, 2.5)
            confidence_95_upper = np.percentile(bootstrap_precisions, 97.5)
            
            return {
                'bootstrap_mean': mean_precision,
                'bootstrap_std': precision_std,
                'confidence_95_lower': confidence_95_lower,
                'confidence_95_upper': confidence_95_upper,
                'precision_distribution': bootstrap_precisions
            }
        
        return {}
    
    def run_comprehensive_validation(self):
        """包括的信頼性検証実行"""
        logger.info("🔍 包括的信頼性検証開始")
        print("🎯 83.33%精度の実運用信頼性検証")
        print("="*60)
        
        # データ準備
        df = self.load_validation_data()
        if df.empty:
            print("❌ 検証データの準備に失敗")
            return False
        
        # 特徴量生成
        df_enhanced = self.create_validation_features(df)
        
        # データ品質確認
        target_rate = df_enhanced['Target'].mean()
        print(f"📊 検証データ概要:")
        print(f"  データ件数: {len(df_enhanced):,}")
        print(f"  対象銘柄数: {df_enhanced['Stock'].nunique()}")
        print(f"  ターゲット陽性率: {target_rate:.2%}")
        print()
        
        # 1. 時系列信頼性検証
        print("📊 1. 時系列信頼性検証")
        print("-"*40)
        time_results = self.time_series_validation(df_enhanced)
        
        baseline_precision = 0.8333  # 成功時の精度
        reliable_periods = 0
        
        for period, result in time_results.items():
            precision = result['precision']
            precision_std = result['precision_std']
            consistency = result['consistency_rate']
            
            # 信頼性判定
            if precision >= 0.75:  # 75%以上
                reliability = "🟢 高信頼性"
                if precision >= baseline_precision * 0.9:  # 90%以上の再現率
                    reliable_periods += 1
            elif precision >= 0.65:  # 65%以上
                reliability = "🟡 中信頼性"
            else:
                reliability = "🔴 低信頼性"
            
            print(f"{period:>8}: {precision:6.1%} (±{precision_std:.1%}) "
                  f"一貫性:{consistency:4.1%} {reliability}")
        
        # 2. 頑健性検証
        print(f"\\n🛡️ 2. 市場環境別頑健性検証")
        print("-"*40)
        robustness_results = self.robustness_validation(df_enhanced)
        
        robust_scenarios = 0
        for scenario, result in robustness_results.items():
            precision = result['precision']
            sample_size = result['sample_size']
            
            if precision >= 0.70:
                robustness = "🟢 頑健"
                robust_scenarios += 1
            elif precision >= 0.60:
                robustness = "🟡 やや頑健"
            else:
                robustness = "🔴 不安定"
            
            print(f"{scenario:>8}: {precision:6.1%} (n={sample_size:3d}) {robustness}")
        
        # 3. 統計的信頼性検証
        print(f"\\n📈 3. 統計的信頼性検証（ブートストラップ）")
        print("-"*40)
        stats_results = self.statistical_validation(df_enhanced)
        
        if stats_results:
            mean_precision = stats_results['bootstrap_mean']
            precision_std = stats_results['bootstrap_std']
            conf_lower = stats_results['confidence_95_lower']
            conf_upper = stats_results['confidence_95_upper']
            
            print(f"平均精度: {mean_precision:.1%} (±{precision_std:.1%})")
            print(f"95%信頼区間: [{conf_lower:.1%}, {conf_upper:.1%}]")
            
            # 信頼区間が60%以上に含まれるかチェック
            confidence_60_above = conf_lower >= 0.60
            if confidence_60_above:
                statistical_reliability = "🟢 統計的に高信頼"
            else:
                statistical_reliability = "🟡 統計的に要注意"
            
            print(f"統計的信頼性: {statistical_reliability}")
        
        # 総合判定
        print(f"\\n🎯 総合信頼性評価")
        print("="*60)
        
        # 評価基準
        total_checks = 4
        passed_checks = 0
        
        # チェック1: 時系列安定性
        stable_ratio = reliable_periods / len(time_results) if time_results else 0
        if stable_ratio >= 0.5:  # 50%以上の期間で安定
            print("✅ 時系列安定性: 合格（複数期間で75%以上維持）")
            passed_checks += 1
        else:
            print("❌ 時系列安定性: 不合格（安定性不足）")
        
        # チェック2: 市場環境耐性
        robust_ratio = robust_scenarios / len(robustness_results) if robustness_results else 0
        if robust_ratio >= 0.6:  # 60%以上の環境で頑健
            print("✅ 市場環境耐性: 合格（多様な環境で70%以上）")
            passed_checks += 1
        else:
            print("❌ 市場環境耐性: 不合格（環境依存性大）")
        
        # チェック3: 統計的信頼性
        if stats_results and stats_results.get('confidence_95_lower', 0) >= 0.65:
            print("✅ 統計的信頼性: 合格（95%信頼区間で65%以上）")
            passed_checks += 1
        else:
            print("❌ 統計的信頼性: 不合格（信頼区間が低い）")
        
        # チェック4: 実用性
        if time_results:
            recent_precision = time_results.get('最新10日', {}).get('precision', 0)
            if recent_precision >= 0.70:
                print("✅ 実用性: 合格（直近10日で70%以上）")
                passed_checks += 1
            else:
                print("❌ 実用性: 不合格（直近性能低下）")
        
        # 最終判定
        reliability_score = passed_checks / total_checks
        
        print(f"\\n📊 【最終信頼性評価】")
        print(f"信頼性スコア: {passed_checks}/{total_checks} ({reliability_score:.1%})")
        
        if reliability_score >= 0.75:
            final_verdict = "🟢 高信頼性 - 実運用推奨"
            recommendation = "実運用に適用可能です"
        elif reliability_score >= 0.5:
            final_verdict = "🟡 中信頼性 - 要注意運用"
            recommendation = "リスク管理を強化して運用"
        else:
            final_verdict = "🔴 低信頼性 - 運用非推奨"
            recommendation = "さらなる改善が必要"
        
        print(f"総合判定: {final_verdict}")
        print(f"推奨事項: {recommendation}")
        
        # 信頼性レポート保存
        with open('reliability_validation_report.txt', 'w') as f:
            f.write("実運用信頼性検証レポート\\n")
            f.write("="*50 + "\\n\\n")
            f.write(f"検証日時: {datetime.now()}\\n")
            f.write(f"対象精度: 83.33%\\n")
            f.write(f"信頼性スコア: {reliability_score:.1%}\\n")
            f.write(f"最終判定: {final_verdict}\\n")
            f.write(f"推奨事項: {recommendation}\\n\\n")
            
            f.write("詳細結果:\\n")
            f.write(f"時系列安定性: {stable_ratio:.1%}の期間で安定\\n")
            f.write(f"市場環境耐性: {robust_ratio:.1%}の環境で頑健\\n")
            if stats_results:
                f.write(f"統計的信頼区間: [{stats_results['confidence_95_lower']:.1%}, {stats_results['confidence_95_upper']:.1%}]\\n")
        
        print("\\n💾 詳細レポート保存: reliability_validation_report.txt")
        
        return reliability_score >= 0.5

# 実行
if __name__ == "__main__":
    validator = ReliabilityValidator()
    reliable = validator.run_comprehensive_validation()
    
    if reliable:
        print("\\n🎉 実運用に適した信頼性が確認されました！")
    else:
        print("\\n⚠️ 実運用前にさらなる検証・改善が推奨されます")