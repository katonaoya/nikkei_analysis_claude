#!/usr/bin/env python3
"""
包括的な候補絞り込み手法の評価
90銘柄 → 5銘柄への全パターンテスト
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# ログ設定
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", level="INFO")

class ComprehensiveFilteringEvaluation:
    """包括的候補絞り込み手法評価"""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.processed_dir = self.data_dir / "processed"
        
        # 最適特徴量
        self.optimal_features = [
            'Market_Breadth', 'Market_Return', 'Volatility_20', 'Price_vs_MA20',
            'sp500_change', 'vix_change', 'nikkei_change', 'us_10y_change', 'usd_jpy_change'
        ]
        
        # 評価パラメータ
        self.initial_candidates = 90  # 初期候補数（概算）
        self.target_candidates = 5    # 最終候補数
        self.confidence_threshold = 0.55
        
    def load_and_prepare_data(self):
        """データ読み込みと準備"""
        logger.info("📊 包括的評価用データ準備...")
        
        integrated_file = self.processed_dir / "integrated_with_external.parquet"
        df = pd.read_parquet(integrated_file)
        
        # データクリーニング
        clean_df = df[df['Binary_Direction'].notna()].copy()
        clean_df = clean_df.sort_values(['Date', 'Code']).reset_index(drop=True)
        
        # 追加指標計算
        clean_df = self.calculate_additional_metrics(clean_df)
        
        # 特徴量とターゲット準備
        X = clean_df[self.optimal_features].fillna(0)
        y = clean_df['Binary_Direction'].astype(int)
        
        logger.info(f"✅ データ準備完了: {len(clean_df):,}件, {len(self.optimal_features)}特徴量")
        
        return clean_df, X, y
    
    def calculate_additional_metrics(self, df):
        """追加指標の計算"""
        logger.info("🔧 追加指標計算...")
        
        df = df.sort_values(['Code', 'Date']).copy()
        
        # 基本的な追加指標
        df['Volume_MA5'] = df.groupby('Code')['Volume'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
        df['Price_Change_5d'] = df.groupby('Code')['Close'].pct_change(5).fillna(0)
        df['RSI'] = self.calculate_rsi(df)
        
        # セクター情報（仮想的に生成）
        np.random.seed(42)
        unique_codes = df['Code'].unique()
        sectors = ['Technology', 'Finance', 'Healthcare', 'Consumer', 'Industrial', 'Materials', 'Energy', 'Utilities']
        sector_mapping = {code: np.random.choice(sectors) for code in unique_codes}
        df['Sector'] = df['Code'].map(sector_mapping)
        
        # 時価総額（仮想的に計算）
        df['Market_Cap'] = df['Close'] * np.random.uniform(1000000, 100000000, size=len(df))
        
        return df
    
    def calculate_rsi(self, df, period=14):
        """RSI計算"""
        delta = df.groupby('Code')['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def get_base_predictions(self, df, X, y, test_start_date):
        """基本予測の取得"""
        train_mask = df['Date'] < test_start_date
        test_mask = df['Date'] >= test_start_date
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_test = X[test_mask]
        
        scaler = StandardScaler()
        model = LogisticRegression(C=0.001, class_weight='balanced', random_state=42, max_iter=1000)
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model.fit(X_train_scaled, y_train)
        pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        return pred_proba, test_mask
    
    def filter_method_1_confidence(self, day_data, n_candidates=5):
        """手法1: 単純確信度上位選択"""
        if 'pred_proba' not in day_data.columns:
            return []
            
        # 高確信度のみ（上昇・下落両方向）
        high_conf_up = day_data[day_data['pred_proba'] >= self.confidence_threshold]
        high_conf_down = day_data[day_data['pred_proba'] <= (1 - self.confidence_threshold)]
        
        # 確信度の絶対値で評価
        high_conf_up = high_conf_up.copy()
        high_conf_down = high_conf_down.copy()
        high_conf_up['abs_confidence'] = high_conf_up['pred_proba']
        high_conf_down['abs_confidence'] = 1 - high_conf_down['pred_proba']
        
        all_high_conf = pd.concat([high_conf_up, high_conf_down])
        
        if len(all_high_conf) == 0:
            return []
            
        selected = all_high_conf.nlargest(n_candidates, 'abs_confidence')
        return selected['Code'].tolist()
    
    def filter_method_2_sector_diversity(self, day_data, n_candidates=5):
        """手法2: セクター分散 + 確信度"""
        if 'pred_proba' not in day_data.columns or 'Sector' not in day_data.columns:
            return []
        
        high_conf = day_data[
            (day_data['pred_proba'] >= self.confidence_threshold) | 
            (day_data['pred_proba'] <= (1 - self.confidence_threshold))
        ].copy()
        
        if len(high_conf) == 0:
            return []
        
        high_conf['abs_confidence'] = np.maximum(high_conf['pred_proba'], 1 - high_conf['pred_proba'])
        
        # セクター別に最高確信度を1つずつ選択
        selected = []
        sector_groups = high_conf.groupby('Sector')
        
        for sector, group in sector_groups:
            best_in_sector = group.loc[group['abs_confidence'].idxmax()]
            selected.append(best_in_sector)
            
            if len(selected) >= n_candidates:
                break
        
        # 不足分は全体から追加
        if len(selected) < n_candidates:
            remaining_codes = [s['Code'] for s in selected]
            remaining_data = high_conf[~high_conf['Code'].isin(remaining_codes)]
            additional = remaining_data.nlargest(n_candidates - len(selected), 'abs_confidence')
            selected.extend(additional.to_dict('records'))
        
        return [s['Code'] if isinstance(s, dict) else s.name for s in selected[:n_candidates]]
    
    def filter_method_3_risk_adjusted(self, day_data, n_candidates=5):
        """手法3: リスク調整 (ボラティリティ + 確信度)"""
        if 'pred_proba' not in day_data.columns or 'Volatility_20' not in day_data.columns:
            return []
        
        high_conf = day_data[
            (day_data['pred_proba'] >= self.confidence_threshold) | 
            (day_data['pred_proba'] <= (1 - self.confidence_threshold))
        ].copy()
        
        if len(high_conf) == 0:
            return []
        
        high_conf['abs_confidence'] = np.maximum(high_conf['pred_proba'], 1 - high_conf['pred_proba'])
        
        # リスク調整スコア = 確信度 / ボラティリティ
        high_conf['risk_adjusted_score'] = high_conf['abs_confidence'] / (high_conf['Volatility_20'] + 0.01)
        
        selected = high_conf.nlargest(n_candidates, 'risk_adjusted_score')
        return selected['Code'].tolist()
    
    def filter_method_4_momentum(self, day_data, n_candidates=5):
        """手法4: モメンタム + 確信度"""
        if 'pred_proba' not in day_data.columns or 'Price_Change_5d' not in day_data.columns:
            return []
        
        high_conf = day_data[
            (day_data['pred_proba'] >= self.confidence_threshold) | 
            (day_data['pred_proba'] <= (1 - self.confidence_threshold))
        ].copy()
        
        if len(high_conf) == 0:
            return []
        
        high_conf['abs_confidence'] = np.maximum(high_conf['pred_proba'], 1 - high_conf['pred_proba'])
        
        # 予測方向とモメンタムの一致度
        high_conf['predicted_up'] = high_conf['pred_proba'] > 0.5
        high_conf['momentum_up'] = high_conf['Price_Change_5d'] > 0
        high_conf['momentum_alignment'] = (high_conf['predicted_up'] == high_conf['momentum_up']).astype(float)
        
        # モメンタム調整スコア
        high_conf['momentum_score'] = high_conf['abs_confidence'] * (1 + high_conf['momentum_alignment'])
        
        selected = high_conf.nlargest(n_candidates, 'momentum_score')
        return selected['Code'].tolist()
    
    def filter_method_5_liquidity(self, day_data, n_candidates=5):
        """手法5: 流動性 + 確信度"""
        if 'pred_proba' not in day_data.columns or 'Volume_MA5' not in day_data.columns:
            return []
        
        high_conf = day_data[
            (day_data['pred_proba'] >= self.confidence_threshold) | 
            (day_data['pred_proba'] <= (1 - self.confidence_threshold))
        ].copy()
        
        if len(high_conf) == 0:
            return []
        
        high_conf['abs_confidence'] = np.maximum(high_conf['pred_proba'], 1 - high_conf['pred_proba'])
        
        # 流動性フィルター（出来高上位50%のみ）
        volume_threshold = high_conf['Volume_MA5'].quantile(0.5)
        high_liquidity = high_conf[high_conf['Volume_MA5'] >= volume_threshold]
        
        if len(high_liquidity) < n_candidates:
            high_liquidity = high_conf  # フィルターが厳しすぎる場合は全体から選択
        
        selected = high_liquidity.nlargest(n_candidates, 'abs_confidence')
        return selected['Code'].tolist()
    
    def filter_method_6_technical(self, day_data, n_candidates=5):
        """手法6: テクニカル指標 + 確信度"""
        if 'pred_proba' not in day_data.columns or 'RSI' not in day_data.columns:
            return []
        
        high_conf = day_data[
            (day_data['pred_proba'] >= self.confidence_threshold) | 
            (day_data['pred_proba'] <= (1 - self.confidence_threshold))
        ].copy()
        
        if len(high_conf) == 0:
            return []
        
        high_conf['abs_confidence'] = np.maximum(high_conf['pred_proba'], 1 - high_conf['pred_proba'])
        
        # RSIベースの技術的評価
        high_conf['predicted_up'] = high_conf['pred_proba'] > 0.5
        high_conf['rsi_signal'] = 0.0
        
        # 上昇予測 + RSI oversold
        high_conf.loc[(high_conf['predicted_up'] == True) & (high_conf['RSI'] < 30), 'rsi_signal'] = 1.0
        # 下落予測 + RSI overbought  
        high_conf.loc[(high_conf['predicted_up'] == False) & (high_conf['RSI'] > 70), 'rsi_signal'] = 1.0
        # 適正範囲
        high_conf.loc[(high_conf['RSI'] >= 30) & (high_conf['RSI'] <= 70), 'rsi_signal'] = 0.5
        
        high_conf['technical_score'] = high_conf['abs_confidence'] * (1 + high_conf['rsi_signal'])
        
        selected = high_conf.nlargest(n_candidates, 'technical_score')
        return selected['Code'].tolist()
    
    def filter_method_7_hybrid(self, day_data, n_candidates=5):
        """手法7: ハイブリッド（複数要素統合）"""
        if 'pred_proba' not in day_data.columns:
            return []
        
        required_cols = ['Volatility_20', 'Volume_MA5', 'Price_Change_5d', 'RSI', 'Sector']
        if not all(col in day_data.columns for col in required_cols):
            return self.filter_method_1_confidence(day_data, n_candidates)
        
        high_conf = day_data[
            (day_data['pred_proba'] >= self.confidence_threshold) | 
            (day_data['pred_proba'] <= (1 - self.confidence_threshold))
        ].copy()
        
        if len(high_conf) == 0:
            return []
        
        # 複数スコアの統合
        high_conf['abs_confidence'] = np.maximum(high_conf['pred_proba'], 1 - high_conf['pred_proba'])
        
        # 正規化
        high_conf['conf_norm'] = (high_conf['abs_confidence'] - high_conf['abs_confidence'].min()) / (high_conf['abs_confidence'].max() - high_conf['abs_confidence'].min() + 1e-8)
        high_conf['vol_norm'] = 1 - (high_conf['Volatility_20'] - high_conf['Volatility_20'].min()) / (high_conf['Volatility_20'].max() - high_conf['Volatility_20'].min() + 1e-8)  # 低ボラティリティが高スコア
        high_conf['volume_norm'] = (high_conf['Volume_MA5'] - high_conf['Volume_MA5'].min()) / (high_conf['Volume_MA5'].max() - high_conf['Volume_MA5'].min() + 1e-8)
        
        # ハイブリッドスコア
        high_conf['hybrid_score'] = (0.5 * high_conf['conf_norm'] + 
                                    0.2 * high_conf['vol_norm'] + 
                                    0.2 * high_conf['volume_norm'] + 
                                    0.1 * (50 - np.abs(high_conf['RSI'] - 50)) / 50)  # RSI中立が高スコア
        
        # セクター分散を考慮
        selected_codes = []
        selected_sectors = set()
        
        for _, stock in high_conf.sort_values('hybrid_score', ascending=False).iterrows():
            if len(selected_codes) >= n_candidates:
                break
            
            # 同じセクターは最大2銘柄まで
            sector_count = sum(1 for code in selected_codes if day_data[day_data['Code'] == code]['Sector'].iloc[0] == stock['Sector'])
            
            if sector_count < 2:
                selected_codes.append(stock['Code'])
                selected_sectors.add(stock['Sector'])
        
        # 不足分は制約なしで追加
        if len(selected_codes) < n_candidates:
            remaining = high_conf[~high_conf['Code'].isin(selected_codes)]
            additional = remaining.nlargest(n_candidates - len(selected_codes), 'hybrid_score')
            selected_codes.extend(additional['Code'].tolist())
        
        return selected_codes[:n_candidates]
    
    def evaluate_all_methods(self, df, X, y):
        """全手法の評価"""
        logger.info("🧪 全絞り込み手法の包括的評価...")
        
        # 評価期間設定
        dates = sorted(df['Date'].unique())
        test_start_idx = int(len(dates) * 0.8)
        test_start_date = dates[test_start_idx]
        test_dates = dates[test_start_idx:]
        
        logger.info(f"評価期間: {test_start_date} - {dates[-1]} ({len(test_dates)}日)")
        
        # 基本予測取得
        pred_proba, test_mask = self.get_base_predictions(df, X, y, test_start_date)
        test_df = df[test_mask].copy()
        test_df['pred_proba'] = pred_proba
        
        # 手法定義
        methods = {
            'Method1_Confidence': self.filter_method_1_confidence,
            'Method2_SectorDiversity': self.filter_method_2_sector_diversity,
            'Method3_RiskAdjusted': self.filter_method_3_risk_adjusted,
            'Method4_Momentum': self.filter_method_4_momentum,
            'Method5_Liquidity': self.filter_method_5_liquidity,
            'Method6_Technical': self.filter_method_6_technical,
            'Method7_Hybrid': self.filter_method_7_hybrid
        }
        
        # 各手法を評価
        method_results = {}
        
        for method_name, method_func in methods.items():
            logger.info(f"  📊 {method_name} 評価中...")
            
            daily_results = []
            total_predictions = 0
            correct_predictions = 0
            selected_count = 0
            
            for date in test_dates[:100]:  # 最初の100日で評価（計算時間短縮）
                day_data = test_df[test_df['Date'] == date]
                if len(day_data) == 0:
                    continue
                
                # 手法適用
                selected_codes = method_func(day_data, self.target_candidates)
                
                if len(selected_codes) == 0:
                    continue
                
                # 選択された銘柄の評価
                selected_data = day_data[day_data['Code'].isin(selected_codes)]
                
                for _, stock in selected_data.iterrows():
                    prediction = stock['pred_proba'] > 0.5
                    actual = stock['Binary_Direction'] == 1
                    
                    total_predictions += 1
                    if prediction == actual:
                        correct_predictions += 1
                
                selected_count += len(selected_codes)
                
                daily_results.append({
                    'date': date,
                    'selected_count': len(selected_codes),
                    'avg_confidence': np.maximum(selected_data['pred_proba'], 1 - selected_data['pred_proba']).mean(),
                    'predictions': len(selected_data),
                })
            
            # 結果集計
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            avg_daily_selections = selected_count / len(daily_results) if daily_results else 0
            avg_confidence = np.mean([r['avg_confidence'] for r in daily_results if not np.isnan(r['avg_confidence'])]) if daily_results else 0
            
            method_results[method_name] = {
                'accuracy': accuracy,
                'total_predictions': total_predictions,
                'correct_predictions': correct_predictions,
                'avg_daily_selections': avg_daily_selections,
                'avg_confidence': avg_confidence,
                'evaluation_days': len(daily_results)
            }
        
        return method_results
    
    def display_evaluation_results(self, results):
        """評価結果表示"""
        logger.info("\\n" + "="*120)
        logger.info("🏆 候補絞り込み手法包括評価結果")
        logger.info("="*120)
        
        logger.info(f"\\n🎯 評価設定:")
        logger.info(f"  初期候補数: ~{self.initial_candidates}銘柄")
        logger.info(f"  最終候補数: {self.target_candidates}銘柄")
        logger.info(f"  確信度閾値: {self.confidence_threshold*100:.0f}%")
        
        # 結果をスコアでソート
        sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        logger.info(f"\\n📊 手法別パフォーマンス（精度順）:")
        logger.info(f"{'順位':>4s} {'手法名':25s} {'精度':>8s} {'予測数':>8s} {'平均確信度':>12s} {'日次選択数':>12s}")
        logger.info("-" * 80)
        
        for i, (method_name, result) in enumerate(sorted_results, 1):
            logger.info(f"{i:4d} {method_name:25s} {result['accuracy']:8.1%} {result['total_predictions']:8,d} {result['avg_confidence']:11.1%} {result['avg_daily_selections']:11.1f}")
        
        # 最高手法の詳細
        best_method, best_result = sorted_results[0]
        logger.info(f"\\n🥇 最高精度手法: {best_method}")
        logger.info(f"  精度: {best_result['accuracy']:.2%}")
        logger.info(f"  正解数: {best_result['correct_predictions']:,}/{best_result['total_predictions']:,}")
        logger.info(f"  平均確信度: {best_result['avg_confidence']:.1%}")
        logger.info(f"  評価日数: {best_result['evaluation_days']:,}日")
        
        # 手法の説明
        method_descriptions = {
            'Method1_Confidence': '単純確信度上位選択',
            'Method2_SectorDiversity': 'セクター分散 + 確信度',
            'Method3_RiskAdjusted': 'リスク調整（ボラティリティ）',
            'Method4_Momentum': 'モメンタム + 確信度',
            'Method5_Liquidity': '流動性 + 確信度',
            'Method6_Technical': 'テクニカル指標 + 確信度',
            'Method7_Hybrid': 'ハイブリッド（複数要素統合）'
        }
        
        logger.info(f"\\n📋 手法説明:")
        for method_name, description in method_descriptions.items():
            status = "🥇" if method_name == best_method else "📊"
            logger.info(f"  {status} {method_name}: {description}")
        
        logger.info("="*120)
        
        return best_method, best_result
    
    def implement_best_method(self, best_method_name, df, X, y):
        """最高手法の実装コード生成"""
        logger.info(f"\\n🚀 最高手法 '{best_method_name}' の実装...")
        
        method_mapping = {
            'Method1_Confidence': self.filter_method_1_confidence,
            'Method2_SectorDiversity': self.filter_method_2_sector_diversity,
            'Method3_RiskAdjusted': self.filter_method_3_risk_adjusted,
            'Method4_Momentum': self.filter_method_4_momentum,
            'Method5_Liquidity': self.filter_method_5_liquidity,
            'Method6_Technical': self.filter_method_6_technical,
            'Method7_Hybrid': self.filter_method_7_hybrid
        }
        
        best_method_func = method_mapping[best_method_name]
        
        # 実装例の生成
        logger.info(f"✅ 最高精度手法 '{best_method_name}' の実装準備完了")
        logger.info(f"💡 この手法を実際の取引システムに組み込むことを推奨")
        
        return best_method_func

def main():
    """メイン実行"""
    logger.info("🧪 包括的候補絞り込み手法評価システム")
    
    evaluator = ComprehensiveFilteringEvaluation()
    
    try:
        # データ準備
        df, X, y = evaluator.load_and_prepare_data()
        
        # 全手法評価
        results = evaluator.evaluate_all_methods(df, X, y)
        
        # 結果表示
        best_method, best_result = evaluator.display_evaluation_results(results)
        
        # 最高手法実装
        evaluator.implement_best_method(best_method, df, X, y)
        
        logger.info("\\n✅ 包括的評価完了")
        
    except Exception as e:
        logger.error(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()