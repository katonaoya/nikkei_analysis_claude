#!/usr/bin/env python3
"""
厳密な絞り込み手法検証システム
異常値検出とデバッグ機能付き
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ログ設定
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", level="INFO")

class RigorousFilteringValidation:
    """厳密な絞り込み手法検証（デバッグ機能付き）"""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.processed_dir = self.data_dir / "processed"
        
        # 最適特徴量
        self.optimal_features = [
            'Market_Breadth', 'Market_Return', 'Volatility_20', 'Price_vs_MA20',
            'sp500_change', 'vix_change', 'nikkei_change', 'us_10y_change', 'usd_jpy_change'
        ]
        
        # 検証パラメータ
        self.confidence_threshold = 0.55
        self.target_candidates = 5
        self.min_evaluation_samples = 100  # 最低評価サンプル数
        
    def load_and_prepare_data(self):
        """データ読み込みと準備"""
        logger.info("📊 厳密検証用データ準備...")
        
        integrated_file = self.processed_dir / "integrated_with_external.parquet"
        df = pd.read_parquet(integrated_file)
        
        # データクリーニング
        clean_df = df[df['Binary_Direction'].notna()].copy()
        clean_df = clean_df.sort_values(['Date', 'Code']).reset_index(drop=True)
        
        # 重複データ処理（日付・銘柄の組み合わせで最新データのみ保持）
        clean_df = clean_df.groupby(['Date', 'Code']).last().reset_index()
        
        # セクター情報追加（一貫性のため固定シード）
        clean_df = self.add_sector_information(clean_df)
        
        # 翌日リターン計算
        clean_df = clean_df.sort_values(['Code', 'Date'])
        clean_df['Next_Return'] = clean_df.groupby('Code')['Close'].pct_change().shift(-1)
        
        # 特徴量とターゲット準備
        X = clean_df[self.optimal_features].fillna(0)
        y = clean_df['Binary_Direction'].astype(int)
        
        logger.info(f"✅ データ準備完了: {len(clean_df):,}件, {len(self.optimal_features)}特徴量")
        
        return clean_df, X, y
    
    def add_sector_information(self, df):
        """セクター情報の追加（一貫性確保）"""
        np.random.seed(42)  # 固定シード
        unique_codes = df['Code'].unique()
        
        sectors = ['Tech', 'Finance', 'Healthcare', 'Consumer', 'Industrial', 'Materials', 'Energy', 'Utilities']
        sector_mapping = {code: np.random.choice(sectors) for code in unique_codes}
        df['Sector'] = df['Code'].map(sector_mapping)
        
        return df
    
    def validate_data_integrity(self, df, X, y):
        """データ整合性検証"""
        logger.info("🔍 データ整合性検証...")
        
        issues = []
        warnings_only = []
        
        # 基本チェック
        if len(df) != len(X) or len(df) != len(y):
            issues.append(f"データ長不一致: df={len(df)}, X={len(X)}, y={len(y)}")
        
        # 欠損値チェック
        missing_features = X.isnull().sum().sum()
        if missing_features > 0:
            warnings_only.append(f"特徴量に欠損値: {missing_features}件")
        
        # 日付重複チェック（警告のみ、停止しない）
        date_code_counts = df.groupby(['Date', 'Code']).size()
        duplicates = (date_code_counts > 1).sum()
        if duplicates > 0:
            warnings_only.append(f"日付・銘柄重複: {duplicates}件 - 最新データを使用")
        
        # ターゲット分布チェック
        target_dist = y.value_counts()
        if len(target_dist) != 2:
            issues.append(f"ターゲット分布異常: {target_dist}")
        
        # 警告表示
        if warnings_only:
            logger.warning("⚠️ データ整合性の警告:")
            for warning in warnings_only:
                logger.warning(f"  - {warning}")
        
        # 重大な問題のみ停止
        if issues:
            logger.error("❌ データ整合性の重大な問題:")
            for issue in issues:
                logger.error(f"  - {issue}")
            return False
        else:
            logger.info("✅ データ整合性: 正常（警告はあるが実行可能）")
            return True
    
    def debug_filtering_method(self, method_name, method_func, test_data_sample):
        """絞り込み手法のデバッグ"""
        logger.info(f"🐛 {method_name} デバッグ...")
        
        debug_info = {
            'method_name': method_name,
            'input_records': len(test_data_sample),
            'has_pred_proba': 'pred_proba' in test_data_sample.columns,
            'has_sector': 'Sector' in test_data_sample.columns,
            'high_conf_count': 0,
            'selected_count': 0,
            'error': None
        }
        
        try:
            # 高確信度候補数チェック
            if 'pred_proba' in test_data_sample.columns:
                high_conf_mask = (
                    (test_data_sample['pred_proba'] >= self.confidence_threshold) | 
                    (test_data_sample['pred_proba'] <= (1 - self.confidence_threshold))
                )
                debug_info['high_conf_count'] = high_conf_mask.sum()
            
            # 手法適用
            selected = method_func(test_data_sample, self.target_candidates)
            debug_info['selected_count'] = len(selected) if selected else 0
            
            # 異常チェック
            if debug_info['selected_count'] > self.target_candidates:
                debug_info['error'] = f"選択数過多: {debug_info['selected_count']} > {self.target_candidates}"
            elif debug_info['high_conf_count'] > 0 and debug_info['selected_count'] == 0:
                debug_info['error'] = f"高確信度候補あるのに選択0件"
            
        except Exception as e:
            debug_info['error'] = f"実行エラー: {str(e)}"
        
        logger.info(f"  入力: {debug_info['input_records']}件")
        logger.info(f"  高確信度: {debug_info['high_conf_count']}件")
        logger.info(f"  選択: {debug_info['selected_count']}件")
        if debug_info['error']:
            logger.warning(f"  ⚠️ 問題: {debug_info['error']}")
        
        return debug_info
    
    def method_1_simple_confidence(self, day_data, n_candidates=5):
        """手法1: 単純確信度上位選択（デバッグ版）"""
        if 'pred_proba' not in day_data.columns:
            return []
        
        # 確信度の絶対値計算
        day_data = day_data.copy()
        day_data['abs_confidence'] = np.maximum(day_data['pred_proba'], 1 - day_data['pred_proba'])
        
        # 確信度閾値フィルター
        high_conf = day_data[day_data['abs_confidence'] >= self.confidence_threshold]
        
        if len(high_conf) == 0:
            return []
        
        # 上位選択
        selected = high_conf.nlargest(n_candidates, 'abs_confidence')
        return selected['Code'].tolist()
    
    def method_2_sector_diversity(self, day_data, n_candidates=5):
        """手法2: セクター分散（デバッグ版）"""
        if 'pred_proba' not in day_data.columns or 'Sector' not in day_data.columns:
            return []
        
        # 確信度計算
        day_data = day_data.copy()
        day_data['abs_confidence'] = np.maximum(day_data['pred_proba'], 1 - day_data['pred_proba'])
        
        # 高確信度フィルター
        high_conf = day_data[day_data['abs_confidence'] >= self.confidence_threshold]
        
        if len(high_conf) == 0:
            return []
        
        # セクター別最高確信度選択
        selected_codes = []
        used_sectors = set()
        
        # セクターごとの最高確信度
        for sector in high_conf['Sector'].unique():
            if len(selected_codes) >= n_candidates:
                break
            
            sector_data = high_conf[high_conf['Sector'] == sector]
            if len(sector_data) == 0:
                continue
            
            best_in_sector = sector_data.loc[sector_data['abs_confidence'].idxmax()]
            selected_codes.append(best_in_sector['Code'])
            used_sectors.add(sector)
        
        # 不足分を全体から補完
        if len(selected_codes) < n_candidates:
            remaining = high_conf[~high_conf['Code'].isin(selected_codes)]
            additional = remaining.nlargest(n_candidates - len(selected_codes), 'abs_confidence')
            selected_codes.extend(additional['Code'].tolist())
        
        return selected_codes[:n_candidates]
    
    def method_3_volatility_adjusted(self, day_data, n_candidates=5):
        """手法3: ボラティリティ調整（デバッグ版）"""
        required_cols = ['pred_proba', 'Volatility_20']
        if not all(col in day_data.columns for col in required_cols):
            return []
        
        day_data = day_data.copy()
        day_data['abs_confidence'] = np.maximum(day_data['pred_proba'], 1 - day_data['pred_proba'])
        
        # 高確信度フィルター
        high_conf = day_data[day_data['abs_confidence'] >= self.confidence_threshold]
        
        if len(high_conf) == 0:
            return []
        
        # ボラティリティ調整スコア
        high_conf = high_conf.copy()
        high_conf['vol_adj_score'] = high_conf['abs_confidence'] / (high_conf['Volatility_20'] + 0.01)
        
        selected = high_conf.nlargest(n_candidates, 'vol_adj_score')
        return selected['Code'].tolist()
    
    def rigorous_evaluation(self, df, X, y):
        """厳密な評価実行"""
        logger.info("🧪 厳密な絞り込み手法評価開始...")
        
        # データ整合性検証
        if not self.validate_data_integrity(df, X, y):
            logger.error("❌ データ整合性エラーのため評価中止")
            return None
        
        # 評価期間設定（全期間を正しく使用）
        dates = sorted(df['Date'].unique())
        train_end_idx = int(len(dates) * 0.8)  # 80%まで学習
        
        train_dates = dates[:train_end_idx]
        test_dates = dates[train_end_idx:]  # 残り全期間で評価（2024-2025年含む）
        
        logger.info(f"学習期間: {train_dates[0]} - {train_dates[-1]} ({len(train_dates)}日)")
        logger.info(f"評価期間: {test_dates[0]} - {test_dates[-1]} ({len(test_dates)}日)")
        
        # 学習データでモデル訓練
        train_mask = df['Date'].isin(train_dates)
        X_train = X[train_mask]
        y_train = y[train_mask]
        
        scaler = StandardScaler()
        model = LogisticRegression(C=0.001, class_weight='balanced', random_state=42, max_iter=1000)
        
        X_train_scaled = scaler.fit_transform(X_train)
        model.fit(X_train_scaled, y_train)
        
        logger.info(f"モデル学習完了: {len(X_train):,}件で学習")
        
        # 手法定義
        methods = {
            'Simple_Confidence': self.method_1_simple_confidence,
            'Sector_Diversity': self.method_2_sector_diversity,
            'Volatility_Adjusted': self.method_3_volatility_adjusted
        }
        
        # 各手法の詳細評価
        method_results = {}
        
        for method_name, method_func in methods.items():
            logger.info(f"📊 {method_name} 詳細評価...")
            
            total_predictions = 0
            correct_predictions = 0
            daily_selections = []
            evaluation_days = 0
            debug_info_list = []
            
            for i, date in enumerate(test_dates):
                day_data = df[df['Date'] == date].copy()
                if len(day_data) == 0:
                    continue
                
                # 予測実行
                X_day = day_data[self.optimal_features].fillna(0)
                X_day_scaled = scaler.transform(X_day)
                pred_proba = model.predict_proba(X_day_scaled)[:, 1]
                day_data['pred_proba'] = pred_proba
                
                # デバッグ情報取得（最初の3日のみ）
                if i < 3:
                    debug_info = self.debug_filtering_method(method_name, method_func, day_data)
                    debug_info_list.append(debug_info)
                
                # 手法適用
                selected_codes = method_func(day_data, self.target_candidates)
                
                if len(selected_codes) == 0:
                    continue
                
                # 選択された銘柄の評価
                selected_data = day_data[day_data['Code'].isin(selected_codes)]
                
                day_correct = 0
                for _, stock in selected_data.iterrows():
                    prediction = stock['pred_proba'] > 0.5
                    actual = stock['Binary_Direction'] == 1
                    
                    total_predictions += 1
                    if prediction == actual:
                        correct_predictions += 1
                        day_correct += 1
                
                daily_selections.append({
                    'date': date,
                    'total_candidates': len(day_data),
                    'selected': len(selected_codes),
                    'accuracy': day_correct / len(selected_codes) if selected_codes else 0
                })
                
                evaluation_days += 1
            
            # 結果集計と異常値検出
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            avg_daily_selections = np.mean([d['selected'] for d in daily_selections]) if daily_selections else 0
            
            # 異常値検出
            anomalies = []
            if accuracy > 0.85:  # 85%超えは異常
                anomalies.append(f"異常に高い精度: {accuracy:.1%}")
            if total_predictions < self.min_evaluation_samples:
                anomalies.append(f"評価サンプル不足: {total_predictions}件 < {self.min_evaluation_samples}件")
            if avg_daily_selections < 1:
                anomalies.append(f"日次選択数不足: {avg_daily_selections:.1f}件")
            
            method_results[method_name] = {
                'accuracy': accuracy,
                'total_predictions': total_predictions,
                'correct_predictions': correct_predictions,
                'evaluation_days': evaluation_days,
                'avg_daily_selections': avg_daily_selections,
                'daily_selections': daily_selections,
                'debug_info': debug_info_list,
                'anomalies': anomalies,
                'is_reliable': len(anomalies) == 0 and total_predictions >= self.min_evaluation_samples
            }
            
            logger.info(f"  精度: {accuracy:.1%}")
            logger.info(f"  評価サンプル: {total_predictions}件")
            logger.info(f"  評価日数: {evaluation_days}日")
            if anomalies:
                logger.warning(f"  ⚠️ 異常検出: {', '.join(anomalies)}")
            else:
                logger.info(f"  ✅ 正常な結果")
        
        return method_results
    
    def display_rigorous_results(self, results):
        """厳密な結果表示"""
        logger.info("\\n" + "="*120)
        logger.info("🔬 厳密な絞り込み手法検証結果")
        logger.info("="*120)
        
        # 信頼性チェック
        reliable_results = {k: v for k, v in results.items() if v['is_reliable']}
        unreliable_results = {k: v for k, v in results.items() if not v['is_reliable']}
        
        logger.info(f"\\n✅ 信頼性のある結果 ({len(reliable_results)}/{len(results)}手法):")
        if reliable_results:
            sorted_reliable = sorted(reliable_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
            
            for i, (method, result) in enumerate(sorted_reliable, 1):
                logger.info(f"  {i}. {method:20s}: {result['accuracy']:6.1%} "
                           f"(評価{result['total_predictions']:,}件, {result['evaluation_days']:,}日)")
            
            # 最高手法の詳細
            best_method, best_result = sorted_reliable[0]
            logger.info(f"\\n🏆 最高精度手法: {best_method}")
            logger.info(f"  精度: {best_result['accuracy']:.2%}")
            logger.info(f"  正解数: {best_result['correct_predictions']:,}/{best_result['total_predictions']:,}")
            logger.info(f"  平均日次選択: {best_result['avg_daily_selections']:.1f}銘柄")
            
        else:
            logger.warning("  信頼性のある結果なし")
        
        # 問題のある結果
        if unreliable_results:
            logger.info(f"\\n⚠️ 問題のある結果 ({len(unreliable_results)}手法):")
            for method, result in unreliable_results.items():
                logger.warning(f"  {method:20s}: {', '.join(result['anomalies'])}")
        
        # 推奨事項
        logger.info(f"\\n💡 推奨事項:")
        if reliable_results:
            best_method = max(reliable_results.keys(), key=lambda k: reliable_results[k]['accuracy'])
            logger.info(f"  推奨手法: {best_method}")
            logger.info(f"  期待精度: {reliable_results[best_method]['accuracy']:.1%}")
        else:
            logger.info(f"  全手法に問題があります。より多くのデータまたは手法見直しが必要")
        
        logger.info("="*120)
        
        return reliable_results

def main():
    """メイン実行"""
    logger.info("🔬 厳密な絞り込み手法検証システム")
    
    validator = RigorousFilteringValidation()
    
    try:
        # データ準備
        df, X, y = validator.load_and_prepare_data()
        
        # 厳密評価
        results = validator.rigorous_evaluation(df, X, y)
        
        if results:
            # 結果表示
            reliable_results = validator.display_rigorous_results(results)
            
            if reliable_results:
                logger.info("\\n✅ 厳密検証完了 - 信頼性のある結果を取得")
            else:
                logger.warning("\\n⚠️ 厳密検証完了 - 全結果に問題あり、再検討が必要")
        else:
            logger.error("❌ 評価失敗")
            
    except Exception as e:
        logger.error(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()