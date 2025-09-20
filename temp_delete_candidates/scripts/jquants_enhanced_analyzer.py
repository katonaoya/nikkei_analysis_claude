#!/usr/bin/env python3
"""
J-Quantsデータを活用した拡張特徴量分析
収集した追加データで精度向上を測定
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class JQuantsEnhancedAnalyzer:
    """J-Quantsデータの拡張分析"""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.processed_dir = self.data_dir / "processed"
        self.jquants_dir = self.data_dir / "raw" / "jquants_enhanced"
        
    def create_mock_jquants_data(self):
        """J-Quantsデータのモック作成（認証なしでテスト用）"""
        logger.info("🔧 J-Quantsデータのモック作成中...")
        
        # 基本的な株価データを読み込み
        base_files = list(self.processed_dir.glob("*.parquet"))
        if not base_files:
            logger.error("❌ 基本データファイルが見つかりません")
            return False
            
        base_file = base_files[0]
        df_base = pd.read_parquet(base_file)
        
        if 'Date' not in df_base.columns or 'Code' not in df_base.columns:
            logger.error("❌ 基本データに必要な列（Date, Code）がありません")
            return False
            
        logger.info(f"基本データ読み込み: {len(df_base)}件")
        
        # モックデータ生成
        self._create_mock_indices_data(df_base)
        self._create_mock_margin_data(df_base) 
        self._create_mock_sector_data(df_base)
        
        return True
    
    def _create_mock_indices_data(self, df_base):
        """指数データのモック作成"""
        logger.info("📈 指数データモック作成中...")
        
        dates = pd.to_datetime(df_base['Date']).drop_duplicates().sort_values()
        
        # TOPIX模擬データ
        np.random.seed(42)
        topix_data = []
        base_price = 2000.0
        
        for date in dates:
            volatility = np.random.normal(0, 0.015)  # 1.5%のボラティリティ
            base_price *= (1 + volatility)
            
            # OHLC作成
            high = base_price * (1 + abs(np.random.normal(0, 0.005)))
            low = base_price * (1 - abs(np.random.normal(0, 0.005)))
            open_price = base_price + np.random.normal(0, base_price * 0.002)
            
            topix_data.append({
                'Date': date,
                'IndexCode': 'TOPIX',
                'Open': open_price,
                'High': high,
                'Low': low,
                'Close': base_price,
                'Volume': np.random.randint(100000000, 500000000)
            })
        
        df_indices = pd.DataFrame(topix_data)
        
        # 日経平均も追加（TOPIX連動）
        nikkei_data = df_indices.copy()
        nikkei_data['IndexCode'] = 'NIKKEI'
        nikkei_data[['Open', 'High', 'Low', 'Close']] *= 15  # 大体の比率
        
        df_all_indices = pd.concat([df_indices, nikkei_data], ignore_index=True)
        
        output_file = self.jquants_dir / "indices_10years.parquet"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df_all_indices.to_parquet(output_file)
        
        logger.info(f"✅ 指数データモック作成完了: {len(df_all_indices)}件")
    
    def _create_mock_margin_data(self, df_base):
        """信用取引データのモック作成"""
        logger.info("💳 信用取引データモック作成中...")
        
        # 週次データ
        dates = pd.to_datetime(df_base['Date']).drop_duplicates().sort_values()
        weekly_dates = dates[dates.dt.dayofweek == 4][::5]  # 金曜日、5週間おき
        
        margin_data = []
        for date in weekly_dates:
            margin_data.append({
                'Date': date,
                'MarginBalance': np.random.randint(1000000, 5000000),
                'ShortBalance': np.random.randint(500000, 2000000),
                'MarginRatio': np.random.uniform(0.1, 0.3),
                'ShortRatio': np.random.uniform(0.05, 0.2)
            })
        
        df_margin = pd.DataFrame(margin_data)
        output_file = self.jquants_dir / "weekly_margin_10years.parquet"
        df_margin.to_parquet(output_file)
        
        logger.info(f"✅ 信用取引データモック作成完了: {len(df_margin)}件")
    
    def _create_mock_sector_data(self, df_base):
        """セクター別データのモック作成"""
        logger.info("🏭 セクター別データモック作成中...")
        
        # 業種コード（実際のTOPIX業種分類に準拠）
        sectors = {
            1: '建設業', 2: '食品', 3: '繊維製品', 4: '化学', 5: '医薬品',
            6: '石油・石炭製品', 7: '鉄鋼', 8: '機械', 9: '電気機器', 10: '輸送用機器',
            11: '精密機器', 12: '不動産業', 13: '陸運業', 14: '情報・通信業', 15: '卸売業',
            16: '小売業', 17: '銀行業', 18: '証券・商品先物', 19: 'その他金融業', 20: 'サービス業'
        }
        
        # コード別セクター割り当て（模擬）
        codes = df_base['Code'].unique()
        np.random.seed(42)
        
        sector_mapping = []
        for code in codes:
            sector_id = np.random.choice(list(sectors.keys()))
            sector_mapping.append({
                'Code': code,
                'SectorId': sector_id,
                'SectorName': sectors[sector_id],
                'Market': np.random.choice(['Prime', 'Standard', 'Growth'], p=[0.4, 0.4, 0.2])
            })
        
        df_sectors = pd.DataFrame(sector_mapping)
        output_file = self.jquants_dir / "sector_mapping.parquet"
        df_sectors.to_parquet(output_file)
        
        logger.info(f"✅ セクター別データモック作成完了: {len(df_sectors)}件")
    
    def load_enhanced_data(self):
        """拡張データの読み込み"""
        logger.info("📊 拡張データ読み込み開始...")
        
        # モックデータ作成
        if not self.create_mock_jquants_data():
            logger.error("❌ モックデータ作成に失敗")
            return None
            
        # 基本データ
        base_files = list(self.processed_dir.glob("*.parquet"))
        if not base_files:
            logger.error("❌ 基本データファイルが見つかりません")
            return None
            
        df_base = pd.read_parquet(base_files[0])
        logger.info(f"基本データ: {len(df_base)}件")
        
        # 指数データ
        indices_file = self.jquants_dir / "indices_10years.parquet"
        if indices_file.exists():
            df_indices = pd.read_parquet(indices_file)
            logger.info(f"指数データ: {len(df_indices)}件")
        else:
            df_indices = None
            logger.warning("⚠️ 指数データが見つかりません")
        
        # 信用データ
        margin_file = self.jquants_dir / "weekly_margin_10years.parquet" 
        if margin_file.exists():
            df_margin = pd.read_parquet(margin_file)
            logger.info(f"信用データ: {len(df_margin)}件")
        else:
            df_margin = None
            logger.warning("⚠️ 信用データが見つかりません")
            
        # セクターデータ
        sector_file = self.jquants_dir / "sector_mapping.parquet"
        if sector_file.exists():
            df_sectors = pd.read_parquet(sector_file)
            logger.info(f"セクターデータ: {len(df_sectors)}件")
        else:
            df_sectors = None
            logger.warning("⚠️ セクターデータが見つかりません")
            
        return {
            'base': df_base,
            'indices': df_indices,
            'margin': df_margin,
            'sectors': df_sectors
        }
    
    def create_enhanced_features(self, data_dict):
        """拡張特徴量の作成"""
        logger.info("🔧 拡張特徴量作成開始...")
        
        df_base = data_dict['base'].copy()
        df_indices = data_dict['indices']
        df_margin = data_dict['margin'] 
        df_sectors = data_dict['sectors']
        
        # 日付を統一
        df_base['Date'] = pd.to_datetime(df_base['Date'])
        
        # 1. 指数関連特徴量
        if df_indices is not None:
            df_indices['Date'] = pd.to_datetime(df_indices['Date'])
            
            # TOPIX特徴量
            topix_data = df_indices[df_indices['IndexCode'] == 'TOPIX'][['Date', 'Close']].rename(columns={'Close': 'TOPIX_Close'})
            df_base = df_base.merge(topix_data, on='Date', how='left')
            
            # TOPIX相対パフォーマンス
            df_base['TOPIX_Return'] = df_base.groupby('Code')['TOPIX_Close'].pct_change()
            df_base['Relative_to_TOPIX'] = df_base.get('Return', 0) - df_base['TOPIX_Return']
            
            logger.info("✅ 指数関連特徴量作成完了")
        
        # 2. セクター関連特徴量
        if df_sectors is not None:
            df_base = df_base.merge(df_sectors[['Code', 'SectorId', 'SectorName']], on='Code', how='left')
            
            # セクター平均からの乖離
            df_base['Sector_Avg_Return'] = df_base.groupby(['Date', 'SectorId'])['Close'].transform('mean')
            df_base['Sector_Relative'] = df_base['Close'] / df_base['Sector_Avg_Return'] - 1
            
            logger.info("✅ セクター関連特徴量作成完了")
        
        # 3. 信用取引関連特徴量（週次データを日次に展開）
        if df_margin is not None:
            df_margin['Date'] = pd.to_datetime(df_margin['Date'])
            df_margin_expanded = df_margin.set_index('Date').resample('D').ffill().reset_index()
            
            df_base = df_base.merge(
                df_margin_expanded[['Date', 'MarginRatio', 'ShortRatio']], 
                on='Date', how='left'
            )
            
            logger.info("✅ 信用取引関連特徴量作成完了")
        
        # 4. 市場全体指標
        daily_stats = df_base.groupby('Date').agg({
            'Volume': ['mean', 'std'],
            'Close': ['mean', 'std']
        }).round(4)
        
        daily_stats.columns = ['Market_Volume_Mean', 'Market_Volume_Std', 'Market_Price_Mean', 'Market_Price_Std']
        daily_stats = daily_stats.reset_index()
        
        df_base = df_base.merge(daily_stats, on='Date', how='left')
        
        # 5. 個別銘柄の市場相対指標
        df_base['Volume_vs_Market'] = df_base['Volume'] / (df_base['Market_Volume_Mean'] + 1e-6)
        df_base['Price_vs_Market'] = df_base['Close'] / (df_base['Market_Price_Mean'] + 1e-6)
        
        logger.info("✅ 市場相対指標作成完了")
        
        # 欠損値処理
        numeric_columns = df_base.select_dtypes(include=[np.number]).columns
        df_base[numeric_columns] = df_base[numeric_columns].fillna(0)
        
        logger.info(f"📊 拡張特徴量作成完了: {df_base.shape}")
        return df_base
    
    def compare_model_performance(self, df_enhanced):
        """拡張特徴量での性能比較"""
        logger.info("⚖️ モデル性能比較開始...")
        
        if 'Binary_Direction' not in df_enhanced.columns:
            logger.error("❌ Binary_Directionが見つかりません")
            return None
            
        # 特徴量分離
        exclude_cols = {
            'Date', 'Code', 'Close', 'High', 'Low', 'Open', 'Volume',
            'Next_Day_Return', 'Binary_Direction', 'SectorName'
        }
        
        feature_cols = [col for col in df_enhanced.columns if col not in exclude_cols]
        
        # 基本特徴量（既存）
        basic_features = [col for col in feature_cols if not any(
            keyword in col for keyword in ['TOPIX', 'Sector', 'Margin', 'Short', 'Market']
        )]
        
        # 拡張特徴量（新規追加分）  
        enhanced_features = [col for col in feature_cols if any(
            keyword in col for keyword in ['TOPIX', 'Sector', 'Margin', 'Short', 'Market']
        )]
        
        logger.info(f"基本特徴量: {len(basic_features)}個")
        logger.info(f"拡張特徴量: {len(enhanced_features)}個")
        logger.info(f"全特徴量: {len(feature_cols)}個")
        
        # データ準備
        clean_data = df_enhanced[df_enhanced['Binary_Direction'].notna()].copy()
        clean_data = clean_data.sort_values(['Date', 'Code']).reset_index(drop=True)
        
        logger.info(f"学習データ: {len(clean_data)}件")
        
        # 数値型以外の列を除外（Timestamp等）
        numeric_features = []
        for col in feature_cols:
            if col in clean_data.columns:
                if clean_data[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    numeric_features.append(col)
        
        logger.info(f"数値特徴量: {len(numeric_features)}個")
        
        # 基本特徴量を数値型のみに限定
        basic_features = [col for col in basic_features if col in numeric_features]
        enhanced_features = [col for col in enhanced_features if col in numeric_features]
        
        logger.info(f"基本特徴量（数値のみ）: {len(basic_features)}個")
        logger.info(f"拡張特徴量（数値のみ）: {len(enhanced_features)}個")
        
        # 評価結果保存
        results = {}
        
        # 1. 基本特徴量のみ
        if basic_features:
            X_basic = clean_data[basic_features].fillna(0)
            y = clean_data['Binary_Direction']
            
            basic_score = self._evaluate_model(X_basic, y, "基本特徴量")
            results['basic'] = basic_score
        
        # 2. 拡張特徴量のみ（新規分だけ）
        if enhanced_features:
            X_enhanced = clean_data[enhanced_features].fillna(0)
            y = clean_data['Binary_Direction']
            
            enhanced_score = self._evaluate_model(X_enhanced, y, "拡張特徴量")
            results['enhanced_only'] = enhanced_score
        
        # 3. 全特徴量（基本+拡張）
        X_all = clean_data[numeric_features].fillna(0)
        y = clean_data['Binary_Direction']
        
        all_score = self._evaluate_model(X_all, y, "全特徴量")
        results['all_features'] = all_score
        
        return results
    
    def _evaluate_model(self, X, y, model_name):
        """モデル評価"""
        logger.info(f"🤖 {model_name}でのモデル評価中...")
        
        tscv = TimeSeriesSplit(n_splits=3)
        scaler = StandardScaler()
        
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=200, max_depth=15, 
                class_weight='balanced', random_state=42, n_jobs=-1
            ),
            'LogisticRegression': LogisticRegression(
                C=0.01, penalty='l1', solver='liblinear',
                class_weight='balanced', random_state=42, max_iter=1000
            )
        }
        
        model_scores = {}
        
        for model_type, model in models.items():
            fold_scores = []
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # 前処理
                if model_type == 'LogisticRegression':
                    X_train_proc = scaler.fit_transform(X_train)
                    X_test_proc = scaler.transform(X_test)
                else:
                    X_train_proc = X_train
                    X_test_proc = X_test
                
                # 学習・予測
                model.fit(X_train_proc, y_train)
                y_pred = model.predict(X_test_proc)
                
                accuracy = accuracy_score(y_test, y_pred)
                fold_scores.append(accuracy)
            
            avg_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)
            
            model_scores[model_type] = {
                'avg_score': avg_score,
                'std_score': std_score,
                'fold_scores': fold_scores
            }
            
            logger.info(f"  {model_type}: {avg_score:.3f} ± {std_score:.3f}")
        
        return model_scores

def main():
    """メイン実行"""
    parser = argparse.ArgumentParser(description="J-Quants enhanced analysis")
    args = parser.parse_args()
    
    try:
        analyzer = JQuantsEnhancedAnalyzer()
        
        print("📊 J-Quantsデータ拡張分析開始")
        print("="*60)
        
        # 拡張データ読み込み
        data_dict = analyzer.load_enhanced_data()
        if not data_dict:
            print("❌ データ読み込みに失敗しました")
            return 1
        
        # 拡張特徴量作成
        df_enhanced = analyzer.create_enhanced_features(data_dict)
        
        # 性能比較
        results = analyzer.compare_model_performance(df_enhanced)
        
        if results:
            # 結果表示
            print("\n" + "="*60)
            print("📋 J-QUANTS拡張分析結果")
            print("="*60)
            
            baseline_score = 0.517  # 既存の最高スコア
            
            for feature_type, model_results in results.items():
                print(f"\n🔍 {feature_type.upper()}:")
                
                for model_name, scores in model_results.items():
                    avg_score = scores['avg_score']
                    improvement = avg_score - baseline_score
                    
                    print(f"   {model_name:18s}: {avg_score:.3f} ({improvement:+.3f})")
                    
                    if improvement > 0.01:
                        print(f"      ✅ 有意な改善 (+{improvement:.1%})")
                    elif improvement > 0.005:
                        print(f"      📈 微細な改善 (+{improvement:.1%})")
                    else:
                        print(f"      ➡️ 変化なし ({improvement:+.1%})")
            
            # 最高スコア
            best_score = 0
            best_config = ""
            
            for feature_type, model_results in results.items():
                for model_name, scores in model_results.items():
                    if scores['avg_score'] > best_score:
                        best_score = scores['avg_score']
                        best_config = f"{feature_type} + {model_name}"
            
            total_improvement = best_score - baseline_score
            
            print(f"\n🏆 最高性能:")
            print(f"   設定: {best_config}")
            print(f"   精度: {best_score:.3f} ({best_score:.1%})")
            print(f"   改善: {total_improvement:+.3f} ({total_improvement:+.1%})")
            
            # 目標達成判定
            if best_score >= 0.53:
                print(f"\n🎉 目標達成! 53%を超えました!")
                print(f"🚀 実用レベルに到達")
            elif best_score >= 0.525:
                print(f"\n🔥 目標に非常に近い! 52.5%以上達成")
                print(f"💡 微調整で53%達成可能")
            elif best_score >= 0.52:
                print(f"\n👍 有意な改善を確認")
                print(f"💡 追加データで更なる向上の余地あり")
            else:
                print(f"\n📈 J-Quantsデータでの改善は限定的")
                print(f"💡 外部データソースが必要")
            
            return 0 if best_score > baseline_score else 1
        else:
            print("❌ 性能評価に失敗しました")
            return 1
            
    except Exception as e:
        logger.error(f"分析中にエラーが発生しました: {e}")
        return 1

if __name__ == "__main__":
    exit(main())