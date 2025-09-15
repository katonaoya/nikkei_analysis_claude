#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Precision System V3 実運用検証スクリプト
日常運用での実際の精度と問題点を詳細に検証
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealWorldValidator:
    """実運用検証クラス"""
    
    def __init__(self):
        """初期化"""
        # ウォークフォワード結果（Enhanced V3から取得）
        self.walkforward_results = [
            {"period": "2018-10-15 to 2018-11-13", "accuracy": 0.7852, "precision": 0.7696},
            {"period": "2018-11-13 to 2018-12-13", "accuracy": 0.7807, "precision": 0.7641},
            {"period": "2018-12-13 to 2019-01-21", "accuracy": 0.7942, "precision": 0.8187},
            {"period": "2019-01-21 to 2019-02-20", "accuracy": 0.7559, "precision": 0.7874},
            {"period": "2019-02-20 to 2019-03-22", "accuracy": 0.7819, "precision": 0.7398},
            {"period": "2019-03-22 to 2019-04-22", "accuracy": 0.8014, "precision": 0.8097},
            {"period": "2019-04-22 to 2019-05-29", "accuracy": 0.7879, "precision": 0.6795},
            {"period": "2019-05-29 to 2019-06-27", "accuracy": 0.7825, "precision": 0.7698},
            {"period": "2019-06-27 to 2019-07-29", "accuracy": 0.8039, "precision": 0.7834},
            {"period": "2019-07-29 to 2019-08-28", "accuracy": 0.7804, "precision": 0.6554},
            {"period": "2019-08-28 to 2019-09-30", "accuracy": 0.7583, "precision": 0.8486},
            {"period": "2019-09-30 to 2019-10-31", "accuracy": 0.7964, "precision": 0.8258},
            {"period": "2019-10-31 to 2019-12-02", "accuracy": 0.7705, "precision": 0.8233},
            {"period": "2019-12-02 to 2020-01-06", "accuracy": 0.7997, "precision": 0.7731},
            {"period": "2020-01-06 to 2020-02-05", "accuracy": 0.7958, "precision": 0.7453},
            {"period": "2020-02-05 to 2020-03-09", "accuracy": 0.8441, "precision": 0.7249},
            {"period": "2020-03-09 to 2020-04-08", "accuracy": 0.8123, "precision": 0.9136},
            {"period": "2020-04-08 to 2020-05-13", "accuracy": 0.7471, "precision": 0.8130},
            {"period": "2020-05-13 to 2020-06-11", "accuracy": 0.8074, "precision": 0.8730},
            {"period": "2020-06-11 to 2020-07-10", "accuracy": 0.7407, "precision": 0.7678},
            {"period": "2020-07-10 to 2020-08-13", "accuracy": 0.8172, "precision": 0.8648},
            {"period": "2020-08-13 to 2020-09-11", "accuracy": 0.7246, "precision": 0.7595},
            {"period": "2020-09-11 to 2020-10-15", "accuracy": 0.8160, "precision": 0.7621},
            {"period": "2020-10-15 to 2020-11-16", "accuracy": 0.7658, "precision": 0.8648},
            {"period": "2020-11-16 to 2020-12-16", "accuracy": 0.8003, "precision": 0.8716},
            {"period": "2020-12-16 to 2021-01-19", "accuracy": 0.8112, "precision": 0.8761},
            {"period": "2021-01-19 to 2021-02-18", "accuracy": 0.8042, "precision": 0.8715},
            {"period": "2021-02-18 to 2021-03-22", "accuracy": 0.7846, "precision": 0.8588},
            {"period": "2021-03-22 to 2021-04-20", "accuracy": 0.7864, "precision": 0.7283},
            {"period": "2021-04-20 to 2021-05-25", "accuracy": 0.7846, "precision": 0.8159},
            {"period": "2021-05-25 to 2021-06-23", "accuracy": 0.7859, "precision": 0.7816},
            {"period": "2021-06-23 to 2021-07-26", "accuracy": 0.7741, "precision": 0.7274},
            {"period": "2021-07-26 to 2021-08-25", "accuracy": 0.7861, "precision": 0.7375},
            {"period": "2021-08-25 to 2021-09-27", "accuracy": 0.7795, "precision": 0.8409},
            {"period": "2021-09-27 to 2021-10-26", "accuracy": 0.7982, "precision": 0.8021},
            {"period": "2021-10-26 to 2021-11-26", "accuracy": 0.7844, "precision": 0.7705},
            {"period": "2021-11-26 to 2021-12-27", "accuracy": 0.7898, "precision": 0.8318},
            {"period": "2021-12-27 to 2022-01-28", "accuracy": 0.7784, "precision": 0.7741},
            {"period": "2022-01-28 to 2022-03-02", "accuracy": 0.7769, "precision": 0.7490},
            {"period": "2022-03-02 to 2022-04-01", "accuracy": 0.8196, "precision": 0.8293},
            {"period": "2022-04-01 to 2022-05-06", "accuracy": 0.7724, "precision": 0.7301},
            {"period": "2022-05-06 to 2022-06-06", "accuracy": 0.7756, "precision": 0.8090},
            {"period": "2022-06-06 to 2022-07-05", "accuracy": 0.7828, "precision": 0.7593},
            {"period": "2022-07-05 to 2022-08-04", "accuracy": 0.7745, "precision": 0.7771},
            {"period": "2022-08-04 to 2022-09-05", "accuracy": 0.7596, "precision": 0.6958},
            {"period": "2022-09-05 to 2022-10-06", "accuracy": 0.7642, "precision": 0.7060},
            {"period": "2022-10-06 to 2022-11-08", "accuracy": 0.7845, "precision": 0.8198},
            {"period": "2022-11-08 to 2022-12-08", "accuracy": 0.7879, "precision": 0.7831},
            {"period": "2022-12-08 to 2023-01-11", "accuracy": 0.7879, "precision": 0.6782},
            {"period": "2023-01-11 to 2023-02-09", "accuracy": 0.7930, "precision": 0.8471},
            {"period": "2023-02-09 to 2023-03-13", "accuracy": 0.8080, "precision": 0.7991},
            {"period": "2023-03-13 to 2023-04-12", "accuracy": 0.7600, "precision": 0.7584},
            {"period": "2023-04-12 to 2023-05-16", "accuracy": 0.7862, "precision": 0.8352},
            {"period": "2023-05-16 to 2023-06-14", "accuracy": 0.7812, "precision": 0.9259},
            {"period": "2023-06-14 to 2023-07-13", "accuracy": 0.7801, "precision": 0.6952},
            {"period": "2023-07-13 to 2023-08-15", "accuracy": 0.7799, "precision": 0.7807},
            {"period": "2023-08-15 to 2023-09-13", "accuracy": 0.7805, "precision": 0.8096},
            {"period": "2023-09-13 to 2023-10-16", "accuracy": 0.8040, "precision": 0.7684},
            {"period": "2023-10-16 to 2023-11-15", "accuracy": 0.7843, "precision": 0.8121},
            {"period": "2023-11-15 to 2023-12-15", "accuracy": 0.8044, "precision": 0.8378},
            {"period": "2023-12-15 to 2024-01-19", "accuracy": 0.7640, "precision": 0.8743},
            {"period": "2024-01-19 to 2024-02-20", "accuracy": 0.7651, "precision": 0.7638},
            {"period": "2024-02-20 to 2024-03-22", "accuracy": 0.8040, "precision": 0.8457},
            {"period": "2024-03-22 to 2024-04-22", "accuracy": 0.7558, "precision": 0.7198},
            {"period": "2024-04-22 to 2024-05-24", "accuracy": 0.7784, "precision": 0.7658},
            {"period": "2024-05-24 to 2024-06-24", "accuracy": 0.7939, "precision": 0.7996},
            {"period": "2024-06-24 to 2024-07-24", "accuracy": 0.8025, "precision": 0.8203},
            {"period": "2024-07-24 to 2024-08-23", "accuracy": 0.7926, "precision": 0.8225},
            {"period": "2024-08-23 to 2024-09-25", "accuracy": 0.7731, "precision": 0.7834},
            {"period": "2024-09-25 to 2024-10-25", "accuracy": 0.7655, "precision": 0.7374},
            {"period": "2024-10-25 to 2024-11-26", "accuracy": 0.7799, "precision": 0.8237},
            {"period": "2024-11-26 to 2024-12-25", "accuracy": 0.7892, "precision": 0.8188},
            {"period": "2024-12-25 to 2025-01-30", "accuracy": 0.7966, "precision": 0.8614},
            {"period": "2025-01-30 to 2025-03-04", "accuracy": 0.7752, "precision": 0.7561},
            {"period": "2025-03-04 to 2025-04-03", "accuracy": 0.7704, "precision": 0.6833},
            {"period": "2025-04-03 to 2025-05-07", "accuracy": 0.7879, "precision": 0.8179},
            {"period": "2025-05-07 to 2025-06-05", "accuracy": 0.7998, "precision": 0.7854},
            {"period": "2025-06-05 to 2025-07-04", "accuracy": 0.8006, "precision": 0.8294},
            {"period": "2025-07-04 to 2025-08-05", "accuracy": 0.7674, "precision": 0.8272},
            {"period": "2025-08-05 to 2025-09-04", "accuracy": 0.7958, "precision": 0.8418}
        ]
        
        self.df = pd.DataFrame(self.walkforward_results)
        
        # 図表保存ディレクトリ
        self.output_dir = Path("validation_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"実運用検証システム初期化完了: {len(self.walkforward_results)}期間のデータ")
    
    def analyze_basic_statistics(self):
        """基本統計分析"""
        logger.info("📊 基本統計分析開始...")
        
        # 基本統計
        accuracy_stats = self.df['accuracy'].describe()
        precision_stats = self.df['precision'].describe()
        
        print("\n" + "="*60)
        print("📊 Enhanced Precision System V3 実運用統計分析")
        print("="*60)
        
        print(f"\n🎯 精度（Accuracy）統計:")
        print(f"   平均: {accuracy_stats['mean']:.3f} ({accuracy_stats['mean']*100:.1f}%)")
        print(f"   中央値: {accuracy_stats['50%']:.3f} ({accuracy_stats['50%']*100:.1f}%)")
        print(f"   標準偏差: {accuracy_stats['std']:.3f} ({accuracy_stats['std']*100:.1f}%)")
        print(f"   最大値: {accuracy_stats['max']:.3f} ({accuracy_stats['max']*100:.1f}%)")
        print(f"   最小値: {accuracy_stats['min']:.3f} ({accuracy_stats['min']*100:.1f}%)")
        print(f"   75%ile: {accuracy_stats['75%']:.3f} ({accuracy_stats['75%']*100:.1f}%)")
        print(f"   25%ile: {accuracy_stats['25%']:.3f} ({accuracy_stats['25%']*100:.1f}%)")
        
        print(f"\n🔍 適合率（Precision）統計:")
        print(f"   平均: {precision_stats['mean']:.3f} ({precision_stats['mean']*100:.1f}%)")
        print(f"   中央値: {precision_stats['50%']:.3f} ({precision_stats['50%']*100:.1f}%)")
        print(f"   標準偏差: {precision_stats['std']:.3f} ({precision_stats['std']*100:.1f}%)")
        print(f"   最大値: {precision_stats['max']:.3f} ({precision_stats['max']*100:.1f}%)")
        print(f"   最小値: {precision_stats['min']:.3f} ({precision_stats['min']*100:.1f}%)")
        
        return accuracy_stats, precision_stats
    
    def analyze_consistency(self):
        """一貫性分析"""
        logger.info("📈 一貫性・安定性分析開始...")
        
        accuracies = self.df['accuracy']
        precisions = self.df['precision']
        
        # 閾値別分析
        thresholds = [0.70, 0.75, 0.80, 0.85]
        
        print(f"\n📈 安定性・一貫性分析:")
        print(f"   検証期間: {len(self.df)}期間 (2018年10月〜2025年9月)")
        
        for threshold in thresholds:
            acc_above = (accuracies >= threshold).sum()
            acc_rate = acc_above / len(accuracies) * 100
            
            prec_above = (precisions >= threshold).sum() 
            prec_rate = prec_above / len(precisions) * 100
            
            print(f"   {threshold*100:.0f}%以上維持:")
            print(f"     精度: {acc_above}/{len(accuracies)}期間 ({acc_rate:.1f}%)")
            print(f"     適合率: {prec_above}/{len(precisions)}期間 ({prec_rate:.1f}%)")
        
        # 連続性分析
        consecutive_above_75 = 0
        max_consecutive = 0
        current_consecutive = 0
        
        for acc in accuracies:
            if acc >= 0.75:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        print(f"\n📊 連続性分析（75%以上）:")
        print(f"   最大連続維持: {max_consecutive}期間")
        
        # ボラティリティ分析
        acc_volatility = accuracies.std()
        prec_volatility = precisions.std()
        
        print(f"\n📉 ボラティリティ分析:")
        print(f"   精度の変動性: {acc_volatility:.3f} ({acc_volatility*100:.1f}%)")
        print(f"   適合率の変動性: {prec_volatility:.3f} ({prec_volatility*100:.1f}%)")
        
        return {
            'accuracy_volatility': acc_volatility,
            'precision_volatility': prec_volatility,
            'max_consecutive_75': max_consecutive
        }
    
    def real_world_scenario_analysis(self):
        """実運用シナリオ分析"""
        logger.info("🌍 実運用シナリオ分析開始...")
        
        print(f"\n🌍 実運用シナリオ分析:")
        
        # シナリオ1: 保守的運用（70%以下で停止）
        poor_periods = self.df[self.df['accuracy'] < 0.70]
        print(f"\n📉 危険期間分析（70%未満）:")
        print(f"   発生回数: {len(poor_periods)}回 / {len(self.df)}期間 ({len(poor_periods)/len(self.df)*100:.1f}%)")
        
        if len(poor_periods) > 0:
            print(f"   最低精度: {poor_periods['accuracy'].min():.3f} ({poor_periods['accuracy'].min()*100:.1f}%)")
            print(f"   該当期間例: {poor_periods.iloc[0]['period']}")
        
        # シナリオ2: 月次リターン計算
        monthly_returns = []
        daily_trades_per_month = 20  # 営業日
        trades_per_day = 3  # 推奨銘柄数
        target_return = 0.01  # 1%上昇
        
        for _, row in self.df.iterrows():
            accuracy = row['accuracy']
            precision = row['precision']
            
            # 月次期待リターン計算
            expected_return = daily_trades_per_month * trades_per_day * target_return * accuracy
            monthly_returns.append(expected_return)
        
        monthly_returns = np.array(monthly_returns)
        
        print(f"\n💰 期待収益率分析:")
        print(f"   平均月次期待リターン: {monthly_returns.mean():.1%}")
        print(f"   月次リターン中央値: {np.median(monthly_returns):.1%}")
        print(f"   最高月次リターン: {monthly_returns.max():.1%}")
        print(f"   最低月次リターン: {monthly_returns.min():.1%}")
        print(f"   年間期待リターン: {monthly_returns.mean() * 12:.1%}")
        
        # リスク分析
        negative_months = (monthly_returns < 0.10).sum()  # 10%未満をリスク期間とする
        print(f"\n⚠️ リスク分析:")
        print(f"   低パフォーマンス期間: {negative_months}期間 ({negative_months/len(monthly_returns)*100:.1f}%)")
        print(f"   リターン標準偏差: {monthly_returns.std():.1%}")
        
        return {
            'monthly_returns': monthly_returns,
            'poor_periods': len(poor_periods),
            'avg_monthly_return': monthly_returns.mean(),
            'annual_expected_return': monthly_returns.mean() * 12
        }
    
    def create_visualizations(self):
        """可視化作成"""
        logger.info("📊 可視化作成開始...")
        
        # 図のサイズとスタイル設定
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Enhanced Precision System V3 実運用分析', fontsize=16, fontweight='bold')
        
        # 1. 精度の時系列推移
        axes[0, 0].plot(range(len(self.df)), self.df['accuracy'], 
                       marker='o', linewidth=2, markersize=4, color='blue', alpha=0.7)
        axes[0, 0].axhline(y=0.75, color='red', linestyle='--', alpha=0.7, label='75%閾値')
        axes[0, 0].axhline(y=0.80, color='green', linestyle='--', alpha=0.7, label='80%閾値')
        axes[0, 0].set_title('精度の時系列推移 (2018-2025)', fontweight='bold')
        axes[0, 0].set_xlabel('期間')
        axes[0, 0].set_ylabel('精度')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 精度分布
        axes[0, 1].hist(self.df['accuracy'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].axvline(x=self.df['accuracy'].mean(), color='red', linestyle='-', 
                          linewidth=2, label=f'平均: {self.df["accuracy"].mean():.1%}')
        axes[0, 1].axvline(x=self.df['accuracy'].median(), color='green', linestyle='--', 
                          linewidth=2, label=f'中央値: {self.df["accuracy"].median():.1%}')
        axes[0, 1].set_title('精度分布', fontweight='bold')
        axes[0, 1].set_xlabel('精度')
        axes[0, 1].set_ylabel('頻度')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 精度 vs 適合率
        scatter = axes[1, 0].scatter(self.df['accuracy'], self.df['precision'], 
                                   alpha=0.6, c=range(len(self.df)), cmap='viridis')
        axes[1, 0].set_title('精度 vs 適合率', fontweight='bold')
        axes[1, 0].set_xlabel('精度')
        axes[1, 0].set_ylabel('適合率')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 年別パフォーマンス
        # 年別集計
        yearly_data = []
        for _, row in self.df.iterrows():
            year = int(row['period'][:4])
            yearly_data.append({'year': year, 'accuracy': row['accuracy'], 'precision': row['precision']})
        
        yearly_df = pd.DataFrame(yearly_data)
        yearly_avg = yearly_df.groupby('year')[['accuracy', 'precision']].mean()
        
        x_pos = range(len(yearly_avg))
        axes[1, 1].bar([p - 0.2 for p in x_pos], yearly_avg['accuracy'], 
                      width=0.4, label='精度', alpha=0.7, color='blue')
        axes[1, 1].bar([p + 0.2 for p in x_pos], yearly_avg['precision'], 
                      width=0.4, label='適合率', alpha=0.7, color='orange')
        axes[1, 1].set_title('年別平均パフォーマンス', fontweight='bold')
        axes[1, 1].set_xlabel('年')
        axes[1, 1].set_ylabel('精度/適合率')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(yearly_avg.index)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        output_file = self.output_dir / "enhanced_v3_validation_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"可視化保存完了: {output_file}")
        
        return output_file
    
    def critical_issues_analysis(self):
        """重要な問題点分析"""
        logger.info("⚠️ 重要な問題点分析開始...")
        
        print(f"\n⚠️ 実運用での重要な問題点・リスク分析:")
        
        # 1. データ遅延問題
        print(f"\n1️⃣ データ遅延・取得問題:")
        print(f"   - 外部データ（USD/JPY, VIX等）のリアルタイム取得遅延")
        print(f"   - Yahoo Finance APIの制限・障害リスク")
        print(f"   - J-Quants APIの日次更新タイミング(通常15:30頃)")
        print(f"   → 推奨対策: 複数データソース併用、エラーハンドリング強化")
        
        # 2. 市場環境変化
        extreme_periods = self.df[
            (self.df['accuracy'] < 0.72) | 
            (self.df['accuracy'] > 0.84)
        ]
        
        print(f"\n2️⃣ 市場環境急変リスク:")
        print(f"   - 極端なパフォーマンス期間: {len(extreme_periods)}回 ({len(extreme_periods)/len(self.df)*100:.1f}%)")
        print(f"   - 最大精度: {self.df['accuracy'].max():.1%} (過信リスク)")
        print(f"   - 最小精度: {self.df['accuracy'].min():.1%} (重大損失リスク)")
        print(f"   → 推奨対策: リスク管理ルール設定、動的ポジションサイズ調整")
        
        # 3. オーバーフィッティング検出
        recent_10_acc = self.df.tail(10)['accuracy'].mean()
        overall_acc = self.df['accuracy'].mean()
        recent_vs_overall = recent_10_acc - overall_acc
        
        print(f"\n3️⃣ オーバーフィッティング・劣化リスク:")
        print(f"   - 直近10期間平均: {recent_10_acc:.1%}")
        print(f"   - 全期間平均: {overall_acc:.1%}")
        print(f"   - 差異: {recent_vs_overall:+.1%}")
        
        if abs(recent_vs_overall) > 0.02:
            print(f"   ⚠️ 警告: 直近パフォーマンスに{abs(recent_vs_overall):.1%}以上の変化")
        else:
            print(f"   ✅ 良好: パフォーマンス安定")
        
        # 4. 取引コスト・流動性
        print(f"\n4️⃣ 取引コスト・流動性リスク:")
        print(f"   - 証券会社手数料: 約0.1-0.3%/取引")
        print(f"   - スプレッドコスト: 約0.05-0.2%")
        print(f"   - 小型株の流動性リスク")
        print(f"   - 1%目標に対するコスト比率: 10-50%")
        print(f"   → 推奨対策: 大型株中心、手数料定額プラン利用")
        
        # 5. 心理的要因
        losing_streaks = self.analyze_losing_streaks()
        
        print(f"\n5️⃣ 心理的・運用継続リスク:")
        print(f"   - 最大連続低調期間: {losing_streaks['max_streak']}期間")
        print(f"   - 75%未満期間数: {losing_streaks['below_75_count']}期間")
        print(f"   - 運用停止誘惑リスク: 高")
        print(f"   → 推奨対策: 機械的運用ルール、感情介入防止")
        
        return {
            'extreme_periods': len(extreme_periods),
            'recent_performance_change': recent_vs_overall,
            'losing_streaks': losing_streaks
        }
    
    def analyze_losing_streaks(self):
        """連続低調期間分析"""
        accuracies = self.df['accuracy']
        
        current_streak = 0
        max_streak = 0
        below_75_count = 0
        
        for acc in accuracies:
            if acc < 0.75:
                current_streak += 1
                below_75_count += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return {
            'max_streak': max_streak,
            'below_75_count': below_75_count,
            'below_75_rate': below_75_count / len(accuracies)
        }
    
    def final_recommendation(self):
        """最終推奨事項"""
        logger.info("📝 最終推奨事項生成...")
        
        print(f"\n📝 Enhanced Precision System V3 実運用推奨事項:")
        print("="*70)
        
        avg_acc = self.df['accuracy'].mean()
        stability = self.df['accuracy'].std()
        
        if avg_acc >= 0.78 and stability <= 0.03:
            recommendation = "🟢 推奨: 実運用適用可能"
            confidence = "高"
        elif avg_acc >= 0.75 and stability <= 0.05:
            recommendation = "🟡 条件付推奨: 慎重な実運用"
            confidence = "中"
        else:
            recommendation = "🔴 非推奨: 更なる改善必要"
            confidence = "低"
        
        print(f"\n{recommendation}")
        print(f"信頼度: {confidence}")
        
        print(f"\n📊 実運用判定根拠:")
        print(f"   - 平均精度: {avg_acc:.1%} (目標: ≥75%)")
        print(f"   - 安定性: {stability:.1%} (目標: ≤5%)")
        print(f"   - 75%以上維持率: {(self.df['accuracy'] >= 0.75).mean():.1%}")
        
        print(f"\n✅ 実運用での成功要因:")
        print(f"   1. 7年間の長期検証済み（2018-2025）")
        print(f"   2. ウォークフォワードによる未来データ漏洩防止")
        print(f"   3. 外部データ統合による環境変化対応")
        print(f"   4. 月次リバランスによる現実的運用周期")
        
        print(f"\n⚠️ 実運用での注意点:")
        print(f"   1. データ取得遅延・障害対策必須")
        print(f"   2. 取引コストを考慮した銘柄選択")
        print(f"   3. 70%未満時の運用停止ルール設定")
        print(f"   4. 感情介入防止の機械的運用")
        print(f"   5. 定期的なシステム見直し（3ヶ月毎）")
        
        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'avg_accuracy': avg_acc,
            'stability': stability
        }
    
    def run_full_validation(self):
        """完全検証実行"""
        logger.info("🚀 Enhanced Precision System V3 実運用検証開始!")
        
        try:
            # 各種分析実行
            basic_stats = self.analyze_basic_statistics()
            consistency = self.analyze_consistency()
            scenarios = self.real_world_scenario_analysis()
            issues = self.critical_issues_analysis()
            visualization = self.create_visualizations()
            recommendation = self.final_recommendation()
            
            # 最終結果まとめ
            print(f"\n🎉 Enhanced Precision System V3 実運用検証完了!")
            print(f"検証期間: 2018年10月〜2025年9月 ({len(self.df)}期間)")
            print(f"可視化ファイル: {visualization}")
            
            return {
                'basic_stats': basic_stats,
                'consistency': consistency,
                'scenarios': scenarios,
                'issues': issues,
                'recommendation': recommendation,
                'visualization': str(visualization)
            }
            
        except Exception as e:
            logger.error(f"検証エラー: {e}")
            return None

def main():
    """メイン実行"""
    validator = RealWorldValidator()
    results = validator.run_full_validation()
    
    if results:
        print(f"\n✅ 実運用検証完了!")
        print(f"詳細な分析結果が上記に表示されました。")
    else:
        print(f"\n❌ 実運用検証に失敗しました")

if __name__ == "__main__":
    main()