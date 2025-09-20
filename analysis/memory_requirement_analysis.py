#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI学習メモリ要件分析
データサイズ別のメモリ使用量を計算
"""

import pandas as pd
import numpy as np
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_memory_requirements():
    """メモリ要件計算"""
    
    logger.info("💾 AI学習メモリ要件分析")
    logger.info("=" * 60)
    
    # 基本データサイズ分析
    stock_records = 541950  # 株価データ件数
    external_records = 391277  # 外部データ件数
    
    # データ統合後の理論サイズ（Cross Join）
    theoretical_combined = stock_records * external_records
    logger.info(f"📊 データサイズ分析:")
    logger.info(f"  株価データ: {stock_records:,}件")
    logger.info(f"  外部データ: {external_records:,}件")
    logger.info(f"  理論統合サイズ: {theoretical_combined:,}件")
    
    # 実際の統合サイズ（日付ベースマージ）
    actual_combined = 87007005  # ログから取得した実際のサイズ
    logger.info(f"  実際統合サイズ: {actual_combined:,}件")
    
    # 各段階でのメモリ使用量計算
    logger.info("\n" + "=" * 60)
    logger.info("💾 段階別メモリ使用量計算")
    logger.info("=" * 60)
    
    # 1行あたりの平均メモリ使用量を推定
    avg_columns = 30  # 統合後の平均カラム数
    bytes_per_value = 8  # float64
    bytes_per_row = avg_columns * bytes_per_value
    
    stages = [
        ("株価データ読み込み", stock_records, 24),
        ("外部データ読み込み", external_records, 58), 
        ("データ統合後", actual_combined, 27),
        ("特徴量作成後", actual_combined, 50),
        ("機械学習用データ", actual_combined, 50)
    ]
    
    total_peak_memory = 0
    
    for stage_name, records, columns in stages:
        memory_bytes = records * columns * bytes_per_value
        memory_mb = memory_bytes / (1024 * 1024)
        memory_gb = memory_mb / 1024
        
        logger.info(f"{stage_name}:")
        logger.info(f"  データ: {records:,}件 × {columns}カラム")
        logger.info(f"  メモリ: {memory_gb:.2f} GB ({memory_mb:.0f} MB)")
        
        total_peak_memory = max(total_peak_memory, memory_gb)
    
    # ピーク時のメモリ使用量（複数データが同時にメモリ上に存在）
    logger.info("\n" + "=" * 60)
    logger.info("🔥 ピーク時メモリ使用量")
    logger.info("=" * 60)
    
    # 同時にメモリ上に存在するデータ
    peak_scenarios = [
        ("データ統合処理中", [
            ("株価データ", stock_records, 24),
            ("外部データ", external_records, 58),
            ("統合データ（作業用）", actual_combined, 27)
        ]),
        ("特徴量作成中", [
            ("統合データ", actual_combined, 27),
            ("特徴量作成中データ", actual_combined, 50),
            ("作業用コピー", actual_combined // 5, 50)  # バッチ処理
        ]),
        ("機械学習実行中", [
            ("特徴量データ", actual_combined, 50),
            ("訓練データ（X）", actual_combined * 0.8, 20),  # 選択後
            ("テストデータ（X）", actual_combined * 0.2, 20),
            ("目的変数（y）", actual_combined, 1),
            ("予測結果", actual_combined * 0.2, 1)
        ])
    ]
    
    max_memory = 0
    worst_scenario = ""
    
    for scenario_name, data_list in peak_scenarios:
        scenario_memory = 0
        logger.info(f"\n{scenario_name}:")
        
        for data_name, records, columns in data_list:
            data_memory = records * columns * bytes_per_value / (1024**3)
            scenario_memory += data_memory
            logger.info(f"  {data_name}: {data_memory:.2f} GB")
        
        logger.info(f"  合計: {scenario_memory:.2f} GB")
        
        if scenario_memory > max_memory:
            max_memory = scenario_memory
            worst_scenario = scenario_name
    
    # システムオーバーヘッド計算
    logger.info("\n" + "=" * 60)
    logger.info("⚙️ システムオーバーヘッド")
    logger.info("=" * 60)
    
    python_overhead = max_memory * 0.3  # Pythonオーバーヘッド 30%
    os_overhead = 2.0  # OS + その他のプロセス
    safety_margin = max_memory * 0.2  # 安全マージン 20%
    
    total_required = max_memory + python_overhead + os_overhead + safety_margin
    
    logger.info(f"データメモリ: {max_memory:.2f} GB")
    logger.info(f"Pythonオーバーヘッド: {python_overhead:.2f} GB")
    logger.info(f"OSオーバーヘッド: {os_overhead:.2f} GB")
    logger.info(f"安全マージン: {safety_margin:.2f} GB")
    logger.info(f"総必要メモリ: {total_required:.2f} GB")
    
    # メモリ容量別の実行可能性
    logger.info("\n" + "=" * 60)
    logger.info("💻 メモリ容量別実行可能性")
    logger.info("=" * 60)
    
    memory_configs = [
        ("8 GB", 8, "❌ 不可能"),
        ("16 GB", 16, "❌ 不可能" if total_required > 16 else "⚠️ 限界"),
        ("32 GB", 32, "✅ 可能" if total_required <= 32 else "❌ 不可能"),
        ("64 GB", 64, "✅ 余裕"),
        ("128 GB", 128, "✅ 十分")
    ]
    
    for config_name, capacity, status in memory_configs:
        usage_rate = (total_required / capacity) * 100 if capacity >= total_required else 100
        logger.info(f"{config_name}: {status} (使用率: {usage_rate:.1f}%)")
    
    # 最適化版での比較
    logger.info("\n" + "=" * 60)
    logger.info("🔧 最適化版との比較")
    logger.info("=" * 60)
    
    optimized_records = 17910675  # 最適化後のデータ量
    optimized_memory = optimized_records * 50 * bytes_per_value / (1024**3)
    optimized_total = optimized_memory * 2.5  # オーバーヘッド込み
    
    logger.info(f"最適化前（全データ）: {total_required:.2f} GB必要")
    logger.info(f"最適化後（2年間）: {optimized_total:.2f} GB必要")
    logger.info(f"削減効果: {((total_required - optimized_total) / total_required * 100):.1f}%削減")
    
    # 推奨仕様
    logger.info("\n" + "=" * 60)
    logger.info("💡 推奨システム仕様")
    logger.info("=" * 60)
    
    logger.info("🔴 全データ学習の場合:")
    logger.info(f"  必要メモリ: {total_required:.0f} GB以上")
    logger.info(f"  推奨メモリ: {total_required * 1.5:.0f} GB")
    logger.info(f"  適用可能: ワークステーション・サーバー級")
    
    logger.info("\n🟢 最適化版学習の場合:")
    logger.info(f"  必要メモリ: {optimized_total:.0f} GB")
    logger.info(f"  推奨メモリ: 16-32 GB")
    logger.info(f"  適用可能: 高性能PC・ラップトップ")
    
    return {
        'full_data_required': total_required,
        'optimized_required': optimized_total,
        'worst_scenario': worst_scenario,
        'peak_data_memory': max_memory
    }

if __name__ == "__main__":
    results = calculate_memory_requirements()