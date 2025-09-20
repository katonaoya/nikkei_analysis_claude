#!/usr/bin/env python
"""
クイックトレード実行（既存データ使用版）- データ収集をスキップして高速実行
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

# プロジェクトルートの設定
project_root = Path(__file__).parent
scripts_dir = project_root / "scripts"

def main():
    """メイン実行 - 既存データを使用した高速版"""
    
    print("⚡ 統合トレーディングシステム（高速版）開始")
    print("="*50)
    print(f"📅 実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("📊 既存データを使用してデータ収集をスキップします")
    print()
    
    # パイプライン実行（データ収集スキップ）
    cmd = [
        sys.executable,
        str(scripts_dir / "run_trading_pipeline.py"),
        "--skip-data-collection"
    ]
    
    print("🔧 パイプライン実行中...")
    print("   特徴量生成 → モデル訓練 → 予測実行")
    print()
    
    try:
        result = subprocess.run(cmd, cwd=project_root)
        
        if result.returncode == 0:
            print("\n✅ 実行完了!")
            print("📁 結果ファイルの確認場所:")
            print("   - data/predictions/     : 予測結果")
            print("   - data/models/          : 訓練済みモデル") 
            print("   - data/reports/         : 実行レポート")
            print("   - trading_pipeline.log  : 実行ログ")
            
            # 最新の予測結果があれば簡単に表示
            predictions_dir = project_root / "data" / "predictions"
            if predictions_dir.exists():
                pred_files = list(predictions_dir.glob("predictions_*.csv"))
                if pred_files:
                    latest_pred = max(pred_files, key=lambda f: f.stat().st_mtime)
                    print(f"\n📈 最新の予測結果: {latest_pred.name}")
                    
                    # 先頭数行を表示
                    try:
                        import pandas as pd
                        df = pd.read_csv(latest_pred)
                        print("\n🏆 上位予測結果（サンプル）:")
                        print(df.head(5).to_string(index=False))
                    except:
                        pass
        else:
            print("\n❌ 実行中にエラーが発生しました")
            print("詳細は trading_pipeline.log を確認してください")
            
        return result.returncode
        
    except Exception as e:
        print(f"\n❌ 実行エラー: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())