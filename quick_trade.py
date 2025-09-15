#!/usr/bin/env python
"""
クイックトレード実行 - 1コマンドで全て実行
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
    """メイン実行 - 最もシンプルな使用方法"""
    
    print("🚀 統合トレーディングシステム開始")
    print("="*50)
    print(f"📅 実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # パイプライン実行
    cmd = [
        sys.executable,
        str(scripts_dir / "run_trading_pipeline.py")
    ]
    
    print("⚡ フルパイプライン実行中...")
    print("   データ取得 → 特徴量生成 → モデル訓練 → 予測実行")
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
        else:
            print("\n❌ 実行中にエラーが発生しました")
            print("詳細は trading_pipeline.log を確認してください")
            
        return result.returncode
        
    except Exception as e:
        print(f"\n❌ 実行エラー: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())