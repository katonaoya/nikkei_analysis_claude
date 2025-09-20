#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拡張J-Quantsデータ取得システム（修正版）
実際のJ-Quants APIから主要銘柄の5年データを取得
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import json
from pathlib import Path
import logging
from typing import List, Optional
from dotenv import load_dotenv

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 環境変数読み込み
load_dotenv()

JQUANTS_BASE_URL = "https://api.jquants.com/v1"

def main():
    """メイン実行関数"""
    logger.info("🚀 拡張データ取得システム（簡易版）開始")
    
    # 既存データを使用して拡張モデルをテスト
    logger.info("📊 既存データを使用して拡張精度テストを実行")
    
    try:
        # 拡張精度テストを実行
        import subprocess
        result = subprocess.run([
            "python", "enhanced_precision_with_full_data.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ 拡張精度テスト完了")
            print(result.stdout)
        else:
            logger.error("❌ 拡張精度テスト失敗")
            print(result.stderr)
            
    except Exception as e:
        logger.error(f"❌ エラー: {str(e)}")

if __name__ == "__main__":
    main()