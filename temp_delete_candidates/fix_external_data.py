#!/usr/bin/env python
"""
外部データファイルの列名を修正するスクリプト
"""

import pandas as pd
from pathlib import Path

# 外部データを読み込み
external_path = Path("data/external/external_macro_data.parquet")
df = pd.read_parquet(external_path)

# 列名を修正（文字列から実際の列名を抽出）
new_columns = []
for col in df.columns:
    if isinstance(col, str) and col.startswith("('"):
        # "('Date', '')" -> "Date"
        actual_name = col.split("'")[1]
        new_columns.append(actual_name)
    else:
        new_columns.append(col)

df.columns = new_columns

# 保存
df.to_parquet(external_path, index=False)
print(f"✅ 列名を修正しました: {list(df.columns)[:5]}...")