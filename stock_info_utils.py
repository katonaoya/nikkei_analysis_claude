#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
株式銘柄情報取得ユーティリティ
銘柄コードから会社名を取得する機能
"""

import pandas as pd
import requests
import time
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class StockInfoProvider:
    """株式銘柄情報取得クラス"""
    
    def __init__(self):
        self.company_names = {}
        self.cache_file = Path("production_data/company_names_cache.json")
        self.load_company_names_cache()
        
        # 主要銘柄の会社名マッピング（キャッシュ用）
        self.default_companies = {
            "7203": "トヨタ自動車",
            "9984": "ソフトバンクグループ",
            "6758": "ソニーグループ",
            "8306": "三菱UFJフィナンシャル・グループ",
            "4063": "信越化学工業",
            "6981": "村田製作所",
            "8035": "東京エレクトロン",
            "7974": "任天堂",
            "4568": "第一三共",
            "9434": "ソフトバンク",
            "6954": "ファナック",
            "6501": "日立製作所",
            "8031": "三井物産",
            "4502": "武田薬品工業",
            "7751": "キヤノン",
            "6367": "ダイキン工業",
            "4543": "テルモ",
            "8058": "三菱商事",
            "7267": "本田技研工業",
            "8002": "丸紅",
            # J-Quantsサンプルにありそうな銘柄
            "13010": "日本経済新聞社",
            "13320": "日本証券金融",
            "16050": "イオンリート投資法人",
            "18010": "東証REIT指数連動型上場投資信託",
            "18020": "NEXT FUNDS 東証REIT指数連動型上場投信",
            "25310": "グリー",
            "77310": "日本取引所グループ",
            "82670": "ギグワークス",
            "97660": "モビルス",
            "99830": "オハラ",
            "78320": "ビットワングループ",
            "63670": "ANYCOLOR"
        }
        
    def load_company_names_cache(self):
        """会社名キャッシュ読み込み"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    file_cache = json.load(f)
                    # ファイルのキャッシュを優先（「銘柄XXXXX」を上書き）
                    for key, value in file_cache.items():
                        if not value.startswith('銘柄'):
                            self.company_names[key] = value
                logger.info(f"会社名キャッシュ読み込み完了: {len(self.company_names)}社")
            except Exception as e:
                logger.warning(f"キャッシュ読み込みエラー: {e}")
                self.company_names = {}
        
    def save_company_names_cache(self):
        """会社名キャッシュ保存"""
        try:
            self.cache_file.parent.mkdir(exist_ok=True)
            # 既存のキャッシュと新しいデータをマージ（既存の方を優先）
            if self.cache_file.exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    existing_cache = json.load(f)
                # 既存のキャッシュにないものだけ追加
                for key, value in self.company_names.items():
                    if key not in existing_cache and not value.startswith('銘柄'):
                        existing_cache[key] = value
                self.company_names = existing_cache
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.company_names, f, ensure_ascii=False, indent=2)
            logger.info(f"会社名キャッシュ保存完了: {len(self.company_names)}社")
        except Exception as e:
            logger.warning(f"キャッシュ保存エラー: {e}")
            
    def get_company_name(self, stock_code):
        """銘柄コードから会社名を取得"""
        stock_code = str(stock_code).strip()
        
        # 最新のキャッシュファイルを読み込む
        self.load_company_names_cache()
        
        # キャッシュから取得
        if stock_code in self.company_names:
            # 「銘柄XXXXX」形式の場合は再取得を試みる
            if not self.company_names[stock_code].startswith('銘柄'):
                return self.company_names[stock_code]
            
        # デフォルト企業リストから取得
        if stock_code in self.default_companies:
            company_name = self.default_companies[stock_code]
            self.company_names[stock_code] = company_name
            self.save_company_names_cache()
            return company_name
            
        # Yahoo Finance APIを試行（簡単な方法）
        try:
            company_name = self._fetch_from_yahoo_finance(stock_code)
            if company_name:
                self.company_names[stock_code] = company_name
                self.save_company_names_cache()
                return company_name
        except Exception as e:
            logger.debug(f"Yahoo Finance取得失敗 {stock_code}: {e}")
            
        # 取得失敗時はデフォルト名
        default_name = f"銘柄{stock_code}"
        # デフォルト名はメモリには保存しない（次回再取得のチャンスを残す）
        return default_name
        
    def _fetch_from_yahoo_finance(self, stock_code):
        """Yahoo Financeから会社名取得（簡易版）"""
        try:
            # Yahoo Finance Japan APIの簡易利用
            url = f"https://query1.finance.yahoo.com/v1/finance/search?q={stock_code}.T"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if 'quotes' in data and len(data['quotes']) > 0:
                    quote = data['quotes'][0]
                    if 'shortname' in quote:
                        return quote['shortname']
                    elif 'longname' in quote:
                        return quote['longname']
                        
            return None
        except Exception:
            return None
            
    def get_multiple_company_names(self, stock_codes):
        """複数の銘柄コードから会社名を一括取得"""
        results = {}
        new_names = 0
        
        for code in stock_codes:
            code_str = str(code)
            company_name = self.get_company_name(code_str)
            results[code_str] = company_name
            
            if code_str not in self.company_names:
                new_names += 1
                
            # API制限対策
            if new_names > 0 and new_names % 10 == 0:
                time.sleep(1)
                
        # 新しい情報が追加されたらキャッシュ保存
        if new_names > 0:
            self.save_company_names_cache()
            
        return results


# グローバルインスタンス
_stock_info_provider = None

def get_stock_info_provider():
    """株式情報プロバイダーのシングルトンインスタンス取得"""
    global _stock_info_provider
    if _stock_info_provider is None:
        _stock_info_provider = StockInfoProvider()
    return _stock_info_provider

def get_company_name(stock_code):
    """銘柄コードから会社名取得（簡易関数）"""
    provider = get_stock_info_provider()
    return provider.get_company_name(stock_code)

def get_multiple_company_names(stock_codes):
    """複数銘柄の会社名一括取得（簡易関数）"""
    provider = get_stock_info_provider()
    return provider.get_multiple_company_names(stock_codes)


if __name__ == "__main__":
    # テスト実行
    provider = StockInfoProvider()
    
    test_codes = ["7203", "9984", "6758", "82670", "25310"]
    print("テスト実行:")
    
    for code in test_codes:
        name = provider.get_company_name(code)
        print(f"{code}: {name}")