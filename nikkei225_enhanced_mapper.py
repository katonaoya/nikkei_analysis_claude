#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日経225拡張マッピングシステム
J-Quants APIから取得した全上場銘柄と日経225銘柄リストを高精度でマッピング
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
from typing import List, Optional, Dict, Tuple
from dotenv import load_dotenv
import re
import threading
from difflib import SequenceMatcher

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 環境変数読み込み
load_dotenv()

JQUANTS_BASE_URL = "https://api.jquants.com/v1"

class EnhancedNikkei225Mapper:
    """日経225拡張マッピングシステム"""
    
    def __init__(self):
        """初期化"""
        self.mail_address = os.getenv("JQUANTS_MAIL_ADDRESS")
        self.password = os.getenv("JQUANTS_PASSWORD")
        self.id_token: Optional[str] = None
        self.token_expires_at: Optional[datetime] = None
        self.token_lock = threading.Lock()
        
        if not self.mail_address or not self.password:
            raise ValueError("JQuantsの認証情報が設定されていません (.envファイルを確認してください)")
        
        # 日経225銘柄リスト読み込み
        self.target_companies = self._load_target_companies()
        logger.info(f"日経225ターゲット銘柄数: {len(self.target_companies)}銘柄")
    
    def _load_target_companies(self) -> pd.DataFrame:
        """日経225ターゲット企業読み込み"""
        try:
            df = pd.read_csv('/Users/naoya/Desktop/AI関係/自動売買ツール/claude_code_develop/docment/ユーザー情報/nikkei225_4digit_list.csv')
            logger.info(f"日経225ターゲット企業読み込み完了: {len(df)}銘柄")
            return df
        except Exception as e:
            logger.error(f"日経225ターゲット企業読み込みエラー: {e}")
            return pd.DataFrame()
    
    def _get_id_token(self) -> str:
        """IDトークンを取得（スレッドセーフ）"""
        with self.token_lock:
            if self.id_token and self.token_expires_at and datetime.now() < self.token_expires_at:
                return self.id_token
            
            logger.info("JQuants認証トークンを取得中...")
            time.sleep(1)
            
            try:
                # リフレッシュトークンを取得
                auth_payload = {
                    "mailaddress": self.mail_address,
                    "password": self.password
                }
                
                resp = requests.post(
                    f"{JQUANTS_BASE_URL}/token/auth_user",
                    data=json.dumps(auth_payload),
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                
                if resp.status_code == 429:
                    logger.warning("認証レート制限により2分待機...")
                    time.sleep(120)
                    return self._get_id_token()
                    
                resp.raise_for_status()
                refresh_token = resp.json().get("refreshToken")
                
                if not refresh_token:
                    raise RuntimeError("リフレッシュトークンの取得に失敗しました")
                
                time.sleep(1)
                resp = requests.post(
                    f"{JQUANTS_BASE_URL}/token/auth_refresh?refreshtoken={refresh_token}",
                    timeout=30
                )
                
                if resp.status_code == 429:
                    logger.warning("認証レート制限により2分待機...")
                    time.sleep(120)
                    return self._get_id_token()
                    
                resp.raise_for_status()
                self.id_token = resp.json().get("idToken")
                
                if not self.id_token:
                    raise RuntimeError("IDトークンの取得に失敗しました")
                
                self.token_expires_at = datetime.now() + timedelta(hours=1)
                
                logger.info("認証トークン取得完了")
                return self.id_token
                
            except Exception as e:
                logger.error(f"認証エラー: {str(e)}")
                raise
    
    def get_all_listed_companies(self) -> pd.DataFrame:
        """API上場銘柄一覧を全て取得"""
        token = self._get_id_token()
        headers = {"Authorization": f"Bearer {token}"}
        url = f"{JQUANTS_BASE_URL}/listed/info"
        
        all_companies = []
        pagination_key = None
        
        logger.info("全上場銘柄一覧を取得中...")
        
        while True:
            params = {}
            if pagination_key:
                params["pagination_key"] = pagination_key
            
            try:
                time.sleep(1)  # レート制限対策
                response = requests.get(url, headers=headers, params=params, timeout=30)
                
                if response.status_code == 429:
                    logger.warning("レート制限により2分待機...")
                    time.sleep(120)
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                if "info" in data:
                    for company in data["info"]:
                        all_companies.append({
                            "api_code": company.get("Code", ""),
                            "company_name": company.get("CompanyName", ""),
                            "sector": company.get("Sector33Code", ""),
                            "sector_name": company.get("Sector33Name", "")
                        })
                
                # 次のページがあるかチェック
                pagination_key = data.get("pagination_key")
                if not pagination_key:
                    break
                    
            except Exception as e:
                logger.error(f"上場銘柄一覧取得エラー: {e}")
                time.sleep(5)
                continue
        
        df = pd.DataFrame(all_companies)
        logger.info(f"API上場銘柄一覧取得完了: {len(df)}銘柄")
        return df
    
    def normalize_company_name(self, name: str) -> str:
        """企業名正規化（株式表記除去など）"""
        if pd.isna(name) or name == "":
            return ""
        
        # 株式会社表記の除去
        patterns = [
            r'（株）', r'\(株\)', r'株式会社', r'\(株式会社\)', r'（株式会社）',
            r'ホールディングス', r'ホールディング', r'HD', r'Hd',
            r'\s+', r'　+',  # 空白の正規化
        ]
        
        normalized = name
        for pattern in patterns:
            normalized = re.sub(pattern, '', normalized)
        
        return normalized.strip()
    
    def similarity_score(self, text1: str, text2: str) -> float:
        """文字列類似度計算"""
        if not text1 or not text2:
            return 0.0
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def enhanced_mapping(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """拡張マッピング実行"""
        logger.info("拡張マッピング処理開始...")
        
        # API上場銘柄取得
        api_companies = self.get_all_listed_companies()
        
        matched_companies = []
        match_details = []
        
        # 各ターゲット企業に対してマッピング実行
        for idx, target_row in self.target_companies.iterrows():
            target_code = str(target_row['code'])
            target_name = str(target_row['company'])
            target_name_norm = self.normalize_company_name(target_name)
            
            matched = False
            best_match = None
            best_score = 0.0
            match_method = ""
            
            # フェーズ1: 直接コード一致（5桁）
            direct_match = api_companies[api_companies['api_code'] == target_code]
            if not direct_match.empty:
                best_match = direct_match.iloc[0]
                match_method = "直接コード一致(5桁)"
                matched = True
                logger.info(f"✅ 直接一致: {target_code} -> {best_match['company_name']}")
            
            # フェーズ2: 4桁コード一致（末尾0追加）
            if not matched:
                code_with_zero = target_code + "0"
                zero_match = api_companies[api_companies['api_code'] == code_with_zero]
                if not zero_match.empty:
                    best_match = zero_match.iloc[0]
                    match_method = "4桁+0パターン"
                    matched = True
                    logger.info(f"✅ 4桁+0一致: {target_code} -> {code_with_zero} -> {best_match['company_name']}")
            
            # フェーズ3: 部分一致検索（前方一致・後方一致）
            if not matched:
                # 前方一致
                prefix_matches = api_companies[api_companies['api_code'].str.startswith(target_code)]
                if not prefix_matches.empty:
                    best_match = prefix_matches.iloc[0]
                    match_method = f"前方一致({target_code}*)"
                    matched = True
                    logger.info(f"✅ 前方一致: {target_code} -> {best_match['api_code']} -> {best_match['company_name']}")
                
                # 後方一致
                if not matched:
                    suffix_matches = api_companies[api_companies['api_code'].str.endswith(target_code)]
                    if not suffix_matches.empty:
                        best_match = suffix_matches.iloc[0]
                        match_method = f"後方一致(*{target_code})"
                        matched = True
                        logger.info(f"✅ 後方一致: {target_code} -> {best_match['api_code']} -> {best_match['company_name']}")
            
            # フェーズ4: 企業名による高精度マッチング
            if not matched:
                max_similarity = 0.0
                best_name_match = None
                
                for _, api_row in api_companies.iterrows():
                    api_name_norm = self.normalize_company_name(api_row['company_name'])
                    
                    # 完全一致
                    if target_name_norm == api_name_norm and target_name_norm != "":
                        best_name_match = api_row
                        max_similarity = 1.0
                        break
                    
                    # 部分一致（含有関係）
                    if target_name_norm in api_name_norm or api_name_norm in target_name_norm:
                        if target_name_norm != "" and api_name_norm != "":
                            similarity = self.similarity_score(target_name_norm, api_name_norm)
                            if similarity > max_similarity:
                                max_similarity = similarity
                                best_name_match = api_row
                    
                    # 類似度計算
                    similarity = self.similarity_score(target_name_norm, api_name_norm)
                    if similarity > max_similarity and similarity > 0.7:
                        max_similarity = similarity
                        best_name_match = api_row
                
                if best_name_match is not None and max_similarity > 0.7:
                    best_match = best_name_match
                    match_method = f"企業名一致(類似度:{max_similarity:.2f})"
                    matched = True
                    logger.info(f"✅ 企業名一致: {target_name_norm} -> {self.normalize_company_name(best_match['company_name'])} (類似度:{max_similarity:.2f})")
            
            # 結果記録
            if matched and best_match is not None:
                matched_companies.append({
                    "target_code": target_code,
                    "target_name": target_name,
                    "api_code": best_match['api_code'],
                    "api_name": best_match['company_name'],
                    "sector": best_match.get('sector_name', ''),
                    "match_method": match_method,
                    "similarity_score": best_score
                })
            
            # マッチング詳細記録
            match_details.append({
                "target_code": target_code,
                "target_name": target_name,
                "matched": matched,
                "match_method": match_method if matched else "未マッチ",
                "api_code": best_match['api_code'] if matched else "",
                "api_name": best_match['company_name'] if matched else ""
            })
        
        # 結果をDataFrameに変換
        matched_df = pd.DataFrame(matched_companies)
        details_df = pd.DataFrame(match_details)
        
        # 取得不可企業リスト
        unmatched_df = details_df[details_df['matched'] == False].copy()
        
        logger.info(f"🎉 拡張マッピング完了!")
        logger.info(f"マッチした銘柄数: {len(matched_df)}/{len(self.target_companies)} ({len(matched_df)/len(self.target_companies)*100:.1f}%)")
        
        return matched_df, unmatched_df, api_companies
    
    def save_mapping_results(self, matched_df: pd.DataFrame, unmatched_df: pd.DataFrame, api_companies: pd.DataFrame):
        """マッピング結果をCSVファイルに保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # docment/ユーザー情報ディレクトリ作成
        output_dir = Path("docment/ユーザー情報")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # マッチした企業一覧
        matched_file = output_dir / f"nikkei225_matched_companies_{timestamp}.csv"
        matched_df.to_csv(matched_file, index=False, encoding='utf-8-sig')
        logger.info(f"マッチした企業一覧保存: {matched_file}")
        
        # 取得できない企業一覧
        unmatched_file = output_dir / f"nikkei225_unmatched_companies_{timestamp}.csv"
        unmatched_df.to_csv(unmatched_file, index=False, encoding='utf-8-sig')
        logger.info(f"取得できない企業一覧保存: {unmatched_file}")
        
        # API全企業一覧
        api_file = output_dir / f"jquants_all_companies_{timestamp}.csv"
        api_companies.to_csv(api_file, index=False, encoding='utf-8-sig')
        logger.info(f"API全企業一覧保存: {api_file}")
        
        # マッピング統計
        stats = {
            "総ターゲット企業数": len(self.target_companies),
            "マッチした企業数": len(matched_df),
            "取得できない企業数": len(unmatched_df),
            "マッチ率": f"{len(matched_df)/len(self.target_companies)*100:.1f}%",
            "API総企業数": len(api_companies)
        }
        
        stats_file = output_dir / f"mapping_statistics_{timestamp}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"マッピング統計保存: {stats_file}")
        
        return matched_file, unmatched_file, api_file, stats

def main():
    """メイン実行"""
    try:
        mapper = EnhancedNikkei225Mapper()
        
        # 拡張マッピング実行
        matched_df, unmatched_df, api_companies = mapper.enhanced_mapping()
        
        # 結果保存
        matched_file, unmatched_file, api_file, stats = mapper.save_mapping_results(
            matched_df, unmatched_df, api_companies
        )
        
        print(f"\n✅ 拡張マッピング処理完了")
        print(f"📊 マッピング結果:")
        print(f"  - 総ターゲット企業数: {stats['総ターゲット企業数']}")
        print(f"  - マッチした企業数: {stats['マッチした企業数']}")
        print(f"  - 取得できない企業数: {stats['取得できない企業数']}")
        print(f"  - マッチ率: {stats['マッチ率']}")
        print(f"  - API総企業数: {stats['API総企業数']}")
        print(f"\n📁 出力ファイル:")
        print(f"  - マッチ企業: {matched_file}")
        print(f"  - 未マッチ企業: {unmatched_file}")
        print(f"  - API全企業: {api_file}")
        
    except Exception as e:
        logger.error(f"実行エラー: {e}")
        print(f"\n❌ エラーが発生しました: {e}")

if __name__ == "__main__":
    main()