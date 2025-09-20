#!/usr/bin/env python3
"""
J-Quants API認証システム
スタンダードプラン対応
"""

import os
import json
import requests
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger

class JQuantsAuth:
    """J-Quants API認証管理クラス"""
    
    def __init__(self, email=None, password=None):
        self.base_url = "https://api.jquants.com"
        self.email = email or os.getenv('JQUANTS_EMAIL')
        self.password = password or os.getenv('JQUANTS_PASSWORD')
        self.token_file = Path("jquants_tokens.json")
        
        if not self.email or not self.password:
            logger.warning("J-Quants認証情報が設定されていません")
            logger.info("環境変数 JQUANTS_EMAIL, JQUANTS_PASSWORD を設定してください")
    
    def get_refresh_token(self):
        """リフレッシュトークン取得"""
        url = f"{self.base_url}/v1/token/auth_user"
        payload = {
            "mailaddress": self.email,
            "password": self.password
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            
            refresh_token = data.get('refreshToken')
            if refresh_token:
                logger.info("✅ リフレッシュトークン取得成功")
                return refresh_token
            else:
                logger.error("❌ リフレッシュトークンが応答に含まれていません")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ リフレッシュトークン取得失敗: {e}")
            return None
    
    def get_id_token(self, refresh_token):
        """IDトークン取得（24時間有効）"""
        url = f"{self.base_url}/v1/token/auth_refresh"
        params = {"refreshtoken": refresh_token}
        
        try:
            response = requests.post(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            id_token = data.get('idToken')
            if id_token:
                logger.info("✅ IDトークン取得成功")
                return id_token
            else:
                logger.error("❌ IDトークンが応答に含まれていません")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ IDトークン取得失敗: {e}")
            return None
    
    def save_tokens(self, refresh_token, id_token):
        """トークンをファイルに保存"""
        token_data = {
            'refresh_token': refresh_token,
            'id_token': id_token,
            'created_at': datetime.now().isoformat(),
            'expires_at': (datetime.now() + timedelta(hours=24)).isoformat()
        }
        
        try:
            with open(self.token_file, 'w') as f:
                json.dump(token_data, f, indent=2)
            logger.info(f"✅ トークン保存完了: {self.token_file}")
        except Exception as e:
            logger.error(f"❌ トークン保存失敗: {e}")
    
    def load_tokens(self):
        """保存されたトークンを読み込み"""
        if not self.token_file.exists():
            return None, None
        
        try:
            with open(self.token_file, 'r') as f:
                token_data = json.load(f)
            
            expires_at = datetime.fromisoformat(token_data['expires_at'])
            if datetime.now() >= expires_at:
                logger.info("⏰ IDトークンの有効期限が切れています")
                return token_data['refresh_token'], None
            
            logger.info("✅ 有効なトークンを読み込みました")
            return token_data['refresh_token'], token_data['id_token']
            
        except Exception as e:
            logger.error(f"❌ トークン読み込み失敗: {e}")
            return None, None
    
    def get_valid_id_token(self):
        """有効なIDトークンを取得（自動更新付き）"""
        # 保存されたトークンを確認
        refresh_token, id_token = self.load_tokens()
        
        # 有効なIDトークンがあればそれを返す
        if id_token:
            return id_token
        
        # リフレッシュトークンで新しいIDトークンを取得
        if refresh_token:
            logger.info("🔄 リフレッシュトークンでIDトークンを更新中...")
            new_id_token = self.get_id_token(refresh_token)
            if new_id_token:
                self.save_tokens(refresh_token, new_id_token)
                return new_id_token
        
        # 新しいリフレッシュトークンを取得
        if self.email and self.password:
            logger.info("🔄 新しいリフレッシュトークンを取得中...")
            new_refresh_token = self.get_refresh_token()
            if new_refresh_token:
                new_id_token = self.get_id_token(new_refresh_token)
                if new_id_token:
                    self.save_tokens(new_refresh_token, new_id_token)
                    return new_id_token
        
        logger.error("❌ 有効なIDトークンを取得できませんでした")
        return None
    
    def get_headers(self):
        """認証ヘッダーを取得"""
        id_token = self.get_valid_id_token()
        if id_token:
            return {"Authorization": f"Bearer {id_token}"}
        return {}
    
    def test_auth(self):
        """認証テスト"""
        headers = self.get_headers()
        if not headers:
            logger.error("❌ 認証ヘッダーが取得できません")
            return False
        
        # 上場銘柄一覧で認証テスト
        url = f"{self.base_url}/v1/listed/info"
        params = {"date": "2024-12-30"}
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'info' in data and len(data['info']) > 0:
                logger.success(f"✅ J-Quants認証テスト成功！取得銘柄数: {len(data['info'])}")
                return True
            else:
                logger.error("❌ 認証テストでデータが取得できませんでした")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ 認証テスト失敗: {e}")
            return False

# 使用例
if __name__ == "__main__":
    # 認証テスト
    auth = JQuantsAuth()
    
    if auth.email and auth.password:
        logger.info("🚀 J-Quants認証テスト開始")
        success = auth.test_auth()
        
        if success:
            logger.success("🎉 J-Quants API認証システム準備完了！")
        else:
            logger.error("⚠️ 認証情報を確認してください")
    else:
        logger.warning("認証情報が設定されていません")
        logger.info("環境変数またはコンストラクタで設定してください：")