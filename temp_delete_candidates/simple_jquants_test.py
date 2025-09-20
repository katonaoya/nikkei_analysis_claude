#!/usr/bin/env python3
"""
J-Quants API認証テスト
"""

import requests
import json
import os
from dotenv import load_dotenv

# .envファイルから環境変数を読み込み
load_dotenv()

# J-Quants API認証情報
MAIL_ADDRESS = os.getenv('JQUANTS_MAIL_ADDRESS')
PASSWORD = os.getenv('JQUANTS_PASSWORD')

def test_jquants_auth():
    """J-Quants API認証テスト"""
    print("🔐 J-Quants API認証テスト開始")
    print(f"メールアドレス: {MAIL_ADDRESS}")
    
    try:
        # Step 1: リフレッシュトークン取得
        print("Step 1: リフレッシュトークン取得中...")
        refresh_url = "https://api.jquants.com/v1/token/auth_user"
        
        # JSON形式で送信
        refresh_data = {
            "mailaddress": MAIL_ADDRESS,
            "password": PASSWORD
        }
        
        response = requests.post(refresh_url, json=refresh_data)
        print(f"リフレッシュトークン取得レスポンス: {response.status_code}")
        print(f"レスポンス内容: {response.text[:200]}")
        
        if response.status_code != 200:
            print(f"❌ リフレッシュトークン取得失敗: {response.status_code}")
            return False
        
        refresh_token = response.json()["refreshToken"]
        print("✅ リフレッシュトークン取得成功")
        
        # Step 2: IDトークン取得
        print(f"リフレッシュトークン: {refresh_token[:50]}...")
        print("Step 2: IDトークン取得中...")
        id_token_url = "https://api.jquants.com/v1/token/auth_refresh"
        
        # J-Quants公式サンプルに従ったJSON送信
        headers = {'Content-Type': 'application/json'}
        id_token_payload = json.dumps({"refreshtoken": refresh_token})
        
        print(f"送信ペイロード: {id_token_payload[:100]}...")
        response = requests.post(id_token_url, data=id_token_payload, headers=headers)
        print(f"IDトークン取得レスポンス: {response.status_code}")
        print(f"レスポンス内容: {response.text[:200]}")
        
        if response.status_code != 200:
            print(f"❌ IDトークン取得失敗: {response.status_code}")
            return False
        
        id_token = response.json()["idToken"]
        print("✅ IDトークン取得成功")
        
        # Step 3: APIテスト（上場銘柄一覧取得）
        print("Step 3: API機能テスト...")
        test_url = "https://api.jquants.com/v1/listed/info"
        headers = {'Authorization': f'Bearer {id_token}'}
        params = {'code': '7203'}  # トヨタ
        
        response = requests.get(test_url, headers=headers, params=params)
        print(f"APIテストレスポンス: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ APIテスト成功: {len(data.get('info', []))}件取得")
            if data.get('info'):
                print(f"サンプル: {data['info'][0]}")
            return True
        else:
            print(f"❌ APIテスト失敗: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ エラー: {e}")
        return False

if __name__ == "__main__":
    success = test_jquants_auth()
    print(f"\n{'✅ 認証成功' if success else '❌ 認証失敗'}")