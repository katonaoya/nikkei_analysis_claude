#!/usr/bin/env python3
"""
Simple test to understand J-Quants API authentication flow
"""

import requests
import json
from utils.config import get_config

def test_jquants_auth_flow():
    """Test the basic J-Quants authentication flow"""
    print("J-Quants API 認証フロー確認")
    print("=" * 40)
    
    config = get_config()
    base_url = "https://api.jquants.com"
    
    mail_address = config.get('api.jquants.mail_address')
    password = config.get('api.jquants.password')
    
    print(f"Email: {mail_address}")
    print(f"Base URL: {base_url}")
    
    # Step 1: Initial password authentication
    print("\n1. パスワード認証テスト...")
    auth_data = {
        'mailaddress': mail_address,
        'password': password
    }
    
    try:
        response = requests.post(
            f"{base_url}/v1/token/auth_user",
            json=auth_data,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            refresh_token = data.get('refreshToken')
            print(f"✅ 認証成功")
            print(f"Refresh Token: {refresh_token[:20]}..." if refresh_token else "No refresh token")
            
            # Step 2: Get ID token using refresh token
            if refresh_token:
                print("\n2. IDトークン取得テスト...")
                
                # Try different payload formats
                print("  テスト1: json形式...")
                refresh_data = {'refreshtoken': refresh_token}
                
                token_response = requests.post(
                    f"{base_url}/v1/token/auth_refresh",
                    json=refresh_data,
                    headers={'Content-Type': 'application/json'},
                    timeout=30
                )
                
                print(f"  Status: {token_response.status_code}")
                print(f"  Response: {token_response.text[:200]}...")
                
                if token_response.status_code != 200:
                    print("  テスト2: data形式...")
                    token_response = requests.post(
                        f"{base_url}/v1/token/auth_refresh",
                        data=refresh_data,
                        headers={'Content-Type': 'application/x-www-form-urlencoded'},
                        timeout=30
                    )
                    
                    print(f"  Status: {token_response.status_code}")
                    print(f"  Response: {token_response.text[:200]}...")
                
                if token_response.status_code != 200:
                    print("  テスト3: 別のフィールド名...")
                    refresh_data2 = {'refreshToken': refresh_token}
                    token_response = requests.post(
                        f"{base_url}/v1/token/auth_refresh",
                        json=refresh_data2,
                        headers={'Content-Type': 'application/json'},
                        timeout=30
                    )
                    
                    print(f"  Status: {token_response.status_code}")
                    print(f"  Response: {token_response.text[:200]}...")
                
                print(f"Status Code: {token_response.status_code}")
                print(f"Response: {token_response.text}")
                
                if token_response.status_code == 200:
                    token_data = token_response.json()
                    id_token = token_data.get('idToken')
                    print(f"✅ トークン取得成功")
                    print(f"ID Token: {id_token[:20]}..." if id_token else "No ID token")
                    
                    # Step 3: Test API call with ID token
                    if id_token:
                        print("\n3. APIコールテスト...")
                        headers = {
                            'Authorization': f'Bearer {id_token}',
                            'Content-Type': 'application/json'
                        }
                        
                        # Try to get listed info
                        api_response = requests.get(
                            f"{base_url}/v1/listed/info",
                            headers=headers,
                            timeout=30
                        )
                        
                        print(f"Status Code: {api_response.status_code}")
                        if api_response.status_code == 200:
                            print("✅ APIコール成功")
                            data = api_response.json()
                            if 'info' in data:
                                print(f"データ数: {len(data['info'])}")
                        else:
                            print(f"Response: {api_response.text}")
                else:
                    print("❌ トークン取得失敗")
            else:
                print("❌ リフレッシュトークンが取得できませんでした")
        else:
            print("❌ 認証失敗")
            
    except Exception as e:
        print(f"❌ エラー: {str(e)}")

if __name__ == "__main__":
    test_jquants_auth_flow()