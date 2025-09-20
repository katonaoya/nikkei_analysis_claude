#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
株式市場営業日判定ユーティリティ
日本の祝日と市場休場日を考慮した営業日判定機能
"""

import pandas as pd
from datetime import datetime, timedelta
import numpy as np

class JapanMarketCalendar:
    """日本株式市場カレンダー"""
    
    # 2025年の日本の祝日（固定）
    # 本来はjpholidayライブラリを使うべきだが、簡易的に主要祝日をハードコード
    JAPAN_HOLIDAYS_2025 = [
        '2025-01-01',  # 元日
        '2025-01-02',  # 年始休暇
        '2025-01-03',  # 年始休暇
        '2025-01-13',  # 成人の日
        '2025-02-11',  # 建国記念の日
        '2025-02-23',  # 天皇誕生日
        '2025-02-24',  # 振替休日
        '2025-03-20',  # 春分の日
        '2025-04-29',  # 昭和の日
        '2025-05-03',  # 憲法記念日
        '2025-05-04',  # みどりの日
        '2025-05-05',  # こどもの日
        '2025-05-06',  # 振替休日
        '2025-07-21',  # 海の日
        '2025-08-11',  # 山の日
        '2025-09-15',  # 敬老の日
        '2025-09-23',  # 秋分の日
        '2025-10-13',  # スポーツの日
        '2025-11-03',  # 文化の日
        '2025-11-23',  # 勤労感謝の日
        '2025-11-24',  # 振替休日
        '2025-12-31',  # 大晦日（市場休場）
    ]
    
    @classmethod
    def is_market_open(cls, date):
        """指定日が市場開場日かどうか判定"""
        if isinstance(date, str):
            date = pd.to_datetime(date)
        
        # 週末チェック
        if date.weekday() >= 5:  # 土曜(5)または日曜(6)
            return False
        
        # 祝日チェック
        date_str = date.strftime('%Y-%m-%d')
        if date_str in cls.JAPAN_HOLIDAYS_2025:
            return False
        
        # 年末年始特別休場（12/31-1/3）
        if date.month == 12 and date.day >= 31:
            return False
        if date.month == 1 and date.day <= 3:
            return False
        
        return True
    
    @classmethod
    def get_last_market_day(cls, from_date=None):
        """指定日から遡って直近の営業日を取得"""
        if from_date is None:
            from_date = datetime.now()
        elif isinstance(from_date, str):
            from_date = pd.to_datetime(from_date)
        
        current_date = from_date
        
        # 最大30日遡る（長期休暇対応）
        for _ in range(30):
            if cls.is_market_open(current_date):
                return current_date
            current_date -= timedelta(days=1)
        
        raise ValueError(f"営業日が見つかりません: {from_date}")
    
    @classmethod
    def get_next_market_day(cls, from_date=None):
        """指定日から先の直近営業日を取得"""
        if from_date is None:
            from_date = datetime.now()
        elif isinstance(from_date, str):
            from_date = pd.to_datetime(from_date)
        
        current_date = from_date + timedelta(days=1)
        
        # 最大30日先まで探す
        for _ in range(30):
            if cls.is_market_open(current_date):
                return current_date
            current_date += timedelta(days=1)
        
        raise ValueError(f"営業日が見つかりません: {from_date}")
    
    @classmethod
    def get_target_date_for_analysis(cls, execution_time=None):
        """
        実行時刻に応じて分析対象日を決定
        - 平日15時以降: 当日データを使用
        - 平日15時前: 前営業日データを使用
        - 週末・祝日: 直近営業日データを使用
        """
        if execution_time is None:
            execution_time = datetime.now()
        elif isinstance(execution_time, str):
            execution_time = pd.to_datetime(execution_time)
        
        # 現在日が営業日かチェック
        if cls.is_market_open(execution_time):
            # 営業日の場合
            if execution_time.hour >= 15:
                # 15時以降なら当日データを使用
                return execution_time.date()
            else:
                # 15時前なら前営業日データを使用
                return cls.get_last_market_day(execution_time - timedelta(days=1)).date()
        else:
            # 休場日の場合は直近営業日を使用
            return cls.get_last_market_day(execution_time - timedelta(days=1)).date()