#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3モデル直近精度モニター
最新のV3実行結果から直近精度を算出し、アラートを表示
"""

import sys
from pathlib import Path
import joblib
import numpy as np
from datetime import datetime
import json
import glob
import logging

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class V3PerformanceMonitor:
    """V3モデル性能モニター"""
    
    # 閾値設定
    PRECISION_WARNING_THRESHOLD = 0.75  # 75%以下で警告
    PRECISION_ALERT_THRESHOLD = 0.70   # 70%以下でアラート
    ACCURACY_WARNING_THRESHOLD = 0.75   # 75%以下で警告
    ACCURACY_ALERT_THRESHOLD = 0.70     # 70%以下でアラート
    
    def __init__(self, results_dir: str = "models/enhanced_v3"):
        self.results_dir = Path(results_dir)
        self.snapshots_dir = Path("production_reports/performance_snapshots")
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
    
    def find_latest_results_file(self) -> Path:
        """最新のV3結果ファイルを検出"""
        pattern = str(self.results_dir / "enhanced_results_v3_*.joblib")
        files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(f"V3結果ファイルが見つかりません: {pattern}")
        
        # ファイル名のタイムスタンプでソート（最新が最後）
        latest_file = max(files, key=lambda x: Path(x).stat().st_mtime)
        return Path(latest_file)
    
    def load_previous_snapshot(self) -> dict:
        """前回のスナップショットを読み込み"""
        pattern = str(self.snapshots_dir / "v3_metrics_*.json")
        files = glob.glob(pattern)
        if not files:
            return None
        
        try:
            latest_snapshot = max(files, key=lambda x: Path(x).stat().st_mtime)
            with open(latest_snapshot, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"前回スナップショット読み込みエラー: {e}")
            return None
    
    def analyze_recent_performance(self, results_file: Path, windows: int = 6) -> dict:
        """直近の性能を分析"""
        try:
            data = joblib.load(results_file)
            wfo_results = data.get("wfo_results", [])
            
            if not wfo_results:
                return {
                    "error": "wfo_results が見つかりません",
                    "source_file": str(results_file)
                }
            
            # 直近N区間を抽出
            recent = wfo_results[-windows:] if len(wfo_results) >= windows else wfo_results
            
            # 精度と適合率を計算
            accuracies = np.array([r.get("accuracy", 0) for r in recent])
            precisions = np.array([r.get("precision", 0) for r in recent])
            
            # 最終モデルの精度も取得
            final_accuracy = data.get("final_model_accuracy", 0)
            wfo_mean_accuracy = data.get("wfo_mean_accuracy", 0)
            
            metrics = {
                "source_file": str(results_file),
                "analysis_date": datetime.now().isoformat(),
                "windows_analyzed": len(recent),
                "total_windows": len(wfo_results),
                
                # 直近N区間の統計
                "recent_mean_accuracy": float(accuracies.mean()),
                "recent_mean_precision": float(precisions.mean()),
                "recent_precision_min": float(precisions.min()),
                "recent_precision_max": float(precisions.max()),
                "recent_accuracy_min": float(accuracies.min()),
                "recent_accuracy_max": float(accuracies.max()),
                
                # 全体統計
                "final_model_accuracy": float(final_accuracy),
                "wfo_mean_accuracy": float(wfo_mean_accuracy),
                
                # 最新区間の詳細
                "last_period": recent[-1].get("period", "N/A") if recent else "N/A",
                "last_accuracy": float(accuracies[-1]) if len(accuracies) > 0 else 0,
                "last_precision": float(precisions[-1]) if len(precisions) > 0 else 0,
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"性能分析エラー: {e}")
            return {"error": str(e), "source_file": str(results_file)}
    
    def compare_with_previous(self, current: dict, previous: dict) -> dict:
        """前回と比較して変化を検出"""
        if not previous or "error" in previous:
            return {"status": "no_previous_data"}
        
        comparison = {
            "status": "normal",
            "precision_change": 0.0,
            "accuracy_change": 0.0,
            "precision_declined": False,
            "accuracy_declined": False,
        }
        
        if "recent_mean_precision" in current and "recent_mean_precision" in previous:
            precision_change = current["recent_mean_precision"] - previous["recent_mean_precision"]
            comparison["precision_change"] = float(precision_change)
            comparison["precision_declined"] = precision_change < -0.02  # 2%以上低下
            
        if "recent_mean_accuracy" in current and "recent_mean_accuracy" in previous:
            accuracy_change = current["recent_mean_accuracy"] - previous["recent_mean_accuracy"]
            comparison["accuracy_change"] = float(accuracy_change)
            comparison["accuracy_declined"] = accuracy_change < -0.02  # 2%以上低下
        
        if comparison["precision_declined"] or comparison["accuracy_declined"]:
            comparison["status"] = "declining"
        
        return comparison
    
    def check_alert_conditions(self, metrics: dict) -> dict:
        """アラート条件をチェック"""
        alerts = {
            "has_warning": False,
            "has_alert": False,
            "warnings": [],
            "alerts": []
        }
        
        if "error" in metrics:
            alerts["alerts"].append(f"❌ エラー: {metrics['error']}")
            alerts["has_alert"] = True
            return alerts
        
        recent_precision = metrics.get("recent_mean_precision", 0)
        recent_accuracy = metrics.get("recent_mean_accuracy", 0)
        last_precision = metrics.get("last_precision", 0)
        last_accuracy = metrics.get("last_accuracy", 0)
        
        # Precisionチェック
        if recent_precision < self.PRECISION_ALERT_THRESHOLD:
            alerts["alerts"].append(
                f"🚨 緊急: 直近{metrics.get('windows_analyzed', 6)}区間の平均Precisionが"
                f"{recent_precision:.1%}と非常に低いです（閾値: {self.PRECISION_ALERT_THRESHOLD:.1%}）"
            )
            alerts["has_alert"] = True
        elif recent_precision < self.PRECISION_WARNING_THRESHOLD:
            alerts["warnings"].append(
                f"⚠️  警告: 直近{metrics.get('windows_analyzed', 6)}区間の平均Precisionが"
                f"{recent_precision:.1%}と低めです（閾値: {self.PRECISION_WARNING_THRESHOLD:.1%}）"
            )
            alerts["has_warning"] = True
        
        if last_precision < self.PRECISION_ALERT_THRESHOLD:
            alerts["alerts"].append(
                f"🚨 緊急: 最新区間のPrecisionが{last_precision:.1%}と非常に低いです"
            )
            alerts["has_alert"] = True
        
        # Accuracyチェック
        if recent_accuracy < self.ACCURACY_ALERT_THRESHOLD:
            alerts["alerts"].append(
                f"🚨 緊急: 直近{metrics.get('windows_analyzed', 6)}区間の平均Accuracyが"
                f"{recent_accuracy:.1%}と非常に低いです（閾値: {self.ACCURACY_ALERT_THRESHOLD:.1%}）"
            )
            alerts["has_alert"] = True
        elif recent_accuracy < self.ACCURACY_WARNING_THRESHOLD:
            alerts["warnings"].append(
                f"⚠️  警告: 直近{metrics.get('windows_analyzed', 6)}区間の平均Accuracyが"
                f"{recent_accuracy:.1%}と低めです（閾値: {self.ACCURACY_WARNING_THRESHOLD:.1%}）"
            )
            alerts["has_warning"] = True
        
        if last_accuracy < self.ACCURACY_ALERT_THRESHOLD:
            alerts["alerts"].append(
                f"🚨 緊急: 最新区間のAccuracyが{last_accuracy:.1%}と非常に低いです"
            )
            alerts["has_alert"] = True
        
        return alerts
    
    def display_results(self, metrics: dict, comparison: dict, alerts: dict):
        """結果をターミナルに表示"""
        print("\n" + "="*70)
        print("📊 V3モデル 直近精度モニター")
        print("="*70)
        
        if "error" in metrics:
            print(f"\n❌ エラー: {metrics['error']}")
            return
        
        # 基本情報
        print(f"\n📅 分析日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 結果ファイル: {Path(metrics['source_file']).name}")
        print(f"📊 分析区間数: {metrics.get('windows_analyzed', 0)}区間 / 全{metrics.get('total_windows', 0)}区間")
        
        # 直近N区間の統計
        print(f"\n{'='*70}")
        print("📈 直近精度統計（直近6区間）")
        print(f"{'='*70}")
        print(f"  平均 Precision: {metrics.get('recent_mean_precision', 0):.4f} ({metrics.get('recent_mean_precision', 0):.1%})")
        print(f"  Precision範囲: {metrics.get('recent_precision_min', 0):.4f} 〜 {metrics.get('recent_precision_max', 0):.4f}")
        print(f"  平均 Accuracy:  {metrics.get('recent_mean_accuracy', 0):.4f} ({metrics.get('recent_mean_accuracy', 0):.1%})")
        print(f"  Accuracy範囲:  {metrics.get('recent_accuracy_min', 0):.4f} 〜 {metrics.get('recent_accuracy_max', 0):.4f}")
        
        # 最新区間
        print(f"\n📌 最新区間 ({metrics.get('last_period', 'N/A')}):")
        print(f"  Precision: {metrics.get('last_precision', 0):.4f} ({metrics.get('last_precision', 0):.1%})")
        print(f"  Accuracy:  {metrics.get('last_accuracy', 0):.4f} ({metrics.get('last_accuracy', 0):.1%})")
        
        # 全体統計
        print(f"\n{'='*70}")
        print("🎯 全体統計")
        print(f"{'='*70}")
        print(f"  最終モデル精度: {metrics.get('final_model_accuracy', 0):.4f} ({metrics.get('final_model_accuracy', 0):.1%})")
        print(f"  WFO平均精度:    {metrics.get('wfo_mean_accuracy', 0):.4f} ({metrics.get('wfo_mean_accuracy', 0):.1%})")
        
        # 前回との比較
        if comparison.get("status") != "no_previous_data":
            print(f"\n{'='*70}")
            print("📊 前回実行との比較")
            print(f"{'='*70}")
            precision_change = comparison.get("precision_change", 0)
            accuracy_change = comparison.get("accuracy_change", 0)
            
            precision_icon = "📉" if precision_change < 0 else "📈" if precision_change > 0 else "➡️"
            accuracy_icon = "📉" if accuracy_change < 0 else "📈" if accuracy_change > 0 else "➡️"
            
            print(f"  Precision変化: {precision_icon} {precision_change:+.4f} ({precision_change:+.1%})")
            print(f"  Accuracy変化:  {accuracy_icon} {accuracy_change:+.4f} ({accuracy_change:+.1%})")
            
            if comparison.get("precision_declined") or comparison.get("accuracy_declined"):
                print(f"\n⚠️  精度が2%以上低下しています。モデルの再評価を検討してください。")
        
        # アラート表示
        if alerts.get("has_alert") or alerts.get("has_warning"):
            print(f"\n{'='*70}")
            if alerts.get("has_alert"):
                print("🚨 緊急アラート")
                print(f"{'='*70}")
                for alert in alerts.get("alerts", []):
                    print(f"  {alert}")
            
            if alerts.get("has_warning"):
                print(f"\n⚠️  警告")
                print(f"{'='*70}")
                for warning in alerts.get("warnings", []):
                    print(f"  {warning}")
        else:
            print(f"\n{'='*70}")
            print("✅ 精度は正常範囲内です")
            print(f"{'='*70}")
        
        print("\n" + "="*70)
    
    def save_snapshot(self, metrics: dict):
        """スナップショットを保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_file = self.snapshots_dir / f"v3_metrics_{timestamp}.json"
        
        try:
            with open(snapshot_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            logger.info(f"スナップショット保存: {snapshot_file}")
        except Exception as e:
            logger.error(f"スナップショット保存エラー: {e}")
    
    def run(self, windows: int = 6):
        """モニター実行"""
        try:
            # 最新結果ファイルを検出
            results_file = self.find_latest_results_file()
            logger.info(f"最新結果ファイル: {results_file.name}")
            
            # 性能分析
            metrics = self.analyze_recent_performance(results_file, windows)
            
            if "error" in metrics:
                print(f"\n❌ エラー: {metrics['error']}")
                return False
            
            # 前回スナップショットと比較
            previous = self.load_previous_snapshot()
            comparison = self.compare_with_previous(metrics, previous)
            
            # アラートチェック
            alerts = self.check_alert_conditions(metrics)
            
            # 結果表示
            self.display_results(metrics, comparison, alerts)
            
            # スナップショット保存
            self.save_snapshot(metrics)
            
            # アラートがある場合は非ゼロ終了コードを返す
            return not alerts.get("has_alert", False)
            
        except Exception as e:
            logger.error(f"モニター実行エラー: {e}")
            print(f"\n❌ モニター実行エラー: {e}")
            return False

def main():
    """メイン実行"""
    import argparse
    
    parser = argparse.ArgumentParser(description="V3モデル直近精度モニター")
    parser.add_argument("--windows", type=int, default=6, help="分析対象の直近区間数")
    parser.add_argument("--results-dir", type=str, default="models/enhanced_v3", help="結果ファイルディレクトリ")
    
    args = parser.parse_args()
    
    monitor = V3PerformanceMonitor(results_dir=args.results_dir)
    success = monitor.run(windows=args.windows)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

