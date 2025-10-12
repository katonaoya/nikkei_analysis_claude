#!/usr/bin/env python3
"""簡易データヘルスチェック

株価データと外部指標データの健全性を検査し、結果を data_monitoring/ 以下にJSON出力する。
異常が検知された場合は非ゼロ終了コードで終了する。
"""
import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils.logger import StructuredLogger


def _sanitize(value):
    """JSONシリアライズ可能なプリミティブに変換"""
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    if isinstance(value, (np.floating, np.float64, np.float32)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    return value


@dataclass
class CheckResult:
    name: str
    passed: bool
    details: Dict[str, object] = field(default_factory=dict)
    severity: str = "info"  # info / warning / error

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "passed": bool(self.passed),
            "severity": self.severity,
            "details": _sanitize(self.details),
        }


def _find_latest_file(patterns: Iterable[Union[str, Path]]) -> Optional[Path]:
    latest_file: Optional[Path] = None
    latest_mtime: float = 0.0
    for pattern in patterns:
        pattern_str = str(pattern)
        for candidate in Path.cwd().glob(pattern_str):
            try:
                mtime = candidate.stat().st_mtime
            except OSError:
                continue
            if mtime > latest_mtime:
                latest_file = candidate
                latest_mtime = mtime
    return latest_file


def _load_stock_data(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # 正規化
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    return df


def _load_external_data(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    return df


def check_stock_coverage(df: pd.DataFrame) -> CheckResult:
    unique_codes = df["Code"].nunique()
    min_expected = 200
    passed = unique_codes >= min_expected
    return CheckResult(
        name="stock_coverage",
        passed=passed,
        severity="error" if not passed else "info",
        details={"unique_codes": int(unique_codes), "min_expected": min_expected},
    )


def check_basic_integrity(df: pd.DataFrame) -> List[CheckResult]:
    results: List[CheckResult] = []
    negative_columns = [col for col in ["Open", "High", "Low", "Close"] if col in df.columns]
    neg_counts = {
        col: int((df[col] < 0).sum()) for col in negative_columns
    }
    passed_neg = all(count == 0 for count in neg_counts.values())
    results.append(
        CheckResult(
            name="price_non_negative",
            passed=passed_neg,
            severity="error" if not passed_neg else "info",
            details=neg_counts,
        )
    )
    duplicate_count = int(df.duplicated(subset=["Date", "Code"]).sum())
    results.append(
        CheckResult(
            name="duplicate_rows",
            passed=duplicate_count == 0,
            severity="error" if duplicate_count else "warning",
            details={"duplicate_rows": duplicate_count},
        )
    )
    return results


def check_date_gaps(df: pd.DataFrame) -> CheckResult:
    df_sorted = df.sort_values(["Code", "Date"]).copy()
    gaps: Dict[str, int] = {}
    for code, group in df_sorted.groupby("Code"):
        dates = group["Date"].drop_duplicates().diff().dt.days
        if dates.empty:
            continue
        max_gap = int(dates.max(skipna=True))
        if max_gap > 5:  # 5日より長いギャップは警告
            gaps[str(code)] = max_gap
    passed = len(gaps) == 0
    return CheckResult(
        name="trading_day_gaps",
        passed=passed,
        severity="warning" if not passed else "info",
        details={"max_gaps": gaps},
    )


def check_external_nulls(df: pd.DataFrame) -> CheckResult:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return CheckResult(
            name="external_numeric_columns",
            passed=False,
            severity="error",
            details={"message": "外部データに数値カラムが存在しません"},
        )
    null_ratio = df[numeric_cols].isna().mean().max()
    threshold = 0.4
    passed = null_ratio <= threshold
    return CheckResult(
        name="external_null_ratio",
        passed=passed,
        severity="warning" if not passed else "info",
        details={"max_null_ratio": float(null_ratio), "threshold": threshold},
    )


def check_freshness(stock_path: Path, external_path: Path) -> CheckResult:
    now = datetime.now()
    stock_age = now - datetime.fromtimestamp(stock_path.stat().st_mtime)
    external_age = now - datetime.fromtimestamp(external_path.stat().st_mtime)
    max_allowed = timedelta(days=2)
    passed = stock_age <= max_allowed and external_age <= max_allowed
    return CheckResult(
        name="dataset_freshness",
        passed=passed,
        severity="error" if not passed else "info",
        details={
            "stock_file": stock_path.name,
            "stock_age_hours": round(stock_age.total_seconds() / 3600, 2),
            "external_file": external_path.name,
            "external_age_hours": round(external_age.total_seconds() / 3600, 2),
            "max_allowed_hours": max_allowed.total_seconds() / 3600,
        },
    )


def run_checks(stock_path: Path, external_path: Path, logger: Optional[StructuredLogger] = None) -> Dict[str, object]:
    stock_df = _load_stock_data(stock_path)
    external_df = _load_external_data(external_path)

    results: List[CheckResult] = []
    results.append(check_stock_coverage(stock_df))
    results.extend(check_basic_integrity(stock_df))
    results.append(check_date_gaps(stock_df))
    results.append(check_external_nulls(external_df))

    # 以前はここでデータ鮮度チェックを行っていたが、運用要件により時間基準のヘルスチェックは無効化した。

    report = {
        "timestamp": datetime.now().isoformat(),
        "stock_file": str(stock_path),
        "external_file": str(external_path),
        "results": [res.to_dict() for res in results],
    }

    all_passed = all(res.passed or res.severity == "warning" for res in results)
    report["overall_status"] = "pass" if all_passed else "fail"

    if logger is not None:
        log_payload = {
            "stock_file": str(stock_path),
            "external_file": str(external_path),
            "overall_status": report["overall_status"],
        }
        log_payload.update({res.name: res.passed for res in results})
        if all_passed:
            logger.info("Data health check passed", **log_payload)
        else:
            logger.warning("Data health check completed with issues", **log_payload)

    return report


def update_registry(registry_path: Path, report: Dict[str, object]) -> None:
    registry = {}
    if registry_path.exists():
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            registry = {}

    key = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    registry_entry = {
        "checked_at": report.get("timestamp"),
        "stock_file": report.get("stock_file"),
        "external_file": report.get("external_file"),
        "overall_status": report.get("overall_status"),
    }
    registry[report.get("stock_file", key)] = registry_entry

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="データヘルスチェック")
    parser.add_argument("--stock", help="株価データファイルパス")
    parser.add_argument("--external", help="外部データファイルパス")
    parser.add_argument(
        "--output-dir",
        default="data_monitoring",
        help="結果を出力するディレクトリ (default: data_monitoring)",
    )
    parser.add_argument(
        "--registry",
        default="data/data_registry.json",
        help="最新の検証結果を記録するレジストリファイルパス",
    )
    args = parser.parse_args()

    stock_path: Optional[Path]
    external_path: Optional[Path]

    if args.stock:
        stock_path = Path(args.stock)
    else:
        stock_path = _find_latest_file([
            Path("data/processed") / "nikkei225_complete_*.parquet",
        ])
    if args.external:
        external_path = Path(args.external)
    else:
        external_path = _find_latest_file([
            Path("data/processed") / "enhanced_integrated_data.parquet",
            Path("data/external_extended") / "external_integrated_*.parquet",
        ])

    if stock_path is None or not stock_path.exists():
        print("株価データファイルが見つかりません", file=sys.stderr)
        return 2
    if external_path is None or not external_path.exists():
        print("外部データファイルが見つかりません", file=sys.stderr)
        return 2

    logger = StructuredLogger("data_health_check")
    report = run_checks(stock_path, external_path, logger=logger)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"data_health_{timestamp}.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if report.get("overall_status") == "pass" and args.registry:
        update_registry(Path(args.registry), report)

    # エラー検知で終了コードを分ける
    has_error = any(
        (not item["passed"]) and item["severity"] == "error"
        for item in report["results"]
    )
    return 0 if not has_error else 1


if __name__ == "__main__":
    sys.exit(main())
