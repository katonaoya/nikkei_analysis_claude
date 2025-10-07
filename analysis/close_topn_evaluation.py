#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""推奨銘柄のTOP Nおよび確率閾値による成績比較"""

import re
from bisect import bisect_right
from pathlib import Path
import sys

import pandas as pd
import argparse
import json

sys.path.append(str(Path(__file__).resolve().parents[1]))
from analysis.close_threshold_evaluation import parse_reports, load_price_data

PROB_PATTERN = re.compile(r"🎯 予測確率: (?P<prob>[0-9.]+)%")
REPORT_DIRS = [Path("production_reports/2025-09"), Path("production_reports/2025-10")]


def parse_reports_with_prob() -> pd.DataFrame:
    records = []
    for report_dir in REPORT_DIRS:
        if not report_dir.exists():
            continue
        for path in sorted(report_dir.glob('*.md')):
            text = path.read_text(encoding='utf-8')
            date_match = re.search(r"基準日付: (\d{4}-\d{2}-\d{2})", text) or re.search(r"対象日: (\d{4}-\d{2}-\d{2})", text)
            if not date_match:
                continue
            base_date = pd.to_datetime(date_match.group(1))
            blocks = text.split('=====================================\n🎯 推奨銘柄一覧\n=====================================\n')
            if len(blocks) < 2:
                continue
            lines = blocks[1].splitlines()
            current_rank = 0
            current_code = None
            current_prob = None
            current_name = None
            for line in lines:
                rank_match = re.match(r"(\d+)位: (.+) \((\d+)\)", line)
                if rank_match:
                    if current_code is not None and current_prob is not None:
                        records.append({
                            'report_date': base_date,
                            'company': current_name,
                            'code_raw': current_code,
                            'rank': current_rank,
                            'probability': current_prob
                        })
                    current_rank = int(rank_match.group(1))
                    current_name = rank_match.group(2).strip()
                    current_code = rank_match.group(3).strip()
                    current_prob = None
                    continue
                prob_match = PROB_PATTERN.search(line)
                if prob_match and current_code is not None:
                    current_prob = float(prob_match.group('prob')) / 100.0
            if current_code is not None and current_prob is not None:
                records.append({
                    'report_date': base_date,
                    'company': current_name,
                    'code_raw': current_code,
                    'rank': current_rank,
                    'probability': current_prob
                })
    return pd.DataFrame(records)


def attach_returns(df: pd.DataFrame) -> pd.DataFrame:
    price_df = load_price_data()
    code_groups = price_df.groupby('Code')
    code_dates = {code: grp['Date'].tolist() for code, grp in code_groups}

    rows = []
    for row in df.itertuples():
        code_int = int(float(row.code_raw))
        code = code_int * 10 if code_int < 10000 else code_int
        entry = dict(row._asdict())
        entry['code'] = code
        if code not in code_dates:
            entry['status'] = 'code_missing'
            rows.append(entry)
            continue
        grp = code_groups.get_group(code)
        base_rows = grp[grp['Date'] == row.report_date]
        if base_rows.empty:
            entry['status'] = 'base_missing'
            rows.append(entry)
            continue
        idx = bisect_right(code_dates[code], row.report_date.to_pydatetime())
        if idx >= len(code_dates[code]):
            entry['status'] = 'next_missing'
            rows.append(entry)
            continue
        base_close = base_rows['Close'].iloc[0]
        next_close = grp.iloc[idx]['Close']
        entry['status'] = 'OK'
        entry['return_next'] = next_close / base_close - 1
        rows.append(entry)
    return pd.DataFrame(rows)


def evaluate(df: pd.DataFrame, top_n: int, min_prob: float, transaction_cost: float, target_return: float) -> dict:
    valid = df[(df['status'] == 'OK') & (df['rank'] <= top_n) & (df['probability'] >= min_prob)].copy()
    if valid.empty:
        return {
            'top_n': top_n,
            'min_prob': min_prob,
            'count': 0,
            'hit_rate': 0.0,
            'avg_return': 0.0,
            'avg_net_return': 0.0
        }
    net = valid['return_next'] - transaction_cost
    hit_rate = (valid['return_next'] >= target_return).mean()
    avg_return = valid['return_next'].mean()
    avg_net_return = net.mean()
    return {
        'top_n': top_n,
        'min_prob': min_prob,
        'count': len(valid),
        'hit_rate': hit_rate,
        'avg_return': avg_return,
        'avg_net_return': avg_net_return
    }


def main():
    parser = argparse.ArgumentParser(description='TOP N/閾値評価')
    parser.add_argument('--tops', type=str, default='3,5', help='検証する上位N (カンマ区切り)')
    parser.add_argument('--probabilities', type=str, default='0.60,0.65', help='検証する確率閾値')
    parser.add_argument('--transaction-cost', type=float, default=0.0, help='片道コスト比率')
    parser.add_argument('--target-return', type=float, default=0.01, help='ヒットとみなすリターン閾値')
    parser.add_argument('--apply-config', action='store_true', help='best結果を設定ファイルに反映')
    args = parser.parse_args()

    report_df = parse_reports_with_prob()
    enriched = attach_returns(report_df)
    enriched.attrs['target_return'] = args.target_return

    tops = [int(x) for x in args.tops.split(',') if x]
    probs = [float(x) for x in args.probabilities.split(',') if x]

    results = []
    for top in tops:
        for prob in probs:
            res = evaluate(enriched, top, prob, args.transaction_cost, args.target_return)
            res['transaction_cost'] = args.transaction_cost
            results.append(res)

    df = pd.DataFrame(results)
    df.sort_values('avg_net_return', ascending=False, inplace=True)
    out_path = Path('reports/monitoring/topn_summary.csv')
    df.to_csv(out_path, index=False)
    print(f"TopN summary saved to {out_path}")

    if args.apply_config and not df.empty:
        best = df.iloc[0]
        cfg_path = Path('config/close_recommendation_config.json')
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
        else:
            cfg = {}
        cfg['top_n'] = int(best['top_n'])
        cfg['min_probability'] = float(best['min_prob'])
        cfg_path.write_text(json.dumps(cfg, indent=2))
        print(f"Config updated: top_n={int(best['top_n'])}, min_probability={best['min_prob']:.2f}")


if __name__ == '__main__':
    main()
