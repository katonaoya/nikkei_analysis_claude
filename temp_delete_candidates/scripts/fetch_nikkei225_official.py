#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fetch current Nikkei 225 constituents (4-digit TSE codes) from public web sources
and write to docment/ユーザー情報/nikkei225_4digit_list.csv.

- Tries multiple sources (HTML tables) and normalizes results
- Does not execute automatically; user runs it manually per instructions

Usage:
  python scripts/fetch_nikkei225_official.py \
    --output docment/ユーザー情報/nikkei225_4digit_list.csv

Notes:
- Uses pandas.read_html (pandas is already in requirements.txt)
- Review output manually for accuracy; sources may change HTML structure
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Tuple
import pandas as pd

CANDIDATE_URLS: List[str] = [
    # Public pages commonly listing Nikkei 225; order by reliability if known
    "https://ja.wikipedia.org/wiki/%E6%97%A5%E7%B5%8C%E5%B9%B3%E5%9D%87%E6%A0%AA%E4%BE%A1",
    "https://stocks.finance.yahoo.co.jp/stocks/index/?code=998407.O",
    "https://kasegulog.com/nikkei225/",
]


def extract_candidates_from_url(url: str) -> List[Tuple[str, str]]:
    tables = pd.read_html(url)
    candidates: List[Tuple[str, str]] = []

    for df in tables:
        cols = [str(c) for c in df.columns]
        # Heuristics for typical column names
        if any("コード" in c for c in cols) and any("銘柄" in c or "社名" in c for c in cols):
            code_col = next(c for c in cols if "コード" in c)
            name_col = next(c for c in cols if ("銘柄" in c or "社名" in c))
            sub = df[[code_col, name_col]].copy()
            sub.columns = ["Code", "CompanyName"]
            # Normalize code to 4-digit string if possible
            sub["Code"] = (
                sub["Code"].astype(str).str.extract(r"(\d{4})", expand=False).fillna("")
            )
            sub = sub[(sub["Code"] != "") & (sub["CompanyName"].astype(str) != "")]
            for _, r in sub.iterrows():
                candidates.append((r["Code"], str(r["CompanyName"])) )

        # Sometimes code/name have English headers
        elif any("Code" == c for c in cols) and any("Name" in c or "Company" in c for c in cols):
            code_col = next(c for c in cols if c == "Code")
            name_col = next(c for c in cols if ("Name" in c or "Company" in c))
            sub = df[[code_col, name_col]].copy()
            sub.columns = ["Code", "CompanyName"]
            sub["Code"] = (
                sub["Code"].astype(str).str.extract(r"(\d{4})", expand=False).fillna("")
            )
            sub = sub[(sub["Code"] != "") & (sub["CompanyName"].astype(str) != "")]
            for _, r in sub.iterrows():
                candidates.append((r["Code"], str(r["CompanyName"])) )

    return candidates


def consolidate(candidates: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    # Deduplicate by code, prefer first occurrence
    seen = set()
    out: List[Tuple[str, str]] = []
    for code, name in candidates:
        if code not in seen:
            seen.add(code)
            out.append((code, name))
    return out


def write_csv(pairs: List[Tuple[str, str]], output_path: str) -> None:
    df = pd.DataFrame(pairs, columns=["Code4", "CompanyName"])
    df = df.sort_values("Code4").reset_index(drop=True)
    df.to_csv(output_path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch Nikkei 225 constituents (4-digit codes)")
    parser.add_argument("--output", default="docment/ユーザー情報/nikkei225_4digit_list.csv")
    args = parser.parse_args()

    all_candidates: List[Tuple[str, str]] = []
    for url in CANDIDATE_URLS:
        try:
            pairs = extract_candidates_from_url(url)
            if pairs:
                all_candidates.extend(pairs)
        except Exception as e:
            print(f"[WARN] Failed to parse {url}: {e}")

    consolidated = consolidate(all_candidates)
    if len(consolidated) < 200:
        print(f"[ERROR] Only {len(consolidated)} codes found. Please check sources.")
        return 1

    write_csv(consolidated, args.output)
    print(f"[OK] Wrote {len(consolidated)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
