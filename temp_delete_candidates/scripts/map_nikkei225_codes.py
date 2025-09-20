#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Map Nikkei 225 4-digit codes to J-Quants codes using /v1/listed/info.

- Input CSV (default): docment/ユーザー情報/nikkei225_4digit_list.csv
  Columns: Code4,CompanyName
- Output CSV (default): docment/ユーザー情報/nikkei225_jquants_codes.csv
  Columns: Code4,CompanyName,CodeJQ,CompanyNameJQ,MatchType,Notes

Requirements:
- No external libraries required (uses standard library only)
- Provide ID token via environment variable: ID_TOKEN

Usage:
  export ID_TOKEN="<your idToken>"
  python scripts/map_nikkei225_codes.py \
    --input docment/ユーザー情報/nikkei225_4digit_list.csv \
    --output docment/ユーザー情報/nikkei225_jquants_codes.csv

Notes:
- This script does not call refresh token endpoint; obtain idToken beforehand.
- Matching logic tries: (1) exact company name (normalized), (2) JQ Code endswith 4-digit,
  (3) fuzzy name match. Review output and adjust names if needed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from difflib import get_close_matches
from typing import Dict, List, Optional, Tuple

BASE_URL = "https://api.jquants.com/v1"


def normalize_name(name: str) -> str:
    if name is None:
        return ""
    # Normalize width (e.g., ＮＴＴ -> NTT), remove spaces/typographic dots
    text = unicodedata.normalize("NFKC", str(name))
    for ch in [" ", "\u3000", "・", "︓", "：", "株式會社", "株式会社"]:
        text = text.replace(ch, "")
    # Common abbreviations unification
    text = text.replace("ＨＤ", "ホールディングス")
    text = text.replace("HD", "ホールディングス")
    text = text.replace("グループ", "ＧＲＰ")  # reduce variance
    return text.strip().lower()


def http_get_json(path: str, headers: Dict[str, str], params: Optional[Dict[str, str]] = None) -> Dict:
    url = f"{BASE_URL}/{path}"
    if params:
        qs = urllib.parse.urlencode(params)
        url = f"{url}?{qs}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body)


def fetch_all_listed_info(id_token: str, sleep_sec: float = 0.5) -> List[Dict]:
    headers = {"Authorization": f"Bearer {id_token}"}
    params: Dict[str, str] = {}
    all_rows: List[Dict] = []
    while True:
        data = http_get_json("listed/info", headers, params)
        rows = data.get("info", [])
        all_rows.extend(rows)
        pagination_key = data.get("pagination_key")
        if not pagination_key:
            break
        params = {"pagination_key": pagination_key}
        time.sleep(sleep_sec)
    return all_rows


def build_name_index(rows: List[Dict]) -> Dict[str, Dict]:
    index: Dict[str, Dict] = {}
    for r in rows:
        name = r.get("CompanyName") or r.get("CompanyNameShort") or ""
        key = normalize_name(name)
        if key and key not in index:
            index[key] = r
    return index


def map_single(row: Dict, name_index: Dict[str, Dict], all_rows: List[Dict]) -> Tuple[str, str, str, str, str]:
    code4 = (row.get("Code4") or row.get("Code") or "").strip()
    input_name = (row.get("CompanyName") or "").strip()

    # 1) exact name match (normalized)
    key = normalize_name(input_name)
    if key in name_index:
        hit = name_index[key]
        return code4, input_name, hit.get("Code", ""), hit.get("CompanyName", ""), "name_exact"

    # 2) code suffix match (JQ code endswith 4 digits)
    candidates = [r for r in all_rows if str(r.get("Code", "")).endswith(code4)] if code4 else []
    if len(candidates) == 1:
        hit = candidates[0]
        return code4, input_name, hit.get("Code", ""), hit.get("CompanyName", ""), "code_suffix"

    # 3) fuzzy name match
    all_names = [r.get("CompanyName", "") for r in all_rows]
    matches = get_close_matches(input_name, all_names, n=1, cutoff=0.95)
    if matches:
        m = matches[0]
        for r in all_rows:
            if r.get("CompanyName") == m:
                return code4, input_name, r.get("Code", ""), r.get("CompanyName", ""), "name_fuzzy"

    return code4, input_name, "", "", "unmatched"


def run_mapping(input_csv: str, output_csv: str, id_token: str) -> None:
    listed_rows = fetch_all_listed_info(id_token)
    name_index = build_name_index(listed_rows)

    with open(input_csv, "r", encoding="utf-8") as f_in, open(output_csv, "w", encoding="utf-8", newline="") as f_out:
        reader = csv.DictReader(f_in)
        fieldnames = ["Code4", "CompanyName", "CodeJQ", "CompanyNameJQ", "MatchType", "Notes"]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for src in reader:
            code4, in_name, code_jq, name_jq, match_type = map_single(src, name_index, listed_rows)
            writer.writerow({
                "Code4": code4,
                "CompanyName": in_name,
                "CodeJQ": code_jq,
                "CompanyNameJQ": name_jq,
                "MatchType": match_type,
                "Notes": "" if match_type != "unmatched" else "確認要: 社名表記揺れ/旧社名/上場区分"
            })


def main() -> int:
    parser = argparse.ArgumentParser(description="Map Nikkei 225 4-digit codes to J-Quants codes")
    parser.add_argument("--input", default="docment/ユーザー情報/nikkei225_4digit_list.csv")
    parser.add_argument("--output", default="docment/ユーザー情報/nikkei225_jquants_codes.csv")
    args = parser.parse_args()

    id_token = os.getenv("ID_TOKEN", "").strip()
    if not id_token:
        print("[ERROR] ID_TOKEN environment variable is required.", file=sys.stderr)
        print("Set it like: export ID_TOKEN=\"<idToken>\"", file=sys.stderr)
        return 2

    try:
        run_mapping(args.input, args.output, id_token)
        print(f"[OK] Mapping completed -> {args.output}")
        return 0
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
