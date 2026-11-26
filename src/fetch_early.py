"""2001〜2009年（7〜9月）の追加取得。**2009年から順に遡って**取得し、年ごとに保存する。

- 既存の data/hourly.csv・region_hourly.csv（2010〜2025年）は上書きしない。保存先は data/early/。
- 那覇・名護（官署）と、全期間（2001〜2025年）で観測している周辺地点だけを取得する:
  糸数・胡屋・読谷・東・本部・奥（安次嶺・宮城島・国頭は、年によって未観測のため使わない）。
- 取得済みの日は再取得しない（data/raw・data/raw_region のキャッシュ）。途中で止めても、続きから再開できる。
- 出典: 気象庁「過去の気象データ検索」（https://www.data.jma.go.jp/stats/etrn/）。
使い方: .venv/bin/python src/fetch_early.py [--years 2009 2008 ...] [--sleep 1.0]
"""
import argparse
import calendar
from pathlib import Path

import pandas as pd

import fetch_hourly as fh
import fetch_region as fr

BASE = Path(__file__).resolve().parent.parent
EARLY = BASE / "data" / "early"
REGION_STATIONS = ["itokazu", "goya", "yomitan", "higashi", "motobu", "oku"]


def fetch_year(y: int, sleep: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    hs, rs = [], []
    for st in fh.STATIONS:
        for m in (7, 8, 9):
            for d in range(1, calendar.monthrange(y, m)[1] + 1):
                df = fh.parse_day(fh.fetch_day(st, y, m, d, sleep), y, m, d)
                df["station"] = st
                hs.append(df)
    h = pd.concat(hs, ignore_index=True)
    for c in [c for c in h.columns if c not in ("time", "station", "wdir")]:
        h[c] = fh.to_num(h[c])
    for st in REGION_STATIONS:
        for m in (7, 8, 9):
            for d in range(1, calendar.monthrange(y, m)[1] + 1):
                df = fr.fetch_day(st, y, m, d, sleep)
                df["time"] = pd.to_datetime(f"{y}-{m:02d}-{d:02d}") + pd.to_timedelta(df["hour"], unit="h")
                df["station"] = st
                rs.append(df)
    r = pd.concat(rs, ignore_index=True)
    r["rain"] = fr.to_rain(r["raw"])
    return h, r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=list(range(2009, 2000, -1)))
    ap.add_argument("--sleep", type=float, default=1.0)
    a = ap.parse_args()
    EARLY.mkdir(parents=True, exist_ok=True)
    for y in a.years:
        f_h, f_r = EARLY / f"hourly_{y}.csv", EARLY / f"region_hourly_{y}.csv"
        if f_h.exists() and f_r.exists():
            print(y, "取得済み", flush=True)
            continue
        h, r = fetch_year(y, a.sleep)
        h.to_csv(f_h, index=False)
        r.to_csv(f_r, index=False)
        print(y, "完了", h.shape, r.shape, flush=True)


if __name__ == "__main__":
    main()
