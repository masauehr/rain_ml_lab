"""風向・風速の1時間値を、沖縄本島の4つのアメダス（奥・宮城島・安次嶺・糸数）から取得する。三角形法の収束量に使う。

- 出典: 気象庁「過去の気象データ検索」 https://www.data.jma.go.jp/stats/etrn/ （hourly_a1）
- 1日1ファイル（hour, wind, wdir）を data/raw_wind/{地点}/{YYYYMMDD}.csv に保存する。取得済みは再取得しない。
- 那覇・名護の風は fetch_hourly.py（data/hourly.csv）で取得済み。
- 使い方: .venv/bin/python src/fetch_wind.py [--years 2010 ... 2025] [--sleep 1.0]
"""
import argparse
import calendar
import io
import time
from pathlib import Path

import pandas as pd
import requests

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw_wind"
STATIONS = {"oku": ("91", "0901"), "miyagijima": ("91", "1596"), "ashimine": ("91", "1482"), "itokazu": ("91", "0909")}
URL = ("https://www.data.jma.go.jp/stats/etrn/view/hourly_a1.php"
       "?prec_no={p}&block_no={b}&year={y}&month={m}&day={d}&view=")


def fetch_day(st: str, y: int, m: int, d: int, sleep: float = 1.0) -> pd.DataFrame:
    """1日分の風速・風向（生の文字列）。ページが出ない日は wind が 'NOPAGE'"""
    p, b = STATIONS[st]
    f = RAW / st / f"{y}{m:02d}{d:02d}.csv"
    if f.exists():
        return pd.read_csv(f, dtype=str, keep_default_na=False)
    f.parent.mkdir(parents=True, exist_ok=True)
    for k in range(3):
        try:
            r = requests.get(URL.format(p=p, b=b, y=y, m=m, d=d), timeout=30)
            r.raise_for_status()
            r.encoding = "utf-8"
            break
        except requests.RequestException:
            time.sleep(5 * (k + 1))
    else:
        raise RuntimeError(f"取得失敗: {st} {y}-{m}-{d}")
    try:
        t = pd.read_html(io.StringIO(r.text), flavor="lxml")[0]
        t.columns = [" ".join(dict.fromkeys(str(x) for x in c)) if isinstance(c, tuple) else str(c) for c in t.columns]
        cw = [c for c in t.columns if "風速" in c and not c.endswith("風向")][0]
        cd = [c for c in t.columns if c.endswith("風向")][0]
        out = pd.DataFrame({"hour": pd.to_numeric(t.iloc[:, 0], errors="coerce"), "wind": t[cw].astype(str), "wdir": t[cd].astype(str)})
        out = out.dropna(subset=["hour"]).astype({"hour": int}).astype(str)
    except (ValueError, IndexError):
        out = pd.DataFrame({"hour": [str(h) for h in range(1, 25)], "wind": ["NOPAGE"] * 24, "wdir": ["NOPAGE"] * 24})
    out.to_csv(f, index=False)
    time.sleep(sleep)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=list(range(2010, 2026)))
    ap.add_argument("--sleep", type=float, default=1.0)
    a = ap.parse_args()
    frames = []
    for st in STATIONS:
        for y in a.years:
            for m in (7, 8, 9):
                for d in range(1, calendar.monthrange(y, m)[1] + 1):
                    df = fetch_day(st, y, m, d, a.sleep)
                    df["station"] = st
                    df["date"] = f"{y}-{m:02d}-{d:02d}"
                    frames.append(df)
            print(st, y, "完了", flush=True)
    al = pd.concat(frames, ignore_index=True)
    al["hour"] = al["hour"].astype(int)
    al["wind_raw"] = al["wind"]
    al["wind"] = pd.to_numeric(al["wind"].str.replace(r"[)\]*#\s]", "", regex=True), errors="coerce")
    al.to_csv(BASE / "data" / "wind_hourly.csv", index=False)
    print("保存:", al.shape)


if __name__ == "__main__":
    main()
