"""気象庁「過去の気象データ検索」から、那覇・名護の1時間値（7〜9月）を取得して CSV にする。

- 出典: https://www.data.jma.go.jp/stats/etrn/ （気象庁。利用規約に従い出典を明記する）
- サーバーに負荷をかけないよう、1リクエストごとに待つ。取得済みの日は再取得しない（data/raw/ にHTMLを保存）。
- 使い方: .venv/bin/python src/fetch_hourly.py [--years 2020 2021 ...] [--sleep 1.5]
"""
import argparse
import calendar
import io
import re
import time
from pathlib import Path

import pandas as pd
import requests

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw"
OUT = BASE / "data" / "hourly.csv"

# 那覇・名護はどちらも気象官署として「s1」（気圧・湿度なども載る表）で取得できる（2026-11-25 確認）
STATIONS = {"naha": ("91", "47936"), "nago": ("91", "47940")}
URL = ("https://www.data.jma.go.jp/stats/etrn/view/hourly_s1.php"
       "?prec_no={p}&block_no={b}&year={y}&month={m}&day={d}&view=")

# 1行目の見出しが日本語で揺れるので、位置ではなく見出し語から列を探す
COLS = {"pres": "現地", "pres_sea": "海面", "rain": "降水量", "temp": "気温", "dew": "露点",
        "vapor": "蒸気圧", "humid": "湿度", "wind": "風速", "wdir": "風向", "sun": "日照", "rad": "日射", "cloud": "雲量"}
# 注: 全天日射量は那覇のみ観測（名護は全期間が欠測）。名護では日照時間だけを使う


def fetch_day(st: str, y: int, m: int, d: int, sleep: float) -> str:
    """1日分のHTMLを返す（保存済みならそれを使う）。"""
    p, b = STATIONS[st]
    f = RAW / st / f"{y}{m:02d}{d:02d}.html"
    if f.exists():
        return f.read_text(encoding="utf-8")
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
    f.write_text(r.text, encoding="utf-8")
    time.sleep(sleep)
    return r.text


def parse_day(html: str, y: int, m: int, d: int) -> pd.DataFrame:
    """1日分の表を、英字の列名の DataFrame にする。観測なし（--）は欠測ではなく0mmとして扱う列がある点に注意（rain のみ）。"""
    tbl = pd.read_html(io.StringIO(html), flavor="lxml")[0]
    tbl.columns = [" ".join(dict.fromkeys(str(x) for x in c)) if isinstance(c, tuple) else str(c) for c in tbl.columns]
    out = pd.DataFrame({"hour": pd.to_numeric(tbl.iloc[:, 0], errors="coerce")})
    for name, key in COLS.items():
        # 風速・風向は上の見出しが同じ（風向・風速）なので、見出しの末尾の語で探す
        cand = [c for c in tbl.columns if c.endswith(key) or (name not in ("wind", "wdir") and key in c)]
        if cand:
            out[name] = tbl[cand[0]].to_numpy()
    out["time"] = pd.to_datetime(f"{y}-{m:02d}-{d:02d}") + pd.to_timedelta(out["hour"], unit="h")  # 1〜24時の「24時」は翌日0時
    return out.drop(columns="hour")


def to_num(s: pd.Series) -> pd.Series:
    """「--」（降水なし）は0、「///」「×」などの欠測はNaN、「)」「]」付きの値は記号を外して数値にする。"""
    s = s.astype(str).str.strip()
    flag = s.eq("--")
    num = pd.to_numeric(s.str.replace(r"[)\]*#]", "", regex=True), errors="coerce")
    return num.mask(flag, 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=list(range(2020, 2027)))
    ap.add_argument("--sleep", type=float, default=1.5)
    a = ap.parse_args()
    frames = []
    for st in STATIONS:
        for y in a.years:
            for m in (7, 8, 9):
                for d in range(1, calendar.monthrange(y, m)[1] + 1):
                    df = parse_day(fetch_day(st, y, m, d, a.sleep), y, m, d)
                    df["station"] = st
                    frames.append(df)
            print(st, y, "完了", flush=True)
    al = pd.concat(frames, ignore_index=True)
    for c in [c for c in al.columns if c not in ("time", "station", "wdir")]:
        al[c] = to_num(al[c])
    al.to_csv(OUT, index=False)
    print("保存:", OUT, al.shape)


if __name__ == "__main__":
    main()
