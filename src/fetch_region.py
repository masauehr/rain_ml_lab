"""沖縄本島の周辺アメダス（降水量）の1時間値を取得する。正解（午後1〜6時に基準以上の雨があったか）を地域で作るために使う。

- 出典: 気象庁「過去の気象データ検索」 https://www.data.jma.go.jp/stats/etrn/
- 取得ページは hourly_a1（アメダス）。那覇・名護は fetch_hourly.py（s1）で取得済みなので、ここでは周辺9地点だけ。
- 1日1ファイル（24時間分の降水量の生の文字列）を data/raw_region/{地点}/{YYYYMMDD}.csv に保存する。取得済みは再取得しない。
- 使い方: .venv/bin/python src/fetch_region.py [--years 2010 ... 2025] [--sleep 1.0]
"""
import argparse
import calendar
import io
import time
from pathlib import Path

import pandas as pd
import requests

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw_region"

# 地点: (prec_no, block_no, 区分)。区分は緯度による暫定案（PLAN.md）
STATIONS = {
    "itokazu": ("91", "0909", "南部"), "ashimine": ("91", "1482", "南部"),
    "goya": ("91", "0907", "中部"), "miyagijima": ("91", "1596", "中部"), "yomitan": ("91", "0906", "中部"),
    "higashi": ("91", "1374", "北部"), "motobu": ("91", "1373", "北部"),
    "kunigami": ("91", "1586", "北部"), "oku": ("91", "0901", "北部"),
}
URL = ("https://www.data.jma.go.jp/stats/etrn/view/hourly_a1.php"
       "?prec_no={p}&block_no={b}&year={y}&month={m}&day={d}&view=")


def fetch_day(st: str, y: int, m: int, d: int, sleep: float = 1.0) -> pd.DataFrame:
    """1日分の降水量（生の文字列）を返す。列: hour, raw。ページが出ない日は raw が空（'NOPAGE'）。"""
    p, b, _ = STATIONS[st]
    f = RAW / st / f"{y}{m:02d}{d:02d}.csv"
    if f.exists():
        return pd.read_csv(f, dtype={"raw": str}, keep_default_na=False)
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
        col = [c for c in t.columns if "降水量" in c][0]
        out = pd.DataFrame({"hour": pd.to_numeric(t.iloc[:, 0], errors="coerce"), "raw": t[col].astype(str)})
        out = out.dropna(subset=["hour"]).astype({"hour": int})
    except (ValueError, IndexError):
        out = pd.DataFrame({"hour": list(range(1, 25)), "raw": ["NOPAGE"] * 24})
    out.to_csv(f, index=False)
    time.sleep(sleep)
    return out


def to_rain(raw: pd.Series) -> pd.Series:
    """「--」は降水なし（0mm）、「)」「]」は記号を外して数値、「///」「×」などの欠測は NaN。"""
    s = raw.astype(str).str.strip()
    num = pd.to_numeric(s.str.replace(r"[)\]*#]", "", regex=True), errors="coerce")
    return num.mask(s.eq("--"), 0.0)


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
                    df["time"] = pd.to_datetime(f"{y}-{m:02d}-{d:02d}") + pd.to_timedelta(df["hour"], unit="h")
                    df["station"] = st
                    frames.append(df)
            print(st, y, "完了", flush=True)
    al = pd.concat(frames, ignore_index=True)
    al["rain"] = to_rain(al["raw"])
    al.to_csv(BASE / "data" / "region_hourly.csv", index=False)
    print("保存:", al.shape)


if __name__ == "__main__":
    main()
