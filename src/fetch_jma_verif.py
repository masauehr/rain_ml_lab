"""気象庁「天気予報検証結果」から、沖縄本島地方（本島中南部）の降水の有無の検証（A〜D）を取得する。

- 対象: 2006〜2025年の7〜9月（月別ページ f11_YYMM.html の「沖縄」＝沖縄本島地方本島中南部）。
- 5時発表予報の「今日(5-24時)」「明日(0-24時)」の分割表（%）を data/jma_verif/okinawa_monthly.csv に保存する。
  A=予報あり・実況あり、B=見逃し（予報なし・実況あり）、C=空振り（予報あり・実況なし）、D=予報なし・実況なし。全予報に対する%。
- 実況の降水あり=1mm以上（気象庁の定義）。
- 出典: 気象庁「天気予報検証結果」（https://www.data.jma.go.jp/yoho/kensho/HPdata/score_f.html）
使い方: .venv/bin/python src/fetch_jma_verif.py
"""
import html
import re
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://www.data.jma.go.jp/yoho/kensho/HPdata/"
OUT = Path(__file__).resolve().parent.parent / "data" / "jma_verif"


def parse(text: str) -> dict:
    t = html.unescape(re.sub(r"<[^>]+>", " ", text))
    t = re.sub(r"\s+", " ", t).replace("]", "").translate(str.maketrans("ＡＢＣＤ", "ABCD"))
    i = t.find("沖縄本島地方本島中南部")
    seg = t[i:i + 1500]
    r = {}
    for key, pat in (("A", r"A "), ("B", r"B（見逃し率） "), ("C", r"C（空振り率） "), ("D", r"D ")):
        m = re.search(pat + r"(\d+) (\d+) " + pat + r"(\d+) (\d+) (\d+)", seg)
        if not m:
            raise ValueError(f"{key} が見つからない")
        r[key + "_今日"], r[key + "_明日"] = int(m.group(1)), int(m.group(2))
    return r


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for y in range(2006, 2026):
        for m in (7, 8, 9):
            yymm = f"{y % 100:02d}{m:02d}"
            resp = requests.get(f"{BASE}HPdata{yymm}/f11_{yymm}.html", timeout=30)
            resp.encoding = resp.apparent_encoding
            try:
                d = parse(resp.text)
            except ValueError as e:
                print(yymm, "解析失敗", e)
                continue
            rows.append({"year": y, "month": m, **d})
            time.sleep(0.5)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "okinawa_monthly.csv", index=False)
    print(df.to_string())


if __name__ == "__main__":
    main()
