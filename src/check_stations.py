"""降水量の観測地点の検証。地域の正解に使う地点が、本当に使えるかを少数の標本で確かめる。

確認すること:
  A. 緯度経度（気象庁の観測点一覧）から、南部・中部・北部の分類と、那覇・名護からの距離
  B. 2010〜2025年の標本日（3年おき×7・8・9月の15日）で、ページが出るか・24時間そろうか・数値に読めるか・欠測や注意記号の割合
  C. 同じ地点の別ページとの一致（那覇の s1 と a1）
  D. 標本日で、地点どうしの日降水量が妥当に似ているか（桁違いの地点がないか）
結果は outputs/station_check.md に保存する。標本日の取得は fetch_region.py のキャッシュに入る。
"""
import io
import sys
import time
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_region as fr  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "outputs" / "station_check.md"
CODES = {"itokazu": "91241", "ashimine": "91236", "goya": "91161", "miyagijima": "91166", "yomitan": "91121",
         "higashi": "91081", "motobu": "91066", "kunigami": "91046", "oku": "91021"}
NAMES = {"itokazu": "糸数", "ashimine": "安次嶺", "goya": "胡屋", "miyagijima": "宮城島", "yomitan": "読谷",
         "higashi": "東", "motobu": "本部", "kunigami": "国頭", "oku": "奥", "naha": "那覇", "nago": "名護"}
YEARS = [2010, 2013, 2016, 2019, 2022, 2025]
DAYS = [(7, 15), (8, 15), (9, 15)]


def km(a, b):
    """2点（緯度, 経度）の距離 km"""
    p1, p2 = radians(a[0]), radians(b[0])
    h = sin((p2 - p1) / 2) ** 2 + cos(p1) * cos(p2) * sin(radians(b[1] - a[1]) / 2) ** 2
    return 2 * 6371 * asin(sqrt(h))


def main() -> None:
    lines = ["# 降水量の観測地点の検証", "", "（`src/check_stations.py` が自動生成。標本日: 2010・2013・2016・2019・2022・2025年の 7/15・8/15・9/15）", ""]

    # A. 位置と区分
    tbl = requests.get("https://www.jma.go.jp/bosai/amedas/const/amedastable.json", timeout=30).json()
    pos = {}
    for st, code in {**CODES, "naha": "91197", "nago": "91107"}.items():
        v = tbl[code]
        pos[st] = (v["lat"][0] + v["lat"][1] / 60, v["lon"][0] + v["lon"][1] / 60)
    lines += ["## A. 位置と区分", "", "| 地点 | 区分（暫定） | 緯度 | 経度 | 那覇から km | 名護から km |", "|---|---|---|---|---|---|"]
    for st in list(CODES) + ["naha", "nago"]:
        kb = fr.STATIONS[st][2] if st in fr.STATIONS else ("南部" if st == "naha" else "北部")
        lines.append(f"| {NAMES[st]} | {kb} | {pos[st][0]:.3f} | {pos[st][1]:.3f} | {km(pos[st], pos['naha']):.0f} | {km(pos[st], pos['nago']):.0f} |")

    # B. 標本日の取得と品質
    rows, daily = [], {}
    for st in CODES:
        for y in YEARS:
            for m, d in DAYS:
                df = fr.fetch_day(st, y, m, d, 1.0)
                rain = fr.to_rain(df["raw"])
                toks = df["raw"].astype(str).str.strip()
                rows.append({"st": st, "y": y, "m": m, "nopage": bool((toks == "NOPAGE").all()), "n": len(df),
                             "nan": int(rain.isna().sum()), "flag": int(toks.str.contains(r"[)\]*#]").sum()),
                             "max": rain.max(), "sum": rain.sum()})
                daily[(st, y, m)] = rain.sum()
    r = pd.DataFrame(rows)
    g = r.groupby("st").agg(日数=("n", "size"), ページなし=("nopage", "sum"), 時間数が24でない日=("n", lambda s: int((s != 24).sum())),
                            欠測の時間=("nan", "sum"), 注意記号付きの時間=("flag", "sum"), 最大1時間雨量=("max", "max"))
    lines += ["", "## B. 標本日の取得と品質（各地点 18日）", "", "| 地点 | 日数 | ページなし | 24時間でない日 | 欠測の時間 | 注意記号の時間 | 最大1時間雨量(mm) |", "|---|---|---|---|---|---|---|"]
    for st in CODES:
        x = g.loc[st]
        lines.append(f"| {NAMES[st]} | {x['日数']} | {x['ページなし']} | {x['時間数が24でない日']} | {x['欠測の時間']} | {x['注意記号付きの時間']} | {x['最大1時間雨量']} |")
    bad = r[r["nopage"] | (r["n"] != 24)]
    lines += ["", "ページが出ない／24時間そろわない日: " + ("なし" if bad.empty else ", ".join(f"{NAMES[a]}{b}年{c}月" for a, b, c in zip(bad.st, bad.y, bad.m)))]

    # C. 官署（那覇・名護）は s1 でしか取れない（a1 は表示不可）。近くのアメダスと降水量が整合するかを見る
    lines += ["", "## C. 官署（s1）と近くのアメダス（a1）の整合", "",
              "那覇・名護は `hourly_a1` が表示できず `hourly_s1` だけで取得する。両者の降水量が同じ尺度（1時間値・mm）かを、近くのアメダスとの相関で確かめる。", "",
              "| 官署 | 比べた地点（距離） | 標本日の時間数 | 1時間値の相関 | 日降水量の相関 |", "|---|---|---|---|---|"]
    raw = Path(__file__).resolve().parent.parent / "data" / "raw"
    for kan, near in (("naha", "ashimine"), ("nago", "motobu")):
        x, y_ = [], []
        for y in YEARS:
            for m, d in DAYS:
                f = raw / kan / f"{y}{m:02d}{d:02d}.html"
                if not f.exists():
                    continue
                t2 = pd.read_html(io.StringIO(f.read_text(encoding="utf-8")), flavor="lxml")[0]
                t2.columns = [" ".join(dict.fromkeys(map(str, c))) if isinstance(c, tuple) else str(c) for c in t2.columns]
                kr = fr.to_rain(t2[[c for c in t2.columns if "降水量" in c][0]].astype(str)).reset_index(drop=True)
                nr = fr.to_rain(fr.fetch_day(near, y, m, d, 1.0)["raw"]).reset_index(drop=True)
                x.append(kr.rename("kan")); y_.append(nr.rename("near"))
        if not x:
            lines.append(f"| {NAMES[kan]} | {NAMES[near]} | 0 | （官署の取得が先） | - |")
            continue
        h = pd.concat([pd.concat(x, ignore_index=True), pd.concat(y_, ignore_index=True)], axis=1).dropna()
        dayc = h.groupby(h.index // 24).sum()
        lines.append(f"| {NAMES[kan]} | {NAMES[near]}（{km(pos[kan], pos[near]):.0f}km） | {len(h)} | {h.kan.corr(h.near):.2f} | {dayc.kan.corr(dayc.near):.2f} |")

    # D. 日降水量の妥当性
    dd = pd.Series(daily).unstack(level=[1, 2])
    lines += ["", "## D. 標本日の日降水量（mm）。同じ日に桁違いの地点がないか", "", "| 地点 | " + " | ".join(f"{y}/{m}" for y, m in dd.columns) + " |", "|---|" + "---|" * len(dd.columns)]
    for st in dd.index:
        lines.append(f"| {NAMES[st]} | " + " | ".join(f"{v:.0f}" for v in dd.loc[st]) + " |")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
