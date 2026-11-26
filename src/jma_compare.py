"""気象庁の「降水の有無」の検証（沖縄本島中南部・5時発表）と、モデル（那覇）を、同じ指標で並べる。

- 気象庁: data/jma_verif/okinawa_monthly.csv（fetch_jma_verif.py）。7〜9月の月別%（A〜D）を月の重みを等しくして平均。
- モデル: 最終設定（outputs/final_1mm/config.json）で、2010〜2024年で学習し、2006〜2009年を予測（Phase 10 と同じ予測）。
  「降水あり」と判定するしきい値は2通り: 確率0.5以上／気象庁の「降水あり」予報の割合（A+C）に合わせる（正解は使わない）。
- 指標: 適中率・捕捉率・空振り率・見逃し率・一致率・Heidke スキル（偶然の的中を除いた進み。陽性率が違っても比べられる）。
- 注意: 気象庁は「沖縄本島中南部の5〜24時のいずれかで1mm以上」を予報者が判断、モデルは「午後1〜6時に共通地点のいずれかで1mm/h以上」を
  午前10・11・12時の観測から確率で予測。対象の時間帯・範囲・手法・年が違うので、優劣ではなく目安として並べる。
使い方: .venv/bin/python src/jma_compare.py
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import dataset as ds
from run_phase9 import make

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent.parent
TRAIN_YEARS = list(range(2010, 2025))
TEST_YEARS = [2006, 2007, 2008, 2009]
SEEDS = list(range(200, 210))


def idx(a, b, c, d):
    """A=的中の降水あり、B=見逃し、C=空振り、D=的中の降水なし（同じ単位）から指標を作る"""
    n = a + b + c + d
    a, b, c, d = a / n, b / n, c / n, d / n
    pe = (a + b) * (a + c) + (c + d) * (b + d)
    return {"実況の降水あり率": a + b, "「降水あり」予報の割合": a + c, "適中率": a + d, "捕捉率": a / (a + b), "一致率": a / (a + c) if a + c else np.nan,
            "空振り率": c, "見逃し率": b, "常に降水なしの適中率": 1 - (a + b), "Heidkeスキル": (a + d - pe) / (1 - pe)}


def main() -> None:
    cfg = json.loads((BASE / "outputs" / "final_1mm" / "config.json").read_text(encoding="utf-8"))
    j = pd.read_csv(BASE / "data" / "jma_verif" / "okinawa_monthly.csv")
    rows = {}
    for name, ys in (("気象庁 2006〜2009年", [2006, 2007, 2008, 2009]), ("気象庁 2016〜2025年", list(range(2016, 2026)))):
        m = j[j.year.isin(ys)][["A_今日", "B_今日", "C_今日", "D_今日"]].mean()
        rows[name] = idx(m["A_今日"], m["B_今日"], m["C_今日"], m["D_今日"])
    x = ds.build("naha", early=True)
    feats = ds.features("naha", cfg["change"])
    tr = (x.year.isin(TRAIN_YEARS) & x.wind_ok).to_numpy()
    te = (x.year.isin(TEST_YEARS) & x.wind_ok).to_numpy()
    y = x.loc[te, "y1_common"].to_numpy().astype(int)
    p = np.mean([make(cfg, s).fit(x.loc[tr, feats], x.loc[tr, "y1_common"]).predict_proba(x.loc[te, feats])[:, 1] for s in SEEDS], axis=0)
    jm = rows["気象庁 2006〜2009年"]["「降水あり」予報の割合"]
    thr = np.quantile(p, 1 - jm)
    for name, t in (("モデル 確率0.5以上", 0.5), (f"モデル 気象庁と同じ割合（しきい値{thr:.2f}）", thr)):
        f = p >= t
        rows[name] = idx((f & (y == 1)).sum(), ((~f) & (y == 1)).sum(), (f & (y == 0)).sum(), ((~f) & (y == 0)).sum())
    keys = list(next(iter(rows.values())).keys())
    L = ["# 気象庁の降水の有無の検証との比較（那覇・沖縄本島中南部）", "",
         "（`src/jma_compare.py` が自動生成。気象庁は5時発表の「今日(5-24時)」、7〜9月の月別集計の平均。出典: 気象庁「天気予報検証結果」）", "",
         "**注意**: 対象の時間帯（気象庁 5〜24時／モデル 午後1〜6時）・範囲（予報区全体／那覇周辺の共通地点）・手法・年が違う。優劣ではなく目安。"
         "実況の降水あり率が違うので、偶然の的中を除いた Heidke スキルで比べる。", "",
         "| 指標 | " + " | ".join(rows) + " |", "|---|" + "---|" * len(rows)]
    for k in keys:
        L.append(f"| {k} | " + " | ".join(f"{rows[r][k]:.1%}" if k != "Heidkeスキル" else f"{rows[r][k]:+.3f}" for r in rows) + " |")
    L += ["", "- Heidke スキル = (適中率 − 偶然の的中率) ÷ (1 − 偶然の的中率)。0 なら偶然と同じ、1 なら完璧。",
          "- 気象庁の月別値は、7〜9月の月ごとの%を等しい重みで平均したもの（月ごとの予報数は未公表）。"]
    (BASE / "outputs" / "jma_compare.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
