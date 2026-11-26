"""Phase 6b（追加実験）: 風の収束量を、三角形法・最小二乗法（沖縄本島の4〜6地点）で求めて説明変数に加えると効くか。

- 設定は Phase 3 で固定したもの。2010〜2024年の leave-one-year-out。**2025年は使わない**。
- 比較: 収束量なし／2地点法（Phase 6）／三角形法＋最小二乗法。同じシードで対にして、日単位ブートストラップ。
- 事前に data/wind_hourly.csv が必要（src/fetch_wind.py）。
使い方: .venv/bin/python src/run_phase6b.py
"""
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

import dataset as ds
import run_phase2 as p2
import run_phase3 as p3

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent.parent / "outputs"
YEARS = p2.YEARS
CFG = {**p2.BASE, **p3.FINAL}
SEEDS = list(range(100, 110))
NAME = {"naha": "那覇", "nago": "名護"}


def main() -> None:
    rng = np.random.default_rng(0)
    L = ["# Phase 6b（追加実験）: 三角形法・最小二乗法による風の収束量", "",
         "（`src/run_phase6b.py` が自動生成。2010〜2024年の leave-one-year-out。2025年は使っていない）", "",
         "- 三角形法: 3地点を通る平面で風を近似し、発散を求める（tri_n=奥・名護・宮城島、tri_c=名護・宮城島・那覇、tri_s=宮城島・糸数・那覇）。",
         "- 最小二乗法（lsq）: 6地点（奥・名護・宮城島・那覇・安次嶺・糸数）に平面を当てはめる。",
         "- 単位 1e-6 /s。正が収束。_mean は6時〜t の平均、conv_lsq_d1/d2 は1・2時間の変化。", ""]
    for target, tlabel in (("y20_region", "地域の正解 × 20mm/h（主）"), ("y30_region", "地域の正解 × 30mm/h（参考）")):
        p3.TARGET = target
        L += [f"# {tlabel}", ""]
        for st in ("naha", "nago"):
            x = ds.build(st)
            e = (x.year.isin(YEARS) & x.wind_ok).to_numpy()
            f0, f2, f3 = ds.features(st, False, False), ds.features(st, False, True), ds.features(st, False, False, tri=True)
            xe = x[e].reset_index(drop=True)
            y = xe[target].to_numpy()
            L += [f"## {NAME[st]}（標本 {len(xe)}、陽性 {int(y.sum())} 標本・{xe.loc[y == 1, 'date'].nunique()} 日）", "",
                  "### 収束量だけをスコアにしたとき", "", "| 特徴量 | PR-AUC | ROC-AUC | 欠測率 |", "|---|---|---|---|"]
            for c in ds.FEATURES_TRI:
                v = xe[c].fillna(xe[c].median()).to_numpy()
                L.append(f"| {c} | {average_precision_score(y, v):.3f} | {roc_auc_score(y, v):.3f} | {xe[c].isna().mean():.1%} |")
            L.append(f"\n（参考）陽性率 = {y.mean():.3f}。ROC-AUC 0.5 が無相関。")

            P = {}
            for tag, f in (("なし", f0), ("2地点法", f2), ("三角形・最小二乗", f3)):
                P[f"ExtraTrees（収束量{tag}）"] = np.mean([p2.oof(x, f, target, "et", CFG, s) for s in SEEDS], axis=0)[e]
            base = "ExtraTrees（収束量なし）"
            pairs = [(k, base) for k in P if k != base]
            b = p3.boot(xe, P, pairs, rng)
            L += ["", "### 比較（10シード平均の予測）", "", "| モデル | PR-AUC | 95%区間 | ROC-AUC |", "|---|---|---|---|"]
            for k in P:
                L.append(f"| {k} | {average_precision_score(y, P[k]):.3f} | {b['ap'][k][1]:.3f}〜{b['ap'][k][2]:.3f} | {roc_auc_score(y, P[k]):.3f} |")
            L += ["", "| 差（あり − なし） | 平均 | 95%区間 | 「あり」が上回った割合 |", "|---|---|---|---|"]
            for (a, c), (m, lo, hi, w) in b["diff"].items():
                L.append(f"| {a} | {m:+.3f} | {lo:+.3f}〜{hi:+.3f} | {w:.0%} |")
            L.append("")
            print(tlabel[:14], st, "完了", flush=True)
    (OUT / "phase6b.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
