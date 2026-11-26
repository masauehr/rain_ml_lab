"""Phase 6（追加実験）: 風の収束量（那覇・名護の2地点の線方向）を説明変数に加えると効くか。

- 設定は Phase 3 で固定したもの（ExtraTrees など）。2010〜2024年の leave-one-year-out。**2025年は使わない**（最終評価は使用済み）。
- 主な設定: 地域の正解 × 20mm/h、評価は風速10m/s未満の標本。比較は「収束量あり／なし」を、同じシードで対にして。
- 収束量そのものの強さも見る（収束量だけをスコアにしたときの PR-AUC・ROC-AUC。符号は収束＝正）。
使い方: .venv/bin/python src/run_phase6.py
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
    L = ["# Phase 6（追加実験）: 風の収束量", "",
         "（`src/run_phase6.py` が自動生成。2010〜2024年の leave-one-year-out。2025年は使っていない）", "",
         "収束量 = 那覇（南端）と名護（北端）の風の、線方向（那覇→名護）の成分の差から求めた収束（正が収束）。単位 1e-6 /s。"
         "線の長さ約51km。風向は16方位・前10分間の平均。特徴量: conv（t時）・conv_d1（1時間の変化）・conv_d2（2時間の変化）・conv_mean（6時〜t の平均）・cross（線に直交する成分の差）・spd_diff（風速差）。", ""]
    for target, tlabel in (("y20_region", "地域の正解 × 20mm/h（主）"), ("y30_region", "地域の正解 × 30mm/h（参考）")):
        p3.TARGET = target
        L += [f"# {tlabel}", ""]
        for st in ("naha", "nago"):
            x = ds.build(st)
            e = (x.year.isin(YEARS) & x.wind_ok).to_numpy()
            f0, f1 = ds.features(st, False, False), ds.features(st, False, True)
            xe = x[e].reset_index(drop=True)
            y = xe[target].to_numpy()
            L += [f"## {NAME[st]}（標本 {len(xe)}、陽性 {int(y.sum())} 標本・{xe.loc[y == 1, 'date'].nunique()} 日、陽性率 {y.mean():.3f}）", ""]

            # 収束量そのものの強さ
            L += ["### 収束量だけをスコアにしたとき", "", "| 特徴量 | PR-AUC | ROC-AUC |", "|---|---|---|"]
            for c in ds.FEATURES_CONV:
                v = xe[c].fillna(xe[c].median()).to_numpy()
                L.append(f"| {c} | {average_precision_score(y, v):.3f} | {roc_auc_score(y, v):.3f} |")
            L.append(f"\n（参考）陽性率 = {y.mean():.3f}、ROC-AUC 0.5 が無相関。ROC-AUC が 0.5 を下回る特徴量は、符号が逆（発散のとき降りやすい）。")

            # あり／なしの比較（ExtraTrees・ロジスティック回帰）
            P = {}
            for kind, mn in (("et", "ExtraTrees"), ("logreg", "ロジスティック回帰")):
                for tag, f in (("なし", f0), ("あり", f1)):
                    if kind == "logreg":
                        P[f"{mn}（収束量{tag}）"] = p2.oof(x, f, target, "logreg", CFG, 0)[e]
                    else:
                        P[f"{mn}（収束量{tag}）"] = np.mean([p2.oof(x, f, target, "et", CFG, s) for s in SEEDS], axis=0)[e]
            pairs = [("ExtraTrees（収束量あり）", "ExtraTrees（収束量なし）"), ("ロジスティック回帰（収束量あり）", "ロジスティック回帰（収束量なし）")]
            b = p3.boot(xe, P, pairs, rng)
            L += ["", "### 収束量の追加（10シード平均の予測）", "", "| モデル | PR-AUC | 95%区間 | ROC-AUC |", "|---|---|---|---|"]
            for k in P:
                L.append(f"| {k} | {average_precision_score(y, P[k]):.3f} | {b['ap'][k][1]:.3f}〜{b['ap'][k][2]:.3f} | {roc_auc_score(y, P[k]):.3f} |")
            L += ["", "| 差（あり − なし） | 平均 | 95%区間 | 「あり」が上回った割合 |", "|---|---|---|---|"]
            for (a, c), (m, lo, hi, w) in b["diff"].items():
                L.append(f"| {a.split('（')[0]} | {m:+.3f} | {lo:+.3f}〜{hi:+.3f} | {w:.0%} |")
            L.append("")
            print(tlabel[:14], st, "完了", flush=True)
    (OUT / "phase6.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
