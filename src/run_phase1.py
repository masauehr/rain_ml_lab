"""Phase 1: 基準づくり。2010〜2024年の leave-one-year-out（2025年は使わない）。

基準:
  A. 常に0と答える（PR-AUC は陽性率と同じ。これより悪ければ意味がない）
  B. 直近6時間の降水量（rain6）をそのままスコアにする
  C. ロジスティック回帰（標準化・欠測は中央値・class_weight=balanced）
評価: 保持した年の標本（学習側の風速条件で絞った標本）の予測を、全年分まとめて PR-AUC を出す。
    陽性は標本数（1日3標本）と日数を併記する。
使い方: .venv/bin/python src/run_phase1.py
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import dataset as ds

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent.parent / "outputs"
YEARS = list(range(2010, 2025))


def loyo(x: pd.DataFrame, feats: list[str], target: str, kind: str, wind_filter: bool) -> np.ndarray:
    """leave-one-year-out の予測確率（x の行順）。学習には風速条件を使い、保持した年は全標本を予測する"""
    pred = np.full(len(x), np.nan)
    for y in YEARS:
        te = (x.year == y).to_numpy()
        tr = (x.year.isin(YEARS) & (x.year != y)).to_numpy().copy()
        if wind_filter:
            tr = tr & x.wind_ok.to_numpy()
        if kind == "logreg":
            m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000))
            m.fit(x.loc[tr, feats], x.loc[tr, target])
            pred[te] = m.predict_proba(x.loc[te, feats])[:, 1]
        elif kind == "rain6":
            pred[te] = x.loc[te, "rain6"].fillna(0)
        elif kind == "zero":
            pred[te] = 0.0
    return pred


def score(x: pd.DataFrame, pred: np.ndarray, target: str, evalmask: np.ndarray) -> dict:
    e = (x.year.isin(YEARS)).to_numpy() & evalmask
    y = x.loc[e, target].to_numpy()
    p = pred[e]
    ap = average_precision_score(y, p) if p.std() > 0 else y.mean()
    return {"標本": int(e.sum()), "陽性標本": int(y.sum()), "陽性日": int(x.loc[e & (x[target] == 1), "date"].nunique()),
            "PR-AUC": ap, "ROC-AUC": roc_auc_score(y, p) if p.std() > 0 else 0.5}


def main() -> None:
    L = ["# Phase 1: 基準（leave-one-year-out、2010〜2024年）", "", "（`src/run_phase1.py` が自動生成。2025年は使っていない）", ""]
    for target, label in (("y20_region", "地域の正解 × 20mm/h（主）"), ("y30_region", "地域の正解 × 30mm/h（参考）"),
                          ("y20_local", "地点の正解 × 20mm/h（参考）")):
        L += [f"## {label}", "", "| 地点 | 評価の対象 | 手法 | 標本 | 陽性標本 | 陽性日 | PR-AUC | ROC-AUC |", "|---|---|---|---|---|---|---|---|"]
        for st in ("naha", "nago"):
            x = ds.build(st)
            tg = target.replace("region", "region").replace("local", "local")
            tg = f"y{target[1:3]}_{target.split('_')[1]}"
            for wf, wlabel in ((True, "風速10m/s未満の標本"), (False, "全標本")):
                mask = x.wind_ok.to_numpy() if wf else np.ones(len(x), bool)
                for name, kind, feats in (("常に0", "zero", []), ("直近6時間降水量", "rain6", []),
                                          ("ロジスティック回帰（基本）", "logreg", ds.features(st, change=False)),
                                          ("ロジスティック回帰（変化量あり）", "logreg", ds.features(st, change=True))):
                    p = loyo(x, feats, tg, kind, wind_filter=True)
                    r = score(x, p, tg, mask)
                    L.append(f"| {'那覇' if st == 'naha' else '名護'} | {wlabel} | {name} | {r['標本']} | {r['陽性標本']} | {r['陽性日']} | {r['PR-AUC']:.3f} | {r['ROC-AUC']:.3f} |")
        L.append("")
    OUT.mkdir(exist_ok=True)
    (OUT / "phase1.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
