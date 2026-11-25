"""Phase 2: ExtraTreesClassifier と RandomForestClassifier の比較（2010〜2024年 leave-one-year-out。2025年は使わない）。

- 主な設定: 地域の正解 × 20mm/h。参考に 30mm/h（地域）。評価は風速10m/s未満の標本、学習も同じ条件。
- 比較は「条件を1つだけ変える」: 基準の設定から、class_weight・min_samples_leaf・max_features・木の数・変化量の特徴を1つずつ変える。
- 各設定を 10 シード（既定）で回し、シードごとに「全15年の保持予測をまとめた PR-AUC」を出して、平均と標準偏差を見る。
  差が標準偏差より小さければ誤差とみなす。ロジスティック回帰（基準）も同じ方法で併記する。
使い方: .venv/bin/python src/run_phase2.py [--seeds 10] [--quick]
"""
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import dataset as ds

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent.parent / "outputs"
YEARS = list(range(2010, 2025))

BASE = dict(n_estimators=200, min_samples_leaf=5, max_features="sqrt", class_weight="balanced_subsample", change=True)
# (名前, 基準からの変更)。1つずつ変える
VARIANTS = [("基準の設定", {}), ("class_weight なし", {"class_weight": None}),
            ("min_samples_leaf=1", {"min_samples_leaf": 1}), ("min_samples_leaf=20", {"min_samples_leaf": 20}),
            ("max_features=0.5", {"max_features": 0.5}), ("木の数 50", {"n_estimators": 50}),
            ("木の数 600", {"n_estimators": 600}), ("変化量の特徴なし", {"change": False})]


def make_model(kind: str, cfg: dict, seed: int):
    if kind == "logreg":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000))
    cls = ExtraTreesClassifier if kind == "et" else RandomForestClassifier
    return make_pipeline(SimpleImputer(strategy="median"),
                         cls(n_estimators=cfg["n_estimators"], min_samples_leaf=cfg["min_samples_leaf"], max_features=cfg["max_features"],
                             class_weight=cfg["class_weight"], n_jobs=-1, random_state=seed))


def oof(x: pd.DataFrame, feats: list[str], target: str, kind: str, cfg: dict, seed: int) -> np.ndarray:
    """leave-one-year-out の保持予測（学習は風速条件で絞る）"""
    pred = np.full(len(x), np.nan)
    ok = x.wind_ok.to_numpy()
    for y in YEARS:
        te = (x.year == y).to_numpy()
        tr = (x.year.isin(YEARS) & (x.year != y)).to_numpy() & ok
        m = make_model(kind, cfg, seed).fit(x.loc[tr, feats], x.loc[tr, target])
        pred[te] = m.predict_proba(x.loc[te, feats])[:, 1]
    return pred


def evaluate(x: pd.DataFrame, pred: np.ndarray, target: str) -> tuple[float, float]:
    e = x.year.isin(YEARS).to_numpy() & x.wind_ok.to_numpy()
    y = x.loc[e, target].to_numpy()
    return average_precision_score(y, pred[e]), roc_auc_score(y, pred[e])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--quick", action="store_true", help="試運転（基準の設定だけ・2シード）")
    a = ap.parse_args()
    variants = VARIANTS[:1] if a.quick else VARIANTS
    seeds = list(range(100, 100 + (2 if a.quick else a.seeds)))  # 開発用シード
    rows = []
    for target, tlabel in (("y20_region", "地域の正解 × 20mm/h（主）"), ("y30_region", "地域の正解 × 30mm/h（参考）")):
        for st in ("naha", "nago"):
            x = ds.build(st)
            e = x.year.isin(YEARS).to_numpy() & x.wind_ok.to_numpy()
            npos = int(x.loc[e, target].sum())
            # ロジスティック回帰（基準）。乱数を使わないので1回
            f0 = ds.features(st, True)
            ap_l, roc_l = evaluate(x, oof(x, f0, target, "logreg", BASE, 0), target)
            rows.append(dict(target=tlabel, st=st, npos=npos, name="ロジスティック回帰（基準）", model="logreg", ap=ap_l, ap_sd=0.0, roc=roc_l))
            for name, chg in variants:
                cfg = {**BASE, **chg}
                feats = ds.features(st, cfg["change"])
                for kind, mname in (("et", "ExtraTrees"), ("rf", "RandomForest")):
                    t0 = time.time()
                    res = [evaluate(x, oof(x, feats, target, kind, cfg, s), target) for s in seeds]
                    aps, rocs = [r[0] for r in res], [r[1] for r in res]
                    rows.append(dict(target=tlabel, st=st, npos=npos, name=name, model=mname, ap=np.mean(aps), ap_sd=np.std(aps), roc=np.mean(rocs)))
                    print(f"{tlabel[:12]} {st} {name} {mname}: PR-AUC {np.mean(aps):.3f}±{np.std(aps):.3f}（{time.time() - t0:.0f}秒）", flush=True)
    r = pd.DataFrame(rows)
    L = ["# Phase 2: ExtraTrees と RandomForest の比較", "",
         f"（`src/run_phase2.py` が自動生成。2010〜2024年の leave-one-year-out、{len(seeds)}シードの平均±標準偏差。2025年は使っていない）", "",
         "評価は風速10m/s未満の標本。PR-AUC の基準（常に0）は陽性標本の割合。", ""]
    for tlabel, g in r.groupby("target", sort=False):
        for st, gg in g.groupby("st", sort=False):
            n = int(gg.npos.iloc[0])
            L += [f"## {tlabel} / {'那覇' if st == 'naha' else '名護'}（陽性 {n} 標本）", "", "| 設定 | モデル | PR-AUC | ROC-AUC |", "|---|---|---|---|"]
            for _, q in gg.iterrows():
                L.append(f"| {q['name']} | {q.model} | {q.ap:.3f} ± {q.ap_sd:.3f} | {q.roc:.3f} |")
            L.append("")
    OUT.mkdir(exist_ok=True)
    (OUT / ("phase2_quick.md" if a.quick else "phase2.md")).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
