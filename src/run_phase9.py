"""Phase 9: 「午後1〜6時に1mm/h以上の雨が降るか」（雨予報）に向けた設定の調整。

- 正解: 共通地点ラベル y1_common（全期間で観測している地点だけ。common_labels.py）。
- **調整に使うのは 2010〜2024年の leave-one-year-out だけ。2025年・2001〜2009年は使わない。**
- 基準の設定から、条件を1つずつ変えて比べる（ExtraTrees・RandomForest・ロジスティック回帰）。評価は風速10m/s未満の標本。
- 指標: PR-AUC（順位）・Brier スキル（確率の良さ。基準は常に陽性率）・対数損失。シードごとに計算して、平均と標準偏差を出す。
- 選び方（結果を見る前に決めた規則）: 各軸で、2地点の PR-AUC 平均が最も高い値を選ぶ。ただし、基準の値との差がシード間の標準偏差以内なら、基準の値のままにする。
  選んだ組み合わせは outputs/final_1mm/config.json に保存する（Phase 10 の最終評価が読む）。
使い方: .venv/bin/python src/run_phase9.py [--seeds 3]
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import dataset as ds

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent.parent / "outputs"
YEARS = list(range(2010, 2025))
TARGET = "y1_common"
TAG = "1mm"          # 出力名の接尾辞（1mm は従来の名前のまま）
LABEL = "1mm/h"
NAME = {"naha": "那覇", "nago": "名護"}

BASE = dict(model="et", n_estimators=300, min_samples_leaf=5, max_features="sqrt", class_weight=None, change=False, train_all_wind=False, C=0.1)
# (軸, 値) を1つずつ変える
AXES = [
    ("class_weight", [None, "balanced_subsample"]),
    ("min_samples_leaf", [1, 5, 10, 20, 50, 100]),
    ("max_features", ["sqrt", 0.3, 0.5, 1.0]),
    ("n_estimators", [150, 300, 600]),
    ("change", [False, True]),
    ("train_all_wind", [False, True]),   # True = 学習に台風を含める（風速条件で除かない）
]


def make(cfg: dict, seed: int):
    if cfg["model"] == "logreg":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=cfg["C"], max_iter=3000))
    cls = ExtraTreesClassifier if cfg["model"] == "et" else RandomForestClassifier
    cw = cfg["class_weight"]
    if cw == "balanced_subsample" and cfg["model"] == "rf":
        cw = "balanced_subsample"
    return make_pipeline(SimpleImputer(strategy="median"),
                         cls(n_estimators=cfg["n_estimators"], min_samples_leaf=cfg["min_samples_leaf"], max_features=cfg["max_features"],
                             class_weight=cw, n_jobs=-1, random_state=seed))


def oof(x, feats, cfg, seed):
    pred = np.full(len(x), np.nan)
    ok = x.wind_ok.to_numpy() | cfg["train_all_wind"]
    for yr in YEARS:
        te = (x.year == yr).to_numpy()
        tr = (x.year.isin(YEARS) & (x.year != yr)).to_numpy() & ok
        pred[te] = make(cfg, seed).fit(x.loc[tr, feats], x.loc[tr, TARGET]).predict_proba(x.loc[te, feats])[:, 1]
    return pred


def metrics(y, p):
    pbar = y.mean()
    return {"PR-AUC": average_precision_score(y, p), "ROC-AUC": roc_auc_score(y, p),
            "Brierスキル": 1 - brier_score_loss(y, p) / (pbar * (1 - pbar)), "対数損失": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))}


def run(data, cfg, seeds):
    """data: {地点: (x, e, y)} → {地点: {指標: (平均, 標準偏差)}}"""
    res = {}
    for st, (x, e, y) in data.items():
        feats = ds.features(st, cfg["change"])
        ms = [metrics(y, oof(x, feats, cfg, s)[e]) for s in (seeds if cfg["model"] != "logreg" else seeds[:1])]
        res[st] = {k: (float(np.mean([m[k] for m in ms])), float(np.std([m[k] for m in ms]))) for k in ms[0]}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--th", type=int, default=1, help="基準（mm/h）: 1 または 20 など")
    ap.add_argument("--fix-cw-none", action="store_true", help="class_weight は なし に固定（確率の質を保つため。規則で選ばない）")
    a = ap.parse_args()
    global TARGET, TAG, LABEL
    TARGET, TAG, LABEL = f"y{a.th}_common", f"{a.th}mm", f"{a.th}mm/h"
    seeds = list(range(100, 100 + a.seeds))
    data = {}
    for st in ("naha", "nago"):
        x = ds.build(st)
        e = (x.year.isin(YEARS) & x.wind_ok).to_numpy()
        data[st] = (x, e, x.loc[e, TARGET].to_numpy())
    L = [f"# Phase 9: {LABEL}向けの設定の調整", "",
         "（`src/run_phase9.py` が自動生成。2010〜2024年の leave-one-year-out だけを使用。2025年・2001〜2009年は使っていない）", "",
         f"正解: 共通地点ラベル（午後1〜6時に{LABEL}以上）。評価は風速10m/s未満の標本。{len(seeds)}シードの平均±標準偏差。", ""]
    for st, (x, e, y) in data.items():
        L.append(f"- {NAME[st]}: 標本 {len(y)}、陽性 {int(y.sum())} 標本・{x.loc[e & (x[TARGET] == 1).to_numpy(), 'date'].nunique()} 日、陽性率 {y.mean():.3f}")
    L.append("")
    # 基準
    t0 = time.time()
    chosen = dict(BASE)
    summary = {}
    base_res = run(data, BASE, seeds)
    print("基準 完了", round(time.time() - t0), "秒", flush=True)
    for axis, values in AXES:
        L += [f"## 軸: {axis}", "", "| 値 | 那覇 PR-AUC | 名護 PR-AUC | 平均 PR-AUC | 那覇 Brierスキル | 名護 Brierスキル | 那覇 対数損失 | 名護 対数損失 |", "|---|---|---|---|---|---|---|---|"]
        rows = {}
        for v in values:
            cfg = {**BASE, axis: v}
            r = base_res if v == BASE[axis] else run(data, cfg, seeds)
            rows[v] = r
            m = np.mean([r[s]["PR-AUC"][0] for s in r])
            tag = "（基準）" if v == BASE[axis] else ""
            L.append(f"| {v}{tag} | {r['naha']['PR-AUC'][0]:.3f}±{r['naha']['PR-AUC'][1]:.3f} | {r['nago']['PR-AUC'][0]:.3f}±{r['nago']['PR-AUC'][1]:.3f} | {m:.3f} "
                     f"| {r['naha']['Brierスキル'][0]:+.3f} | {r['nago']['Brierスキル'][0]:+.3f} | {r['naha']['対数損失'][0]:.3f} | {r['nago']['対数損失'][0]:.3f} |")
        avg = {v: np.mean([rows[v][s]["PR-AUC"][0] for s in rows[v]]) for v in values}
        sd = np.mean([base_res[s]["PR-AUC"][1] for s in base_res])          # 基準のシード間の標準偏差
        best = max(values, key=lambda v: avg[v])
        pick = best if avg[best] - avg[BASE[axis]] > sd else BASE[axis]
        if axis == "class_weight" and a.fix_cw_none:
            L += ["", f"（規則では {pick} だが、確率の質を保つため class_weight なしに固定。1mm/h と同じ扱い）"]
            pick = None
        chosen[axis] = pick
        L += ["", f"→ 選択: **{pick}**（最高 {best} の平均 PR-AUC {avg[best]:.3f}、基準 {avg[BASE[axis]]:.3f}、基準のシード間標準偏差 {sd:.4f}）", ""]
        summary[axis] = {str(k): float(v) for k, v in avg.items()}
        print(axis, "→", pick, round(time.time() - t0), "秒", flush=True)

    # 他の手法（選んだ設定）
    L += ["## 選んだ設定での比較（ExtraTrees・RandomForest・ロジスティック回帰）", "", f"選んだ設定: {chosen}", "",
          "| モデル | 那覇 PR-AUC | 名護 PR-AUC | 那覇 Brierスキル | 名護 Brierスキル |", "|---|---|---|---|---|"]
    final = {}
    for name, cfg in (("ExtraTrees（選んだ設定）", chosen), ("RandomForest（同じ設定）", {**chosen, "model": "rf"}),
                      ("ロジスティック回帰 C=0.1", {**chosen, "model": "logreg", "C": 0.1}),
                      ("ロジスティック回帰 C=1", {**chosen, "model": "logreg", "C": 1.0}),
                      ("ロジスティック回帰 C=0.01", {**chosen, "model": "logreg", "C": 0.01})):
        r = run(data, cfg, seeds)
        final[name] = r
        L.append(f"| {name} | {r['naha']['PR-AUC'][0]:.3f} | {r['nago']['PR-AUC'][0]:.3f} | {r['naha']['Brierスキル'][0]:+.3f} | {r['nago']['Brierスキル'][0]:+.3f} |")
    L.append("")
    (OUT / f"final_{TAG}").mkdir(parents=True, exist_ok=True)
    (OUT / f"final_{TAG}" / "config.json").write_text(json.dumps(chosen, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / ("phase9.md" if TAG == "1mm" else f"phase9_{TAG}.md")).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
