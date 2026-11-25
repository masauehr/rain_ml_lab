"""Phase 3: 差が偶然かの確認（日単位ブートストラップ）、確率の使い方（しきい値・再現率・適合率）、校正。

- 設定の固定（Phase 2 の傾向から。2025年は見ていない）: n_estimators=300, min_samples_leaf=20, max_features='sqrt',
  class_weight='balanced_subsample', 変化量の特徴は入れない。ロジスティック回帰も同じ特徴を使う。
- 予測は 10 シードの平均。2010〜2024年の leave-one-year-out（学習は風速10m/s未満の標本のみ。評価も同じ標本）。
- 主な設定: 地域の正解 × 20mm/h。
使い方: .venv/bin/python src/run_phase3.py
"""
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

import dataset as ds
import run_phase2 as p2

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent.parent / "outputs"
YEARS = p2.YEARS
FINAL = dict(n_estimators=300, min_samples_leaf=20, max_features="sqrt", class_weight="balanced_subsample", change=False)
SEEDS = list(range(100, 110))
TARGET = "y20_region"
NB = 2000
NAME = {"naha": "那覇", "nago": "名護"}


def avg_oof(x, feats, kind, cfg):
    if kind == "logreg":
        return p2.oof(x, feats, TARGET, "logreg", cfg, 0)
    return np.mean([p2.oof(x, feats, TARGET, kind, cfg, s) for s in SEEDS], axis=0)


def boot(x, preds: dict, pairs: list[tuple[str, str]], rng) -> dict:
    """日単位のブートストラップ。同じ日の3標本は一緒に入れ替える。PR-AUC とその差の 95% 区間"""
    dates = x["date"].to_numpy()
    uniq = np.unique(dates)
    idx = {d: np.where(dates == d)[0] for d in uniq}
    y = x[TARGET].to_numpy()
    res = {k: [] for k in preds}
    diff = {p: [] for p in pairs}
    for _ in range(NB):
        pick = np.concatenate([idx[d] for d in rng.choice(uniq, len(uniq))])
        if y[pick].sum() == 0:
            continue
        ap = {k: average_precision_score(y[pick], v[pick]) for k, v in preds.items()}
        for k, v in ap.items():
            res[k].append(v)
        for a, b in pairs:
            diff[(a, b)].append(ap[a] - ap[b])
    ci = lambda v: (np.mean(v), *np.percentile(v, [2.5, 97.5]))
    return {"ap": {k: ci(v) for k, v in res.items()}, "diff": {p: (*ci(v), float(np.mean(np.array(v) > 0))) for p, v in diff.items()}}


def main() -> None:
    rng = np.random.default_rng(0)
    L = ["# Phase 3: 差の確認・しきい値・校正", "",
         "（`src/run_phase3.py` が自動生成。2010〜2024年の leave-one-year-out、10シード平均の予測。2025年は使っていない）", "",
         f"固定した設定: {FINAL}", "", "評価は風速10m/s未満の標本、地域の正解 × 20mm/h。日単位のブートストラップ（2000回）の95%区間。", ""]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, st in zip(axes, ("naha", "nago")):
        x = ds.build(st)
        e = (x.year.isin(YEARS) & x.wind_ok).to_numpy()
        f_no, f_chg = ds.features(st, False), ds.features(st, True)
        P = {"ロジスティック回帰": avg_oof(x, f_no, "logreg", FINAL), "ExtraTrees": avg_oof(x, f_no, "et", FINAL),
             "RandomForest": avg_oof(x, f_no, "rf", FINAL), "ExtraTrees（変化量あり）": avg_oof(x, f_chg, "et", {**FINAL, "change": True}),
             "直近6時間降水量": x["rain6"].fillna(0).to_numpy()}
        xe = x[e].reset_index(drop=True)
        Pe = {k: v[e] for k, v in P.items()}
        y = xe[TARGET].to_numpy()
        prev = y.mean()
        pairs = [("ExtraTrees", "RandomForest"), ("ExtraTrees", "ロジスティック回帰"), ("ExtraTrees", "直近6時間降水量"),
                 ("ExtraTrees（変化量あり）", "ExtraTrees")]
        b = boot(xe, Pe, pairs, rng)
        L += [f"## {NAME[st]}（標本 {len(xe)}、陽性 {int(y.sum())} 標本・{xe.loc[y == 1, 'date'].nunique()} 日、陽性率 {prev:.3f}）", "",
              "### PR-AUC（95%区間）", "", "| モデル | PR-AUC | 95%区間 | 陽性率の何倍か |", "|---|---|---|---|"]
        for k, (m, lo, hi) in b["ap"].items():
            L.append(f"| {k} | {average_precision_score(y, Pe[k]):.3f} | {lo:.3f}〜{hi:.3f} | {average_precision_score(y, Pe[k]) / prev:.1f} |")
        L += ["", "### 差（前者 − 後者）", "", "| 比較 | 差の平均 | 95%区間 | 前者が上回った割合 |", "|---|---|---|---|"]
        for (a, c), (m, lo, hi, w) in b["diff"].items():
            L.append(f"| {a} − {c} | {m:+.3f} | {lo:+.3f}〜{hi:+.3f} | {w:.0%} |")

        # しきい値: 上位 q% を「警戒」とする。再現率・適合率・陽性日の検出
        p = Pe["ExtraTrees"]
        L += ["", "### 警戒の出し方（ExtraTrees。予測確率の上位 q% を「警戒」とする）", "",
              "| 警戒の割合 | しきい値 | 警戒標本数 | 適合率 | 再現率 | 陽性日のうち警戒できた日 | 警戒した日（のべ） |", "|---|---|---|---|---|---|---|"]
        pos_days = set(xe.loc[y == 1, "date"])
        for q in (2, 5, 10, 20, 30):
            thr = np.percentile(p, 100 - q)
            fl = p >= thr
            hit_days = set(xe.loc[fl & (y == 1), "date"])
            L.append(f"| {q}% | {thr:.3f} | {int(fl.sum())} | {y[fl].mean():.3f} | {y[fl].sum() / y.sum():.2f} | {len(hit_days)}/{len(pos_days)} | {xe.loc[fl, 'date'].nunique()} |")

        # 校正: 予測の順位で5分割し、平均予測確率と実際の陽性率を比べる
        q = pd.qcut(pd.Series(p).rank(method="first"), 5, labels=False)
        L += ["", "### 校正（予測確率を5分割。class_weight で確率の大きさがずれる）", "", "| 区分 | 標本数 | 平均の予測確率 | 実際の陽性率 |", "|---|---|---|---|"]
        for k in range(5):
            m = (q == k).to_numpy()
            L.append(f"| {k + 1}（{'低' if k == 0 else '高' if k == 4 else ''}） | {m.sum()} | {p[m].mean():.3f} | {y[m].mean():.3f} |")
        L.append("")

        # 図: PR 曲線
        for k, c, en in (("ExtraTrees", "C0", "ExtraTrees"), ("ロジスティック回帰", "C1", "LogisticReg"), ("直近6時間降水量", "C2", "rain6h")):
            pr, rc, _ = precision_recall_curve(y, Pe[k])
            ax.plot(rc, pr, c, label=en)
        ax.axhline(prev, color="gray", ls="--", label="base rate")
        ax.set_title(f"{st}: PR curve (LOYO 2010-2024)")
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(OUT / "phase3_pr.png", dpi=110)
    (OUT / "phase3.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
