"""Phase 10: 1mm/h（雨予報）の最終評価。検証期間 2006〜2009年を **1回だけ** 評価する。

- 設定は outputs/final_1mm/config.json（Phase 9 で2010〜2024年だけを使って決定）。結果を見て設定を変えない。
- 学習: 2010〜2024年（風速10m/s未満）。評価: 2006〜2009年（風速10m/s未満）。正解: 共通地点ラベル y1_common。
- 比較: RandomForest（同設定）・ロジスティック回帰・直近6時間降水量。雨量の気候値は基準に使わない。
- 95%区間は日単位ブートストラップ（2000回）。
- 再実行防止: outputs/final_1mm/TEST_USED.json があれば実行しない。
使い方: .venv/bin/python src/run_phase10.py
"""
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

import dataset as ds
from run_phase9 import make, NAME

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "outputs"
TRAIN_YEARS = list(range(2010, 2025))
TEST_YEARS = [2006, 2007, 2008, 2009]
TARGET = "y1_common"
SEEDS = list(range(200, 210))
NB = 2000


def bss(y, p):
    pbar = y.mean()
    return 1.0 - brier_score_loss(y, p) / (pbar * (1 - pbar))


def aps(a, ref):
    return (a - ref) / (1.0 - ref)


def main() -> None:
    guard = OUT / "final_1mm" / "TEST_USED.json"
    if guard.exists():
        sys.exit("最終評価は実行済みです（TEST_USED.json）。再実行しません。")
    cfg = json.loads((OUT / "final_1mm" / "config.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(0)
    L = ["# Phase 10: 1mm/h（雨予報）の最終評価（2006〜2009年）", "",
         "（`src/run_phase10.py` が自動生成。1回だけ実行。設定は outputs/final_1mm/config.json）", "",
         f"設定: {cfg}", "",
         "学習 2010〜2024年（風速10m/s未満）、評価 2006〜2009年（風速10m/s未満）、正解は共通地点ラベル（午後1〜6時に1mm/h以上）。",
         f"ExtraTrees・RandomForest は{len(SEEDS)}シードの予測平均。95%区間は日単位ブートストラップ（{NB}回）。雨量の気候値は基準に使っていない。", ""]
    meta = {}
    for st in ("naha", "nago"):
        x = ds.build(st, early=True)
        feats = ds.features(st, cfg["change"])
        tr = (x.year.isin(TRAIN_YEARS) & x.wind_ok).to_numpy()
        te = (x.year.isin(TEST_YEARS) & x.wind_ok & x[TARGET].notna()).to_numpy()
        xt = x[te].reset_index(drop=True)
        y = xt[TARGET].to_numpy().astype(int)
        P = {}
        for name, c in (("ExtraTrees", cfg), ("RandomForest", {**cfg, "model": "rf"})):
            P[name] = np.mean([make(c, s).fit(x.loc[tr, feats], x.loc[tr, TARGET]).predict_proba(x.loc[te, feats])[:, 1] for s in SEEDS], axis=0)
        P["ロジスティック回帰"] = make({**cfg, "model": "logreg"}, 0).fit(x.loc[tr, feats], x.loc[tr, TARGET]).predict_proba(x.loc[te, feats])[:, 1]
        P["直近6時間降水量"] = xt["rain6"].fillna(0).to_numpy()
        dates = xt["date"].to_numpy()
        uniq = np.unique(dates)
        idx = {d: np.where(dates == d)[0] for d in uniq}

        def scores(yy, pp):
            ap = {k: average_precision_score(yy, v) for k, v in pp.items()}
            prev = yy.mean()
            s = {"PR-AUC ExtraTrees": ap["ExtraTrees"], "PR-AUC RandomForest": ap["RandomForest"],
                 "PR-AUC ロジスティック回帰": ap["ロジスティック回帰"], "PR-AUC 直近6時間降水量": ap["直近6時間降水量"],
                 "PR-AUC 陽性率（何も知らない予測）": prev,
                 "スキル ExtraTrees／基準: 陽性率": aps(ap["ExtraTrees"], prev),
                 "スキル ExtraTrees／基準: 直近6時間降水量": aps(ap["ExtraTrees"], ap["直近6時間降水量"]),
                 "スキル ExtraTrees／基準: ロジスティック回帰": aps(ap["ExtraTrees"], ap["ロジスティック回帰"]),
                 "スキル ExtraTrees／基準: RandomForest": aps(ap["ExtraTrees"], ap["RandomForest"]),
                 "Brierスキル ExtraTrees": bss(yy, pp["ExtraTrees"]),
                 "Brierスキル RandomForest": bss(yy, pp["RandomForest"]),
                 "Brierスキル ロジスティック回帰": bss(yy, pp["ロジスティック回帰"])}
            return s

        point = scores(y, P)
        res = {k: [] for k in point}
        for _ in range(NB):
            pick = np.concatenate([idx[d] for d in rng.choice(uniq, len(uniq))])
            if y[pick].sum() == 0:
                continue
            for k, v in scores(y[pick], {kk: vv[pick] for kk, vv in P.items()}).items():
                res[k].append(v)
        n_pos_days = xt.loc[y == 1, "date"].nunique()
        L += [f"## {NAME[st]}（標本 {len(xt)}、陽性 {int(y.sum())} 標本・{n_pos_days} 日、陽性率 {y.mean():.3f}）", "",
              "| 指標 | 値 | 95%区間 | 0を上回った割合 |", "|---|---|---|---|"]
        for k, v in res.items():
            v = np.array(v)
            lo, hi = np.percentile(v, [2.5, 97.5])
            L.append(f"| {k} | {point[k]:+.3f} | {lo:+.3f}〜{hi:+.3f} | {np.mean(v > 0):.0%} |")
        roc = {k: roc_auc_score(y, v) for k, v in P.items()}
        L += ["", "ROC-AUC: " + "、".join(f"{k} {v:.3f}" for k, v in roc.items()), ""]
        meta[st] = {"samples": len(xt), "pos_samples": int(y.sum()), "pos_days": int(n_pos_days)}
        print(NAME[st], "完了", flush=True)
    L.append("注: スキル（PR-AUC）は 0 なら基準と同じ、1 なら完璧。Brier スキルの基準は常に陽性率を返す予測。"
             "区間が0を含めば、基準より進んだとは言えない。")
    (OUT / "phase10.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    guard.write_text(json.dumps({"実行日時": datetime.now().isoformat(timespec="seconds"), "評価期間": TEST_YEARS,
                                 "設定": cfg, "件数": meta}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
