"""Phase 5: 2025年での最終評価。**1回だけ**。実行すると outputs/final/TEST_USED.json が作られ、以後は拒否する。

- 学習: 2010〜2024年（風速10m/s未満の標本のみ）。設定は Phase 3 で固定したもの（ExtraTrees、n=300、min_samples_leaf=20、
  max_features='sqrt'、balanced_subsample、変化量なし、全特徴量の群あり）。結果を見て設定を変えない。
- 評価: 2025年。主は「風速10m/s未満の標本 × 地域の正解 × 20mm/h」。参考に 30mm/h・地点の正解・全標本も出す。
- 10シード（200〜209。開発用の100番台とは別）の予測を平均する。PR-AUC の95%区間は日単位のブートストラップ。
- 補足（結論の選択には使わない）: 「時刻・月を外した版」、「Platt 補正（2010〜2024年の保持予測で当てはめ）」。
使い方: .venv/bin/python src/run_phase5.py
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import dataset as ds
import run_phase2 as p2
import run_phase3 as p3

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent.parent
FIN = BASE / "outputs" / "final"
CFG = p3.FINAL
SEEDS = list(range(200, 210))
NAME = {"naha": "那覇", "nago": "名護"}
YEARS = p2.YEARS


def fit_predict(kind, x, feats, target, tr, te, seed):
    if kind == "logreg":
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000))
    else:
        cls = ExtraTreesClassifier if kind == "et" else RandomForestClassifier
        m = make_pipeline(SimpleImputer(strategy="median"),
                          cls(n_estimators=CFG["n_estimators"], min_samples_leaf=CFG["min_samples_leaf"], max_features=CFG["max_features"],
                              class_weight=CFG["class_weight"], n_jobs=-1, random_state=seed))
    return m.fit(x.loc[tr, feats], x.loc[tr, target]).predict_proba(x.loc[te, feats])[:, 1]


def main() -> None:
    if (FIN / "TEST_USED.json").exists():
        sys.exit("最終評価（2025年）は使用済みです。再評価は新しい検証データで行ってください。")
    L = ["# Phase 5: 2025年の最終評価（1回だけ）", "",
         "（`src/run_phase5.py` が自動生成。学習 2010〜2024年・風速10m/s未満の標本。設定は Phase 3 で固定したまま）", "",
         f"設定: {CFG}。シード {SEEDS[0]}〜{SEEDS[-1]} の平均。PR-AUC の区間は日単位ブートストラップ（2000回）の95%区間。", ""]
    rng = np.random.default_rng(0)
    allres, pooled = {}, {}
    for target, tlabel in (("y20_region", "地域の正解 × 20mm/h（主）"), ("y30_region", "地域の正解 × 30mm/h（参考）"), ("y20_local", "地点の正解 × 20mm/h（参考）")):
        L += [f"# {tlabel}", ""]
        pooled[target] = []
        for st in ("naha", "nago"):
            x = ds.build(st)
            ok = x.wind_ok.to_numpy()
            tr = (x.year.isin(YEARS)).to_numpy() & ok
            te = (x.year == 2025).to_numpy()
            f = ds.features(st, change=False)
            f_notm = [c for c in f if c not in ("t", "month")]
            xt = x[te].reset_index(drop=True)
            P = {"ExtraTrees": np.mean([fit_predict("et", x, f, target, tr, te, s) for s in SEEDS], axis=0),
                 "RandomForest": np.mean([fit_predict("rf", x, f, target, tr, te, s) for s in SEEDS], axis=0),
                 "ロジスティック回帰": fit_predict("logreg", x, f, target, tr, te, 0),
                 "直近6時間降水量": xt["rain6"].fillna(0).to_numpy(),
                 "（補足）ExtraTrees 時刻・月なし": np.mean([fit_predict("et", x, f_notm, target, tr, te, s) for s in SEEDS], axis=0)}
            for mask_name, em in (("風速10m/s未満の標本（主）", xt.wind_ok.to_numpy()), ("全標本（参考）", np.ones(len(xt), bool))):
                xe = xt[em].reset_index(drop=True)
                y = xe[target].to_numpy()
                Pe = {k: v[em] for k, v in P.items()}
                p3.TARGET = target
                b = p3.boot(xe, Pe, [("ExtraTrees", "RandomForest"), ("ExtraTrees", "ロジスティック回帰"), ("ExtraTrees", "直近6時間降水量")], rng)
                prev = y.mean()
                L += [f"## {NAME[st]} / {mask_name}（標本 {len(xe)}、陽性 {int(y.sum())} 標本・{xe.loc[y == 1, 'date'].nunique()} 日、陽性率 {prev:.3f}）", "",
                      "| モデル | PR-AUC | 95%区間 | 陽性率の何倍 | ROC-AUC |", "|---|---|---|---|---|"]
                for k in Pe:
                    ap = average_precision_score(y, Pe[k]) if y.sum() else float("nan")
                    lo, hi = (b["ap"][k][1], b["ap"][k][2]) if k in b["ap"] else (float("nan"),) * 2
                    L.append(f"| {k} | {ap:.3f} | {lo:.3f}〜{hi:.3f} | {ap / prev:.1f} | {roc_auc_score(y, Pe[k]):.3f} |")
                L += ["", "| 差（前者 − 後者） | 平均 | 95%区間 | 前者が上回った割合 |", "|---|---|---|---|"]
                for (a, c), (m, lo, hi, w) in b["diff"].items():
                    L.append(f"| {a} − {c} | {m:+.3f} | {lo:+.3f}〜{hi:+.3f} | {w:.0%} |")
                if "主" in mask_name and target == "y20_region":
                    p = Pe["ExtraTrees"]
                    L += ["", "警戒の出し方（ExtraTrees。予測確率の上位 q% を警戒）:", "", "| 警戒の割合 | 警戒標本数 | 適合率 | 再現率 | 陽性日のうち警戒できた日 |", "|---|---|---|---|---|"]
                    for q in (5, 10, 20, 30):
                        fl = p >= np.percentile(p, 100 - q)
                        hit = xe.loc[fl & (y == 1), "date"].nunique()
                        L.append(f"| {q}% | {int(fl.sum())} | {y[fl].mean():.3f} | {y[fl].sum() / max(y.sum(), 1):.2f} | {hit}/{xe.loc[y == 1, 'date'].nunique()} |")
                    # Platt 補正（補足）。2010〜2024年の保持予測で当てはめる
                    dev = x.year.isin(YEARS).to_numpy() & ok
                    po = np.mean([p2.oof(x, f, target, "et", {**p2.BASE, **CFG}, s) for s in SEEDS[:3]], axis=0)
                    zc = lambda v: np.log(np.clip(v, 1e-4, 1 - 1e-4) / (1 - np.clip(v, 1e-4, 1 - 1e-4))).reshape(-1, 1)
                    pl = LogisticRegression(C=1e6).fit(zc(po[dev]), x.loc[dev, target])
                    pc = pl.predict_proba(zc(p))[:, 1]
                    L += ["", f"（補足）Brier スコア: 補正なし {brier_score_loss(y, p):.4f}、Platt 補正あり {brier_score_loss(y, pc):.4f}、常に陽性率 {brier_score_loss(y, np.full(len(y), prev)):.4f}"]
                L.append("")
                if "主" in mask_name:
                    pooled[target].append((st, y, Pe))
        # 那覇と名護をまとめた値（20mm/h・地域の主）
        if pooled[target]:
            y = np.concatenate([a[1] for a in pooled[target]])
            L += [f"## 那覇と名護をまとめた値（風速10m/s未満の標本、陽性 {int(y.sum())} 標本）", "", "| モデル | PR-AUC | 陽性率の何倍 |", "|---|---|---|"]
            for k in pooled[target][0][2]:
                p = np.concatenate([a[2][k] for a in pooled[target]])
                L.append(f"| {k} | {average_precision_score(y, p):.3f} | {average_precision_score(y, p) / y.mean():.1f} |")
            L.append("")
    FIN.mkdir(parents=True, exist_ok=True)
    (BASE / "outputs" / "phase5.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (FIN / "TEST_USED.json").write_text(json.dumps({"used": True, "date": str(pd.Timestamp.today().date()), "config": CFG, "seeds": SEEDS}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
