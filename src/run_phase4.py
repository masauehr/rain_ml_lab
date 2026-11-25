"""Phase 4: 特徴量重要度（impurity と permutation）、特徴量の群ごとの寄与、乱数のぶれ、校正の補正。

- 設定は Phase 3 で固定したもの（ExtraTrees、n=300、min_samples_leaf=20、max_features='sqrt'、balanced_subsample、変化量なし）。
- 2010〜2024年の leave-one-year-out。評価は風速10m/s未満の標本、地域の正解 × 20mm/h。2025年は使わない。
- permutation: 保持した年の特徴量（群は全列を同じ並べ替えで一緒に）を並べ替え、全年の保持予測をまとめた PR-AUC の低下量を見る。
- 群を抜いて学習し直した PR-AUC（leave-group-out）も出す。
使い方: .venv/bin/python src/run_phase4.py
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import make_pipeline

import dataset as ds
import run_phase3 as p3

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent.parent / "outputs"
YEARS = p3.YEARS
TARGET = p3.TARGET
CFG = p3.FINAL
SEEDS = [100, 101, 102]
REPS_G, REPS_F = 5, 3
NAME = {"naha": "那覇", "nago": "名護"}
GROUPS = {"気圧": ["pres", "pres_d1", "pres_d3"], "気温": ["temp"], "水蒸気（露点・湿度）": ["dew", "dewdep", "humid"],
          "風（風速・風向・午前の風）": ["wind", "wdir_sin", "wdir_cos", "wind_mean", "wind_max"],
          "直近の降水": ["rain1", "rain3", "rain6", "rainmax6"], "午前の日照・日射": ["sun_morning", "rad_morning"],
          "時刻・月": ["t", "month"]}


def model(seed):
    return make_pipeline(SimpleImputer(strategy="median"),
                         ExtraTreesClassifier(n_estimators=CFG["n_estimators"], min_samples_leaf=CFG["min_samples_leaf"], max_features=CFG["max_features"],
                                              class_weight=CFG["class_weight"], n_jobs=-1, random_state=seed))


def ap_of(x, pred, e):
    return average_precision_score(x.loc[e, TARGET], pred[e])


def importance(x, feats, e, ok):
    """impurity（全学習データで1回）と permutation（leave-one-year-out の保持予測）"""
    groups = {g: [c for c in cols if c in feats] for g, cols in GROUPS.items()}
    groups = {g: c for g, c in groups.items() if c}
    base = {s: np.full(len(x), np.nan) for s in SEEDS}
    perm = {}  # (seed, 名前, 反復) -> 予測
    rng = np.random.default_rng(1)
    names = [(g, cols, REPS_G) for g, cols in groups.items()] + [(f, [f], REPS_F) for f in feats]
    for y in YEARS:
        te = (x.year == y).to_numpy()
        tr = (x.year.isin(YEARS) & (x.year != y)).to_numpy() & ok
        Xte = x.loc[te, feats].reset_index(drop=True)
        for s in SEEDS:
            m = model(s).fit(x.loc[tr, feats], x.loc[tr, TARGET])
            base[s][te] = m.predict_proba(Xte)[:, 1]
            for nm, cols, reps in names:
                for r in range(reps):
                    Xp = Xte.copy()
                    Xp[cols] = Xte[cols].iloc[rng.permutation(len(Xte))].to_numpy()
                    perm.setdefault((s, nm, r), np.full(len(x), np.nan))[te] = m.predict_proba(Xp)[:, 1]
    ap_base = {s: ap_of(x, base[s], e) for s in SEEDS}
    rows = []
    for nm, cols, reps in names:
        drops = [np.mean([ap_base[s] - ap_of(x, perm[(s, nm, r)], e) for r in range(reps)]) for s in SEEDS]
        rows.append({"name": nm, "is_group": nm in groups, "drop": np.mean(drops), "sd": np.std(drops)})
    # impurity: 全学習データ（風速条件）で学習
    tr = x.year.isin(YEARS).to_numpy() & ok
    imp = np.mean([model(s).fit(x.loc[tr, feats], x.loc[tr, TARGET])[-1].feature_importances_ for s in SEEDS], axis=0)
    return pd.DataFrame(rows), pd.Series(imp, index=feats), float(np.mean(list(ap_base.values()))), float(np.std(list(ap_base.values()))), base


def loyo_ap(x, feats, e, ok):
    """群を抜いて学習し直したときの、シードごとの PR-AUC"""
    out = []
    for s in SEEDS:
        pred = np.full(len(x), np.nan)
        for y in YEARS:
            te = (x.year == y).to_numpy()
            tr = (x.year.isin(YEARS) & (x.year != y)).to_numpy() & ok
            pred[te] = model(s).fit(x.loc[tr, feats], x.loc[tr, TARGET]).predict_proba(x.loc[te, feats])[:, 1]
        out.append(ap_of(x, pred, e))
    return np.mean(out), np.std(out)


def calibrate(x, p, e, ok):
    """Platt 補正（予測確率のロジットに対するロジスティック回帰）を、保持した年を除いた他の年で当てはめる"""
    out = np.full(len(x), np.nan)
    z = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4))).reshape(-1, 1)
    for y in YEARS:
        te = (x.year == y).to_numpy()
        tr = (x.year.isin(YEARS) & (x.year != y)).to_numpy() & ok
        out[te] = LogisticRegression(C=1e6).fit(z[tr], x.loc[tr, TARGET]).predict_proba(z[te])[:, 1]
    return out


def main() -> None:
    L = ["# Phase 4: 特徴量重要度・群ごとの寄与・校正", "",
         "（`src/run_phase4.py` が自動生成。2010〜2024年の leave-one-year-out。2025年は使っていない）", "",
         f"設定: {CFG}。シード {SEEDS}、permutation の反復は群 {REPS_G} 回・個別 {REPS_F} 回。評価は風速10m/s未満の標本、地域の正解 × 20mm/h。", ""]
    for st in ("naha", "nago"):
        x = ds.build(st)
        feats = ds.features(st, change=False)
        ok = x.wind_ok.to_numpy()
        e = x.year.isin(YEARS).to_numpy() & ok
        print(st, "重要度…", flush=True)
        imp_df, imp, apb, apb_sd, base = importance(x, feats, e, ok)
        L += [f"## {NAME[st]}（基準の設定 PR-AUC {apb:.3f} ± {apb_sd:.3f}（シード間の標準偏差））", "",
              "### 群ごとの寄与（permutation で並べ替えたときの PR-AUC の低下量 / 群を抜いて学習し直したときの PR-AUC）", "",
              "| 群 | permutation の低下量 | 標準偏差 | 抜いて学習し直した PR-AUC | 抜いたことによる変化 |", "|---|---|---|---|---|"]
        gdf = imp_df[imp_df.is_group].sort_values("drop", ascending=False)
        for _, r in gdf.iterrows():
            cols = [c for c in GROUPS[r["name"]] if c in feats]
            m, sd = loyo_ap(x, [f for f in feats if f not in cols], e, ok)
            L.append(f"| {r['name']} | {r['drop']:+.4f} | {r['sd']:.4f} | {m:.3f} ± {sd:.3f} | {m - apb:+.3f} |")
        L += ["", "### 特徴量ごとの重要度（impurity と permutation の比較）", "",
              "| 特徴量 | impurity | permutation の低下量 | impurity の順位 | permutation の順位 |", "|---|---|---|---|---|"]
        fdf = imp_df[~imp_df.is_group].set_index("name")
        fdf["imp"] = imp
        fdf["r_imp"] = fdf["imp"].rank(ascending=False).astype(int)
        fdf["r_perm"] = fdf["drop"].rank(ascending=False).astype(int)
        for f, r in fdf.sort_values("drop", ascending=False).iterrows():
            L.append(f"| {f} | {r['imp']:.3f} | {r['drop']:+.4f} ± {r['sd']:.4f} | {r['r_imp']} | {r['r_perm']} |")
        rho = spearmanr(fdf["imp"], fdf["drop"]).statistic
        L += ["", f"impurity と permutation の順位相関（スピアマン）: {rho:.2f}", ""]

        # 校正
        p = np.mean([base[s] for s in SEEDS], axis=0)
        pc = calibrate(x, p, e, ok)
        y = x.loc[e, TARGET].to_numpy()
        prev = y.mean()
        L += ["### 校正（Platt 補正）", "",
              f"| | Brier スコア | PR-AUC |\n|---|---|---|\n| 補正なし | {brier_score_loss(y, p[e]):.4f} | {average_precision_score(y, p[e]):.3f} |\n"
              f"| Platt 補正あり | {brier_score_loss(y, pc[e]):.4f} | {average_precision_score(y, pc[e]):.3f} |\n"
              f"| 常に陽性率を返す | {brier_score_loss(y, np.full(len(y), prev)):.4f} | {prev:.3f} |", "",
              "| 区分（5分割） | 補正前の平均 | 補正後の平均 | 実際の陽性率 |", "|---|---|---|---|"]
        q = pd.qcut(pd.Series(p[e]).rank(method="first"), 5, labels=False)
        for k in range(5):
            m = (q == k).to_numpy()
            L.append(f"| {k + 1} | {p[e][m].mean():.3f} | {pc[e][m].mean():.3f} | {y[m].mean():.3f} |")
        L.append("")
    (OUT / "phase4.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
