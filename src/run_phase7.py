"""Phase 7（追加）: スキルスコア（基準からどれだけ進んだか）と、その95%区間（日単位ブートストラップ）。

- スキルスコア = (モデルの点数 − 基準の点数) ÷ (完璧な点数 − 基準の点数)。0 なら基準と同じ、1 なら完璧、負なら基準より悪い。
- PR-AUC のスキル: 基準は「陽性率（何も知らない予測）」「直近6時間降水量」「ロジスティック回帰」。完璧は 1。
- Brier スキル(BSS) = 1 − Brier(モデル) ÷ Brier(常に陽性率を返す)。確率の値が必要なので、class_weight なしの ExtraTrees を使う。
  （class_weight ありは確率が膨らむので Brier には使えない）
- 設定は Phase 3 で固定。2010〜2024年の leave-one-year-out、10シード平均、風速10m/s未満、地域の正解 × 20mm/h。2025年は使わない。
- 雨量の気候値（月別・時刻別の陽性率など）は基準に使わない。
使い方: .venv/bin/python src/run_phase7.py
"""
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss

import dataset as ds
import run_phase3 as p3

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent.parent / "outputs"
YEARS = p3.YEARS
NAME = p3.NAME
NB = 2000


def ap_skill(ap_m: float, ap_ref: float) -> float:
    return (ap_m - ap_ref) / (1.0 - ap_ref)


def bss(y: np.ndarray, p: np.ndarray) -> float:
    pbar = y.mean()
    return 1.0 - brier_score_loss(y, p) / (pbar * (1 - pbar))


def main(target: str = "y20_region", outname: str = "phase7.md", title: str = "Phase 7（追加）: スキルスコア", cond: str = "地域の正解 × 20mm/h") -> None:
    TARGET = target
    p3.TARGET = target
    rng = np.random.default_rng(0)
    L = [f"# {title}", "",
         "（`src/run_phase7.py` が自動生成。2010〜2024年の leave-one-year-out、10シード平均の予測。2025年は使っていない）", "",
         "スキルスコア = (モデル − 基準) ÷ (完璧 − 基準)。0 なら基準と同じ、1 なら完璧、負なら基準より悪い。",
         f"{cond}、風速10m/s未満の標本。95%区間は日単位ブートストラップ（2000回）。雨量の気候値は基準に使っていない。", ""]
    for st in ("naha", "nago"):
        x = ds.build(st)
        e = (x.year.isin(YEARS) & x.wind_ok).to_numpy()
        f = ds.features(st, False)
        P = {"ExtraTrees": p3.avg_oof(x, f, "et", p3.FINAL),
             "ExtraTrees（class_weightなし）": p3.avg_oof(x, f, "et", {**p3.FINAL, "class_weight": None}),
             "ロジスティック回帰": p3.avg_oof(x, f, "logreg", p3.FINAL),
             "直近6時間降水量": x["rain6"].fillna(0).to_numpy()}
        xe = x[e].reset_index(drop=True)
        Pe = {k: v[e] for k, v in P.items()}
        y = xe[TARGET].to_numpy()
        dates = xe["date"].to_numpy()
        uniq = np.unique(dates)
        idx = {d: np.where(dates == d)[0] for d in uniq}

        def scores(yy, pp):
            """1組のデータ（元データまたは引き直し）で、全スキルスコアを計算"""
            ap = {k: average_precision_score(yy, v) for k, v in pp.items()}
            prev = yy.mean()
            s = {}
            for m in ("ExtraTrees", "ロジスティック回帰"):
                s[f"PR-AUC {m} ／ 基準: 陽性率"] = ap_skill(ap[m], prev)
                s[f"PR-AUC {m} ／ 基準: 直近6時間降水量"] = ap_skill(ap[m], ap["直近6時間降水量"])
            s["PR-AUC ExtraTrees ／ 基準: ロジスティック回帰"] = ap_skill(ap["ExtraTrees"], ap["ロジスティック回帰"])
            s["Brier ExtraTrees（class_weightなし） ／ 基準: 陽性率"] = bss(yy, pp["ExtraTrees（class_weightなし）"])
            return s

        point = scores(y, Pe)
        res = {k: [] for k in point}
        for _ in range(NB):
            pick = np.concatenate([idx[d] for d in rng.choice(uniq, len(uniq))])
            if y[pick].sum() == 0:
                continue
            for k, v in scores(y[pick], {kk: vv[pick] for kk, vv in Pe.items()}).items():
                res[k].append(v)
        L += [f"## {NAME[st]}（標本 {len(xe)}、陽性 {int(y.sum())} 標本・{xe.loc[y == 1, 'date'].nunique()} 日、陽性率 {y.mean():.3f}）", "",
              "| スキルスコア | 値 | 95%区間 | 0を上回った割合 |", "|---|---|---|---|"]
        for k, v in res.items():
            v = np.array(v)
            lo, hi = np.percentile(v, [2.5, 97.5])
            L.append(f"| {k} | {point[k]:+.3f} | {lo:+.3f}〜{hi:+.3f} | {np.mean(v > 0):.0%} |")
        L.append("")
        print(NAME[st], "完了", flush=True)
    L += ["注: PR-AUC のスキルスコアは、完璧が 1 で基準から遠いため、まれな事象では値が小さく出る。"
          "「0を上回った割合」が高く、区間が0を含まなければ、基準より進んだと言える。"]
    (OUT / outname).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
