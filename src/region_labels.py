"""全期間の品質確認と、「午後1〜6時に基準以上の1時間降水量があったか」の正解（地点・地域）の集計。

- 午後1〜6時 = 表の hour 14〜18（hour h は h-1時〜h時の降水量）
- 地域: 那覇＝南部3＋中部3（那覇自身を含む6地点）、名護＝中部3＋北部5（名護自身を含む8地点）
- 結果は outputs/region_labels.md と data/daily_labels.csv に保存
"""
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
NAMES = {"itokazu": "糸数", "ashimine": "安次嶺", "goya": "胡屋", "miyagijima": "宮城島", "yomitan": "読谷",
         "higashi": "東", "motobu": "本部", "kunigami": "国頭", "oku": "奥", "naha": "那覇", "nago": "名護"}
REGION = {"naha": ["naha", "itokazu", "ashimine", "goya", "miyagijima", "yomitan"],
          "nago": ["nago", "goya", "miyagijima", "yomitan", "higashi", "motobu", "kunigami", "oku"]}


def load() -> pd.DataFrame:
    """全11地点の1時間降水量を (station, date, hour, rain) にそろえる"""
    reg = pd.read_csv(BASE / "data" / "region_hourly.csv", dtype={"raw": str}, keep_default_na=False)
    reg["date"] = pd.to_datetime(reg["time"]).sub(pd.to_timedelta(reg["hour"], unit="h")).dt.normalize()
    reg["rain"] = pd.to_numeric(reg["rain"], errors="coerce")
    main = pd.read_csv(BASE / "data" / "hourly.csv", parse_dates=["time"])
    main["hour"] = main["time"].dt.hour.replace(0, 24)
    main["date"] = (main["time"] - pd.Timedelta(minutes=1)).dt.normalize()
    return pd.concat([reg[["station", "date", "hour", "rain", "raw"]], main[["station", "date", "hour", "rain"]]], ignore_index=True)


def main() -> None:
    d = load()
    L = ["# 周辺地点を含む正解の集計（全期間 2010〜2025年の7〜9月）", "", "（`src/region_labels.py` が自動生成）", ""]

    # 品質
    q = d.groupby("station").agg(時間数=("rain", "size"), 欠測=("rain", lambda s: int(s.isna().sum())))
    nop = d[d["raw"].astype(str).eq("NOPAGE")].groupby("station").size() if "raw" in d else pd.Series(dtype=int)
    flag = d[d["raw"].astype(str).str.contains(r"[)\]*#]", na=False)].groupby("station").size()
    L += ["## 全期間の品質", "", "| 地点 | 時間数 | 欠測 | ページなしの時間 | 注意記号の時間 |", "|---|---|---|---|---|"]
    for st in NAMES:
        L.append(f"| {NAMES[st]} | {q.loc[st, '時間数']} | {q.loc[st, '欠測']} | {int(nop.get(st, 0))} | {int(flag.get(st, 0))} |")

    # 日ごとの午後1〜6時の最大
    aft = d[d.hour.between(14, 18)].groupby(["station", "date"]).rain.max().unstack(0)
    aft.index = pd.to_datetime(aft.index)
    out = pd.DataFrame(index=aft.index)
    for tgt, sts in REGION.items():
        out[f"{tgt}_local"] = aft[tgt]
        out[f"{tgt}_region"] = aft[sts].max(axis=1)
    out.to_csv(BASE / "data" / "daily_labels.csv")

    for th in (30, 20):
        L += ["", f"## 陽性の日数（午後1〜6時に {th}mm/h 以上）", "", "| 年 | 那覇 地点 | 那覇 地域 | 名護 地点 | 名護 地域 |", "|---|---|---|---|---|"]
        p = (out >= th).groupby(out.index.year).sum()
        for y, r in p.iterrows():
            L.append(f"| {y} | {r.naha_local} | {r.naha_region} | {r.nago_local} | {r.nago_region} |")
        t = (out >= th).sum()
        L.append(f"| **計** | **{t.naha_local}** | **{t.naha_region}** | **{t.nago_local}** | **{t.nago_region}** |")
        L.append(f"| 学習(2010-24) | {p.loc[:2024].naha_local.sum()} | {p.loc[:2024].naha_region.sum()} | {p.loc[:2024].nago_local.sum()} | {p.loc[:2024].nago_region.sum()} |")
        L.append(f"| 検証(2025) | {p.loc[2025].naha_local} | {p.loc[2025].naha_region} | {p.loc[2025].nago_local} | {p.loc[2025].nago_region} |")
    L.append(f"\n日数: {len(out)}（各地点 1,472日）")

    # 官署と近傍アメダスの全日の相関
    L += ["", "## 官署と近傍アメダスの整合（全期間）", "", "| 官署 | 比べた地点 | 1時間値の相関 | 日降水量の相関 |", "|---|---|---|---|"]
    for kan, near in (("naha", "ashimine"), ("nago", "motobu")):
        w = d[d.station.isin([kan, near])].pivot_table(index=["date", "hour"], columns="station", values="rain").dropna()
        dd = w.groupby(level=0).sum()
        L.append(f"| {NAMES[kan]} | {NAMES[near]} | {w[kan].corr(w[near]):.2f} | {dd[kan].corr(dd[near]):.2f} |")

    (BASE / "outputs").mkdir(exist_ok=True)
    (BASE / "outputs" / "region_labels.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
