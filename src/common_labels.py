"""共通地点ラベル: 全期間（2001〜2025年）で観測している地点だけで作る「午後1〜6時の最大1時間降水量」。

- 那覇: 那覇・糸数・胡屋・読谷 ／ 名護: 名護・胡屋・読谷・東・本部・奥
  （安次嶺・宮城島・国頭は、2001〜2007年に未観測の年があるため使わない。v1（20mm/h・地域の正解）の定義は daily_labels.csv に残してある）
- 2010〜2025年は data/hourly.csv・region_hourly.csv、2001〜2009年は data/early/ から作る。
- 出力: data/daily_labels_common.csv（列: naha_common, nago_common = 午後1〜6時の最大1時間降水量。欠測の地点は除いて最大を取る）
使い方: .venv/bin/python src/common_labels.py   （2001〜2009年の取得が進むたびに、実行し直してよい）
"""
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
SETS = {"naha": ["naha", "itokazu", "goya", "yomitan"], "nago": ["nago", "goya", "yomitan", "higashi", "motobu", "oku"]}


def _load() -> pd.DataFrame:
    parts = []
    reg = pd.read_csv(BASE / "data" / "region_hourly.csv", dtype={"raw": str}, keep_default_na=False)
    main = pd.read_csv(BASE / "data" / "hourly.csv", parse_dates=["time"])
    early = sorted((BASE / "data" / "early").glob("region_hourly_*.csv")), sorted((BASE / "data" / "early").glob("hourly_*.csv"))
    regs = [reg] + [pd.read_csv(f, dtype={"raw": str}, keep_default_na=False) for f in early[0]]
    mains = [main] + [pd.read_csv(f, parse_dates=["time"]) for f in early[1]]
    for r in regs:
        r["date"] = pd.to_datetime(r["time"]).sub(pd.to_timedelta(r["hour"], unit="h")).dt.normalize()
        r["rain"] = pd.to_numeric(r["rain"], errors="coerce")
        parts.append(r[["station", "date", "hour", "rain"]])
    for m in mains:
        m["hour"] = m["time"].dt.hour.replace(0, 24)
        m["date"] = (m["time"] - pd.Timedelta(minutes=1)).dt.normalize()
        parts.append(m[["station", "date", "hour", "rain"]])
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    d = _load()
    aft = d[d.hour.between(14, 18)].groupby(["station", "date"]).rain.max().unstack(0)
    out = pd.DataFrame(index=aft.index)
    for tgt, sts in SETS.items():
        out[f"{tgt}_common"] = aft[sts].max(axis=1)           # skipna: 欠測の地点は除く
        out[f"{tgt}_nst"] = aft[sts].notna().sum(axis=1)       # その日に使えた地点数（確認用）
    out.sort_index().to_csv(BASE / "data" / "daily_labels_common.csv")
    y = out.groupby(out.index.year).agg(日数=("naha_common", "size"), 那覇1mm=("naha_common", lambda s: int((s >= 1).sum())),
                                        名護1mm=("nago_common", lambda s: int((s >= 1).sum())),
                                        那覇最少地点=("naha_nst", "min"), 名護最少地点=("nago_nst", "min"))
    print(y.to_string())


if __name__ == "__main__":
    main()
