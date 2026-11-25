"""説明変数と正解を作る。標本 = (地点, 日, 予測時刻 t)。t は 10・11・12 時。

- 説明変数は「t 時までの観測だけ」から作る（未来の値は使わない）。
- 時別値の hour h は h-1時〜h時の値。6時〜t は hour 6..t（風・気温の基準）、日照・日射の合計は hour 7..t。
- 正解: 午後1〜6時（hour 14〜18）の最大1時間降水量が基準以上か。地点だけ（local）と、周辺アメダスを含む地域（region）の両方。
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
TIMES = (10, 11, 12)
WIND_LIMIT = 10.0  # 台風の除外: 当日6時〜t の最大風速がこれ以上の標本は学習に使わない


def _hourly() -> pd.DataFrame:
    d = pd.read_csv(BASE / "data" / "hourly.csv", parse_dates=["time"])
    d["hour"] = d["time"].dt.hour.replace(0, 24)
    d["date"] = (d["time"] - pd.Timedelta(minutes=1)).dt.normalize()
    return d


def build(station: str) -> pd.DataFrame:
    """1地点の標本表を返す。列: date, t, 説明変数..., y30_local など, wind_ok"""
    d = _hourly()
    d = d[d.station == station].set_index(["date", "hour"]).sort_index()
    rows = []
    for date, g in d.groupby(level=0):
        g = g.droplevel(0)  # index = hour
        if not {6, 7, 8, 9, 10, 11, 12} <= set(g.index):
            continue
        for t in TIMES:
            h = lambda c, k: g.at[t - k, c] if (t - k) in g.index else np.nan  # t-k 時の値
            win = g.loc[6:t]  # 6時〜t
            morn = g.loc[7:t]  # 日照・日射の合計に使う（6時から t まで）
            f = {
                "date": date, "t": t, "month": date.month,
                "pres": h("pres", 0), "pres_d1": h("pres", 0) - h("pres", 1), "pres_d3": h("pres", 0) - h("pres", 3),
                "temp": h("temp", 0), "dew": h("dew", 0), "dewdep": h("temp", 0) - h("dew", 0), "humid": h("humid", 0),
                "wind": h("wind", 0),
                "temp_d1": h("temp", 0) - h("temp", 1), "temp_d2": h("temp", 0) - h("temp", 2),
                "temp_rise6": h("temp", 0) - g.at[6, "temp"], "temp_gap_max": h("temp", 0) - win["temp"].max(),
                "dew_d1": h("dew", 0) - h("dew", 1), "dew_d2": h("dew", 0) - h("dew", 2),
                "humid_d1": h("humid", 0) - h("humid", 1), "humid_d2": h("humid", 0) - h("humid", 2),
                "rain1": h("rain", 0), "rain3": g.loc[t - 2:t, "rain"].sum(), "rain6": g.loc[max(t - 5, 1):t, "rain"].sum(),
                "rainmax6": g.loc[max(t - 5, 1):t, "rain"].max(),
                "sun_morning": morn["sun"].sum(min_count=1), "wind_mean": win["wind"].mean(), "wind_max": win["wind"].max(),
            }
            rad = morn["rad"].sum(min_count=1)
            f["rad_morning"] = rad if station == "naha" else np.nan  # 全天日射量は那覇のみ
            wd = np.deg2rad(_wdir(g.at[t, "wdir"]))
            f["wdir_sin"], f["wdir_cos"] = np.sin(wd), np.cos(wd)
            f["wind_ok"] = bool(win["wind"].max() < WIND_LIMIT)
            rows.append(f)
    x = pd.DataFrame(rows)
    lab = pd.read_csv(BASE / "data" / "daily_labels.csv", index_col=0, parse_dates=True)
    for kind in ("local", "region"):
        v = lab[f"{station}_{kind}"]
        for th in (20, 30):
            x[f"y{th}_{kind}"] = x["date"].map(v >= th).astype(int)
    x["year"] = x["date"].dt.year
    x["station"] = station
    return x


_DIR = {"北": 0, "北北東": 22.5, "北東": 45, "東北東": 67.5, "東": 90, "東南東": 112.5, "南東": 135, "南南東": 157.5, "南": 180,
        "南南西": 202.5, "南西": 225, "西南西": 247.5, "西": 270, "西北西": 292.5, "北西": 315, "北北西": 337.5}


def _wdir(s) -> float:
    """風向（16方位の文字）を度にする。静穏などは NaN（sin/cos も NaN）"""
    return _DIR.get(str(s).strip(), np.nan)


FEATURES_BASE = ["pres", "pres_d1", "pres_d3", "temp", "dew", "dewdep", "humid", "wind", "wdir_sin", "wdir_cos",
                 "rain1", "rain3", "rain6", "rainmax6", "sun_morning", "wind_mean", "wind_max", "t", "month"]
FEATURES_RAD = ["rad_morning"]  # 那覇のみ
FEATURES_CHANGE = ["temp_d1", "temp_d2", "temp_rise6", "temp_gap_max", "dew_d1", "dew_d2", "humid_d1", "humid_d2"]


def features(station: str, change: bool = True) -> list[str]:
    return FEATURES_BASE + (FEATURES_RAD if station == "naha" else []) + (FEATURES_CHANGE if change else [])
