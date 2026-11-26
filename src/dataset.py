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


def _hourly(early: bool = False) -> pd.DataFrame:
    """時別値。early=True なら、2001〜2009年（data/early/）も含める（既定は2010〜2025年だけ）"""
    d = pd.read_csv(BASE / "data" / "hourly.csv", parse_dates=["time"])
    if early:
        fs = sorted((BASE / "data" / "early").glob("hourly_*.csv"))
        d = pd.concat([d] + [pd.read_csv(f, parse_dates=["time"]) for f in fs], ignore_index=True)
    d["hour"] = d["time"].dt.hour.replace(0, 24)
    d["date"] = (d["time"] - pd.Timedelta(minutes=1)).dt.normalize()
    return d


def build(station: str, early: bool = False) -> pd.DataFrame:
    """1地点の標本表を返す。列: date, t, 説明変数..., y30_local など, wind_ok。
    early=True なら 2001〜2009年も含める（その年の local/region ラベルは NaN。共通地点ラベル y*_common を使う）"""
    d = _hourly(early)
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
    # 風の収束量（那覇・名護の2地点の線方向）。t・t-1・t-2 と、6時〜t の平均
    ct = conv_table()
    cv = []
    for date, t in zip(x["date"], x["t"]):
        try:
            c = ct.loc[date]
            win = c.loc[6:t]
            cv.append({"conv": c.at[t, "conv"], "conv_d1": c.at[t, "conv"] - c.at[t - 1, "conv"],
                       "conv_d2": c.at[t, "conv"] - c.at[t - 2, "conv"], "conv_mean": win["conv"].mean(),
                       "cross": c.at[t, "cross"], "spd_diff": c.at[t, "spd_diff"]})
        except KeyError:
            cv.append({})
    x = pd.concat([x, pd.DataFrame(cv)], axis=1)
    if (BASE / "data" / "wind_hourly.csv").exists():
        tt = _tri_cached()
        tv = []
        for date, t in zip(x["date"], x["t"]):
            try:
                c = tt.loc[date]
                d = {}
                for k in ("tri_n", "tri_c", "tri_s", "lsq"):
                    d[f"conv_{k}"] = c.at[t, k]
                    d[f"conv_{k}_mean"] = c.loc[6:t, k].mean()
                d["conv_lsq_d1"] = c.at[t, "lsq"] - c.at[t - 1, "lsq"]
                d["conv_lsq_d2"] = c.at[t, "lsq"] - c.at[t - 2, "lsq"]
                tv.append(d)
            except KeyError:
                tv.append({})
        x = pd.concat([x, pd.DataFrame(tv)], axis=1)
    lab = pd.read_csv(BASE / "data" / "daily_labels.csv", index_col=0, parse_dates=True)
    for kind in ("local", "region"):
        v = lab[f"{station}_{kind}"]
        for th in (1, 10, 20, 30):
            m = x["date"].map(v >= th)
            x[f"y{th}_{kind}"] = m.astype(int) if m.notna().all() else m.astype(float)   # 2001〜2009年は NaN（v1 の定義では作れない）
    cf = BASE / "data" / "daily_labels_common.csv"
    if cf.exists():  # 共通地点ラベル（全期間で観測している地点だけ）
        cl = pd.read_csv(cf, index_col=0, parse_dates=True)[f"{station}_common"]
        for th in (1, 10, 20, 30):
            m = x["date"].map(cl.notna() & (cl >= th)).astype(float)
            m[x["date"].map(cl).isna()] = np.nan
            x[f"y{th}_common"] = m.astype(int) if m.notna().all() else m
    x["year"] = x["date"].dt.year
    x["station"] = station
    return x


_DIR = {"北": 0, "北北東": 22.5, "北東": 45, "東北東": 67.5, "東": 90, "東南東": 112.5, "南東": 135, "南南東": 157.5, "南": 180,
        "南南西": 202.5, "南西": 225, "西南西": 247.5, "西": 270, "西北西": 292.5, "北西": 315, "北北西": 337.5}


def _wdir(s) -> float:
    """風向（16方位の文字）を度にする。静穏などは NaN（sin/cos も NaN）"""
    return _DIR.get(str(s).strip(), np.nan)


def _uv(g: pd.DataFrame) -> pd.DataFrame:
    """風速・風向（16方位、風が吹いてくる向き）から東西成分 u・南北成分 v（m/s）。静穏は 0。判定できない値は NaN"""
    d = g["wdir"].astype(str).str.replace(r"[)\]*#\s]", "", regex=True)
    ang = np.deg2rad(d.map(_DIR))
    u = -g["wind"] * np.sin(ang)
    v = -g["wind"] * np.cos(ang)
    calm = d.eq("静穏") | (g["wind"] <= 0.0)
    return pd.DataFrame({"u": u.mask(calm, 0.0), "v": v.mask(calm, 0.0)}, index=g.index)


# 那覇（南端）と名護（北端）を結ぶ線。線方向の単位ベクトルと距離
_LAT_S, _LON_S, _LAT_N, _LON_N = 26.207, 127.687, 26.593, 127.965
_DY = (_LAT_N - _LAT_S) * 111.2
_DX = (_LON_N - _LON_S) * 111.32 * np.cos(np.deg2rad((_LAT_N + _LAT_S) / 2))
_L = float(np.hypot(_DX, _DY))
_EX, _EY = _DX / _L, _DY / _L  # 南端（那覇）から北端（名護）へ向かう単位ベクトル


def conv_table() -> pd.DataFrame:
    """那覇・名護の風から、2地点の線方向の収束量（1/s）を (date, hour) ごとに求める。
    conv = -(北端の線方向風 − 南端の線方向風) / 距離。正なら、線に沿って風が集まる（収束）。
    cross = 線に直交する成分の差 / 距離（シア。発散ではない）。説明変数は t 以前の値だけで作る（build で選ぶ）"""
    h = _hourly()
    parts = {}
    for st in ("naha", "nago"):
        g = h[h.station == st].set_index(["date", "hour"]).sort_index()
        uv = _uv(g)
        parts[st] = pd.DataFrame({"s": uv.u * _EX + uv.v * _EY, "c": -uv.u * _EY + uv.v * _EX, "spd": g["wind"]})
    t = pd.DataFrame({"conv": -(parts["nago"].s - parts["naha"].s) / _L * 1e3,  # 単位: 1e-6 /s（(m/s)/km を 1e3 倍。1 が 1e-6 /s）
                      "cross": (parts["nago"].c - parts["naha"].c) / _L * 1e3,
                      "spd_diff": parts["nago"].spd - parts["naha"].spd})
    return t


# ---- 三角形法・最小二乗法による収束量（風を観測する沖縄本島のアメダス） ----
# 緯度・経度は気象庁の観測点一覧（amedastable.json）。(緯度, 経度)
STATION_LATLON = {"oku": (26.835, 128.272), "nago": (26.593, 127.965), "miyagijima": (26.363, 127.972),
                  "naha": (26.207, 127.687), "ashimine": (26.193, 127.638), "itokazu": (26.153, 127.765)}
TRIANGLES = {"tri_n": ("oku", "nago", "miyagijima"),          # 北部〜中部
             "tri_c": ("nago", "miyagijima", "naha"),         # 中部（南北に長い）
             "tri_s": ("miyagijima", "itokazu", "naha")}      # 中部〜南部
LSQ_STATIONS = ("oku", "nago", "miyagijima", "naha", "ashimine", "itokazu")   # 最小二乗は6地点すべて
_LAT0, _LON0 = 26.5, 127.9


def _xy(name: str) -> tuple[float, float]:
    """地点の位置を、基準点からの東向き x・北向き y（km）にする"""
    lat, lon = STATION_LATLON[name]
    return (lon - _LON0) * 111.32 * np.cos(np.deg2rad(_LAT0)), (lat - _LAT0) * 111.2


def divergence_fit(xy: np.ndarray, u: np.ndarray, v: np.ndarray) -> tuple[float, float]:
    """風が空間的に直線（1次式）で変わると仮定し、u = u0 + a x + b y、v = v0 + c x + d y を最小二乗で当てはめる。
    発散 = a + d、渦度 = c − b（単位: (m/s)/km の 1e3 倍 = 1e-6 /s）。3点なら厳密に解け（三角形法）、4点以上なら最小二乗。
    風が欠けた地点は除く。有効な点が3つ未満なら NaN"""
    ok = ~(np.isnan(u) | np.isnan(v))
    if ok.sum() < 3:
        return np.nan, np.nan
    A = np.column_stack([np.ones(ok.sum()), xy[ok, 0], xy[ok, 1]])
    cu = np.linalg.lstsq(A, u[ok], rcond=None)[0]
    cv = np.linalg.lstsq(A, v[ok], rcond=None)[0]
    return (cu[1] + cv[2]) * 1e3, (cv[1] - cu[2]) * 1e3


def _wind_all() -> pd.DataFrame:
    """6地点の風の東西成分 u・南北成分 v を (date, hour) × 地点 で返す（列: (u|v, 地点)）。fetch_wind.py の取得が済んでいること"""
    h = _hourly()
    parts = {st: h[h.station == st][["date", "hour", "wind", "wdir"]] for st in ("naha", "nago")}
    w = pd.read_csv(BASE / "data" / "wind_hourly.csv", dtype={"wdir": str})
    w["date"] = pd.to_datetime(w["date"])
    for st in ("oku", "miyagijima", "ashimine", "itokazu"):
        parts[st] = w[w.station == st][["date", "hour", "wind", "wdir"]]
    uvs = {}
    for st, g in parts.items():
        g = g.set_index(["date", "hour"]).sort_index()
        g = g[~g.index.duplicated()]
        uv = _uv(g)
        uvs[("u", st)], uvs[("v", st)] = uv["u"], uv["v"]
    return pd.DataFrame(uvs)


def tri_table() -> pd.DataFrame:
    """(date, hour) ごとに、三角形3つと最小二乗（6地点）の「収束量」（= −発散、1e-6 /s）を返す"""
    uv = _wind_all()
    out = {k: [] for k in list(TRIANGLES) + ["lsq"]}
    idx = uv.index
    xy_all = np.array([_xy(st) for st in LSQ_STATIONS])
    U = {st: uv[("u", st)].to_numpy() for st in LSQ_STATIONS}
    V = {st: uv[("v", st)].to_numpy() for st in LSQ_STATIONS}
    for i in range(len(idx)):
        for name, sts in TRIANGLES.items():
            xy = np.array([_xy(st) for st in sts])
            out[name].append(-divergence_fit(xy, np.array([U[st][i] for st in sts]), np.array([V[st][i] for st in sts]))[0])
        out["lsq"].append(-divergence_fit(xy_all, np.array([U[st][i] for st in LSQ_STATIONS]), np.array([V[st][i] for st in LSQ_STATIONS]))[0])
    return pd.DataFrame(out, index=idx)


FEATURES_BASE = ["pres", "pres_d1", "pres_d3", "temp", "dew", "dewdep", "humid", "wind", "wdir_sin", "wdir_cos",
                 "rain1", "rain3", "rain6", "rainmax6", "sun_morning", "wind_mean", "wind_max", "t", "month"]
FEATURES_RAD = ["rad_morning"]  # 那覇のみ
FEATURES_CHANGE = ["temp_d1", "temp_d2", "temp_rise6", "temp_gap_max", "dew_d1", "dew_d2", "humid_d1", "humid_d2"]


_TRI_CACHE = {}


def _tri_cached() -> pd.DataFrame:
    """tri_table は時間がかかるので、同じプロセスでは1回だけ計算する"""
    if "t" not in _TRI_CACHE:
        _TRI_CACHE["t"] = tri_table()
    return _TRI_CACHE["t"]


FEATURES_TRI = [f"conv_{k}{m}" for k in ("tri_n", "tri_c", "tri_s", "lsq") for m in ("", "_mean")] + ["conv_lsq_d1", "conv_lsq_d2"]  # 三角形法・最小二乗
FEATURES_CONV = ["conv", "conv_d1", "conv_d2", "conv_mean", "cross", "spd_diff"]  # 風の収束量（那覇・名護の2地点）


def features(station: str, change: bool = True, conv: bool = False, tri: bool = False) -> list[str]:
    return (FEATURES_BASE + (FEATURES_RAD if station == "naha" else []) + (FEATURES_CHANGE if change else [])
            + (FEATURES_CONV if conv else []) + (FEATURES_TRI if tri else []))
