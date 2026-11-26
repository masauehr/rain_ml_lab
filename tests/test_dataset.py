"""説明変数が「予測時刻 t までの観測だけ」から作られていること（未来の値を使っていないこと）を確かめる。"""
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import dataset as ds  # noqa: E402


class TestNoLeak(unittest.TestCase):
    def test_未来の値を変えても説明変数は変わらない(self):
        orig = ds._hourly()
        base = ds.build("naha")
        # 13時以降（午後）の観測をすべて壊す。t=10・11・12 の説明変数は変わらないはず
        bad = orig.copy()
        late = bad["hour"] >= 13
        for c in ("rain", "temp", "dew", "humid", "pres", "wind", "sun", "rad"):
            bad.loc[late, c] = 999.0
        with mock.patch.object(ds, "_hourly", return_value=bad):
            changed = ds.build("naha")
        cols = ds.features("naha", change=True)
        a = base[cols].to_numpy(dtype=float)
        b = changed[cols].to_numpy(dtype=float)
        self.assertTrue(np.allclose(a, b, equal_nan=True))

    def test_t時点より後の1時間は使われない(self):
        orig = ds._hourly()
        base = ds.build("nago")
        bad = orig.copy()
        # 12時の値だけ壊す → t=10・11 の標本は変わらない（t=12 は変わってよい）
        m = bad["hour"] == 12
        bad.loc[m, ["rain", "temp", "humid", "pres", "wind"]] = 999.0
        with mock.patch.object(ds, "_hourly", return_value=bad):
            changed = ds.build("nago")
        for t in (10, 11):
            s = base["t"] == t
            cols = ds.features("nago", change=True)
            self.assertTrue(np.allclose(base.loc[s, cols].to_numpy(float), changed.loc[s, cols].to_numpy(float), equal_nan=True))

    def test_収束量も未来の風を使わない(self):
        orig = ds._hourly()
        base = ds.build("nago")
        bad = orig.copy()
        late = bad["hour"] >= 13  # 午後の風をすべて壊す
        bad.loc[late, "wind"] = 99.0
        bad.loc[late, "wdir"] = "南"
        with mock.patch.object(ds, "_hourly", return_value=bad):
            changed = ds.build("nago")
        cols = ds.FEATURES_CONV
        self.assertTrue(np.allclose(base[cols].to_numpy(float), changed[cols].to_numpy(float), equal_nan=True))

    @unittest.skipUnless((ds.BASE / "data" / "wind_hourly.csv").exists(), "data/wind_hourly.csv がない（fetch_wind.py で取得）")
    def test_三角形法の収束量も未来の風を使わない(self):
        base = ds.build("naha")
        bad_uv = ds._wind_all().copy()
        bad_uv.loc[bad_uv.index.get_level_values("hour") >= 13] = 99.0   # 午後の風をすべて壊す
        with mock.patch.object(ds, "_wind_all", return_value=bad_uv):
            bad_tab = ds.tri_table()
        with mock.patch.dict(ds._TRI_CACHE, {"t": bad_tab}, clear=True):
            changed = ds.build("naha")
        cols = ds.FEATURES_TRI
        self.assertTrue(np.allclose(base[cols].to_numpy(float), changed[cols].to_numpy(float), equal_nan=True))

    def test_収束の符号(self):
        # 南端（那覇）で北向き（南風）、北端（名護）で南向き（北風）なら、線に沿って風が集まる＝収束（正）
        idx = pd.MultiIndex.from_product([[pd.Timestamp("2020-07-01")], [6, 7]], names=["date", "hour"])
        rows = []
        for st, d in (("naha", "南"), ("nago", "北")):
            for date, hour in idx:
                rows.append({"station": st, "date": date, "hour": hour, "wind": 5.0, "wdir": d})
        with mock.patch.object(ds, "_hourly", return_value=pd.DataFrame(rows)):
            t = ds.conv_table()
        self.assertTrue((t["conv"] > 0).all())

    def test_三角形法の発散を人工の風で検算(self):
        # 中心に向かって吹く風 u=-a x, v=-a y は、発散 = -2a、渦度 0（単位 1e-6/s は (m/s)/km の 1e3 倍）
        a = 0.01
        xy = np.array([ds._xy(s) for s in ds.LSQ_STATIONS])
        u, v = -a * xy[:, 0], -a * xy[:, 1]
        div3, vort3 = ds.divergence_fit(xy[:3], u[:3], v[:3])      # 3点（三角形法。厳密に解ける）
        divn, _ = ds.divergence_fit(xy, u, v)                      # 6点（最小二乗）
        self.assertAlmostEqual(div3, -2 * a * 1e3, places=6)
        self.assertAlmostEqual(divn, -2 * a * 1e3, places=6)
        self.assertAlmostEqual(vort3, 0.0, places=6)

    def test_回転する風は発散ゼロ(self):
        xy = np.array([ds._xy(s) for s in ds.LSQ_STATIONS])
        div, vort = ds.divergence_fit(xy, -0.01 * xy[:, 1], 0.01 * xy[:, 0])
        self.assertAlmostEqual(div, 0.0, places=6)
        self.assertAlmostEqual(vort, 20.0, places=6)

    def test_風が欠けた地点は除いて計算する(self):
        xy = np.array([ds._xy(s) for s in ds.LSQ_STATIONS])
        u, v = -0.01 * xy[:, 0], -0.01 * xy[:, 1]
        u[0] = np.nan
        self.assertAlmostEqual(ds.divergence_fit(xy, u, v)[0], -20.0, places=6)   # 5点でも同じ
        u[1:4] = np.nan
        self.assertTrue(np.isnan(ds.divergence_fit(xy, u, v)[0]))                 # 有効な点が3未満なら NaN

    def test_標本は1日3つ(self):
        x = ds.build("naha")
        self.assertEqual(set(x["t"]), {10, 11, 12})
        self.assertTrue((x.groupby("date").size() == 3).all())

    def test_日射量は那覇だけ(self):
        self.assertIn("rad_morning", ds.features("naha"))
        self.assertNotIn("rad_morning", ds.features("nago"))
        self.assertTrue(ds.build("nago")["rad_morning"].isna().all())


if __name__ == "__main__":
    unittest.main()
