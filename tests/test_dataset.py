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
