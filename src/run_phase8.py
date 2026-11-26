"""Phase 8（追加・参考）: 基準を 10mm/h・1mm/h など、20mm/h より低くした場合。

- 設定は Phase 3 で固定したまま（基準に合わせて調整し直していない）。2010〜2024年の leave-one-year-out、10シード平均。2025年は使っていない。
- 評価の内容は Phase 7 と同じ（PR-AUC・スキルスコア・Brier スキル。日単位ブートストラップの95%区間）。
- 注意: 陽性率が変わるので、点数そのものは、20mm/h と直接比べない（基準との比・スキルスコアで比べる）。
  10mm/h は「強い雨」、1mm/h は「午後に雨が降るか」で、「大雨」ではない。
使い方: .venv/bin/python src/run_phase8.py [しきい値mm/h。既定 10]   → outputs/phase8.md（10）／phase8_1mm.md（1）
"""
import sys

import run_phase7 as p7

if __name__ == "__main__":
    th = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    out = "phase8.md" if th == 10 else f"phase8_{th}mm.md"
    p7.main(f"y{th}_region", out, f"Phase 8（追加・参考）: 基準を {th}mm/h にした場合", f"地域の正解 × {th}mm/h")
