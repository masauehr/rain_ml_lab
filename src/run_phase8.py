"""Phase 8（追加）: 基準を 10mm/h 以上にした場合（参考）。

- 設定は Phase 3 で固定したまま（10mm/h に合わせて調整し直していない）。2010〜2024年の leave-one-year-out、10シード平均。2025年は使っていない。
- 評価の内容は Phase 7 と同じ（PR-AUC・スキルスコア・Brier スキル。日単位ブートストラップの95%区間）。
- 注意: 10mm/h は「強い雨」で、「大雨」ではない。陽性率が変わるので、点数そのものは、20mm/h と直接比べない（基準との比で比べる）。
使い方: .venv/bin/python src/run_phase8.py
"""
import run_phase7 as p7

if __name__ == "__main__":
    p7.main("y10_region", "phase8.md", "Phase 8（追加）: 基準を 10mm/h にした場合（参考）", "地域の正解 × 10mm/h")
