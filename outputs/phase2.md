# Phase 2: ExtraTrees と RandomForest の比較

（`src/run_phase2.py` が自動生成。2010〜2024年の leave-one-year-out、10シードの平均±標準偏差。2025年は使っていない）

評価は風速10m/s未満の標本。PR-AUC の基準（常に0）は陽性標本の割合。

## 地域の正解 × 20mm/h（主） / 那覇（陽性 130 標本）

| 設定 | モデル | PR-AUC | ROC-AUC |
|---|---|---|---|
| ロジスティック回帰（基準） | logreg | 0.075 ± 0.000 | 0.694 |
| 基準の設定 | ExtraTrees | 0.080 ± 0.004 | 0.690 |
| 基準の設定 | RandomForest | 0.079 ± 0.004 | 0.712 |
| class_weight なし | ExtraTrees | 0.079 ± 0.003 | 0.690 |
| class_weight なし | RandomForest | 0.090 ± 0.003 | 0.700 |
| min_samples_leaf=1 | ExtraTrees | 0.068 ± 0.003 | 0.670 |
| min_samples_leaf=1 | RandomForest | 0.072 ± 0.002 | 0.699 |
| min_samples_leaf=20 | ExtraTrees | 0.084 ± 0.003 | 0.705 |
| min_samples_leaf=20 | RandomForest | 0.086 ± 0.002 | 0.722 |
| max_features=0.5 | ExtraTrees | 0.070 ± 0.002 | 0.679 |
| max_features=0.5 | RandomForest | 0.069 ± 0.002 | 0.692 |
| 木の数 50 | ExtraTrees | 0.079 ± 0.007 | 0.684 |
| 木の数 50 | RandomForest | 0.077 ± 0.006 | 0.702 |
| 木の数 600 | ExtraTrees | 0.078 ± 0.003 | 0.691 |
| 木の数 600 | RandomForest | 0.082 ± 0.003 | 0.715 |
| 変化量の特徴なし | ExtraTrees | 0.077 ± 0.005 | 0.691 |
| 変化量の特徴なし | RandomForest | 0.083 ± 0.002 | 0.722 |

## 地域の正解 × 20mm/h（主） / 名護（陽性 205 標本）

| 設定 | モデル | PR-AUC | ROC-AUC |
|---|---|---|---|
| ロジスティック回帰（基準） | logreg | 0.106 ± 0.000 | 0.686 |
| 基準の設定 | ExtraTrees | 0.124 ± 0.005 | 0.692 |
| 基準の設定 | RandomForest | 0.114 ± 0.003 | 0.694 |
| class_weight なし | ExtraTrees | 0.116 ± 0.004 | 0.695 |
| class_weight なし | RandomForest | 0.109 ± 0.002 | 0.701 |
| min_samples_leaf=1 | ExtraTrees | 0.109 ± 0.005 | 0.681 |
| min_samples_leaf=1 | RandomForest | 0.111 ± 0.003 | 0.684 |
| min_samples_leaf=20 | ExtraTrees | 0.114 ± 0.004 | 0.692 |
| min_samples_leaf=20 | RandomForest | 0.108 ± 0.003 | 0.692 |
| max_features=0.5 | ExtraTrees | 0.125 ± 0.003 | 0.697 |
| max_features=0.5 | RandomForest | 0.122 ± 0.003 | 0.694 |
| 木の数 50 | ExtraTrees | 0.119 ± 0.003 | 0.687 |
| 木の数 50 | RandomForest | 0.109 ± 0.004 | 0.681 |
| 木の数 600 | ExtraTrees | 0.126 ± 0.002 | 0.694 |
| 木の数 600 | RandomForest | 0.116 ± 0.002 | 0.697 |
| 変化量の特徴なし | ExtraTrees | 0.127 ± 0.005 | 0.700 |
| 変化量の特徴なし | RandomForest | 0.118 ± 0.004 | 0.699 |

## 地域の正解 × 30mm/h（参考） / 那覇（陽性 45 標本）

| 設定 | モデル | PR-AUC | ROC-AUC |
|---|---|---|---|
| ロジスティック回帰（基準） | logreg | 0.035 ± 0.000 | 0.686 |
| 基準の設定 | ExtraTrees | 0.023 ± 0.001 | 0.740 |
| 基準の設定 | RandomForest | 0.020 ± 0.001 | 0.698 |
| class_weight なし | ExtraTrees | 0.024 ± 0.001 | 0.742 |
| class_weight なし | RandomForest | 0.024 ± 0.001 | 0.705 |
| min_samples_leaf=1 | ExtraTrees | 0.020 ± 0.001 | 0.663 |
| min_samples_leaf=1 | RandomForest | 0.017 ± 0.001 | 0.626 |
| min_samples_leaf=20 | ExtraTrees | 0.024 ± 0.001 | 0.756 |
| min_samples_leaf=20 | RandomForest | 0.023 ± 0.001 | 0.729 |
| max_features=0.5 | ExtraTrees | 0.025 ± 0.001 | 0.750 |
| max_features=0.5 | RandomForest | 0.020 ± 0.001 | 0.693 |
| 木の数 50 | ExtraTrees | 0.022 ± 0.001 | 0.719 |
| 木の数 50 | RandomForest | 0.019 ± 0.002 | 0.657 |
| 木の数 600 | ExtraTrees | 0.024 ± 0.000 | 0.748 |
| 木の数 600 | RandomForest | 0.021 ± 0.000 | 0.708 |
| 変化量の特徴なし | ExtraTrees | 0.025 ± 0.001 | 0.748 |
| 変化量の特徴なし | RandomForest | 0.020 ± 0.001 | 0.695 |

## 地域の正解 × 30mm/h（参考） / 名護（陽性 93 標本）

| 設定 | モデル | PR-AUC | ROC-AUC |
|---|---|---|---|
| ロジスティック回帰（基準） | logreg | 0.045 ± 0.000 | 0.564 |
| 基準の設定 | ExtraTrees | 0.050 ± 0.002 | 0.626 |
| 基準の設定 | RandomForest | 0.042 ± 0.002 | 0.611 |
| class_weight なし | ExtraTrees | 0.051 ± 0.002 | 0.623 |
| class_weight なし | RandomForest | 0.043 ± 0.001 | 0.643 |
| min_samples_leaf=1 | ExtraTrees | 0.042 ± 0.002 | 0.604 |
| min_samples_leaf=1 | RandomForest | 0.036 ± 0.002 | 0.579 |
| min_samples_leaf=20 | ExtraTrees | 0.053 ± 0.003 | 0.630 |
| min_samples_leaf=20 | RandomForest | 0.047 ± 0.002 | 0.631 |
| max_features=0.5 | ExtraTrees | 0.048 ± 0.002 | 0.625 |
| max_features=0.5 | RandomForest | 0.038 ± 0.001 | 0.600 |
| 木の数 50 | ExtraTrees | 0.047 ± 0.004 | 0.618 |
| 木の数 50 | RandomForest | 0.041 ± 0.002 | 0.607 |
| 木の数 600 | ExtraTrees | 0.051 ± 0.001 | 0.627 |
| 木の数 600 | RandomForest | 0.043 ± 0.001 | 0.612 |
| 変化量の特徴なし | ExtraTrees | 0.044 ± 0.001 | 0.632 |
| 変化量の特徴なし | RandomForest | 0.035 ± 0.001 | 0.617 |

