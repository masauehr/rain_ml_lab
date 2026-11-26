# rain_ml_lab

**午前の観測から、午後の大雨を当てられるか。** 沖縄本島（那覇・名護）の7〜9月で、`ExtraTreesClassifier` を `RandomForest`・ロジスティック回帰・単純な基準と比べる機械学習の学習プロジェクト。

詳しくは [MANUAL.md](MANUAL.md) / [REPORT.md](REPORT.md) / [PLAN.md](PLAN.md) を参照。
用語が難しいときは [GLOSSARY.md](GLOSSARY.md)（初心者向けの用語集）、動かして学ぶなら [notebooks/](notebooks/README.md)（Jupyter 7本）へ。

> データは気象庁「過去の気象データ検索」の時別値を加工して作成（出典: 気象庁）。個人データは使っていない。

## 何を予測するか

午後1時〜6時の間に、**1時間降水量が基準（主: 20mm/h）以上になるか**（2クラス分類）。予測時刻は10・11・12時で、その時刻までの観測だけを使う。

- 学習: 2010〜2024年（当日6時〜予測時刻の最大風速が10m/s以上の標本は、台風を除くため学習に使わない）
- 最終評価: 2025年（1回だけ）。2026年は台風が多く傾向が違うため使わない
- 正解は、地点だけでは陽性が足りないため、**周辺のアメダス（降水量のみの地点を含む）も使った「地域の正解」**で作る

## 結論（詳しくは REPORT.md）

1. 午前の観測から午後の大雨を当てるのは難しい。PR-AUC は陽性率の**約2〜2.5倍**。
2. ExtraTrees・RandomForest・ロジスティック回帰は**ほぼ同等で、優劣は決められなかった**（差の95%区間はすべて0をまたぐ）。
3. 名護では、2025年の最終評価で単純な「直近6時間降水量」が最も高かった。陽性が5日しかなく、偶然の範囲。
4. 気温などの変化量の特徴は効かなかった。重要度は impurity より **permutation** で見るべき（順位が大きくずれる）。
5. 効くのは手法ではなく、**陽性の件数と評価の設計**。

## 構成

| パス | 内容 |
|---|---|
| `src/fetch_hourly.py` / `src/fetch_region.py` | 気象庁の時別値の取得（那覇・名護 / 周辺9地点） |
| `src/fetch_wind.py` | 風の追加取得（奥・宮城島・安次嶺・糸数。収束量の実験用。約2時間） |
| `src/run_phase6.py` / `run_phase6b.py` | 風の収束量の実験（2地点法 / 三角形法・最小二乗法） |
| `notebooks/` | 動かして学ぶ Jupyter（用語と手法の解説） |
| `GLOSSARY.md` | 初心者向けの用語集 |
| `src/check_stations.py` / `src/region_labels.py` | 降水量の観測地点の検証 / 正解（地点・地域）の集計 |
| `src/dataset.py` | 説明変数と正解（予測時刻 t までの観測だけ） |
| `src/run_phase1.py`〜`run_phase5.py` | 基準 → 比較 → 差の確認・しきい値 → 重要度・校正 → 最終評価 |
| `outputs/` | 各 Phase の結果（`phase1.md`〜`phase5.md`、`station_check.md`、`region_labels.md`） |
| `data/` | `hourly.csv`・`region_hourly.csv`・`daily_labels.csv`（取得済みのデータ） |
| `tests/` | 説明変数が未来の値を使っていないことのテスト |

## 使い方

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests
.venv/bin/python src/run_phase1.py      # 〜 run_phase4.py
```

データの再取得（約5時間）は MANUAL.md を参照。`run_phase5.py`（2025年の最終評価）は使用済みのため、再実行は拒否される。

## ライセンス・出典

コードは自由に使ってよい。データは気象庁の利用規約に従い、「気象庁」を出典として明記して加工したもの。
