# rain_ml_lab マニュアル

午前の観測から午後の大雨を当てる学習プロジェクト（ExtraTrees・RandomForest・ロジスティック回帰の比較）。
結論は [REPORT.md](REPORT.md)、設計は [PLAN.md](PLAN.md)。
用語は [GLOSSARY.md](GLOSSARY.md)（初心者向け）、動かして学ぶ Jupyter は [notebooks/README.md](notebooks/README.md)。

## セットアップ

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests     # 4件
```

## データ

| ファイル | 内容 |
|---|---|
| `data/hourly.csv` | 那覇・名護の1時間値（気圧・降水量・気温・露点・湿度・風・日照・日射）。2010〜2025年の7〜9月 |
| `data/region_hourly.csv` | 周辺9地点の1時間降水量（生の文字列と数値） |
| `data/daily_labels.csv` | 午後1〜6時の最大1時間降水量（地点・地域） |

再取得する場合（気象庁のサーバーに負荷をかけないよう、1件ごとに1秒待つ。取得済みの日は再取得しない）:

```
.venv/bin/python src/fetch_hourly.py --years 2010 ... 2025   # 約2,900件・約1時間
.venv/bin/python src/fetch_region.py                          # 約13,000件・約4時間
.venv/bin/python src/region_labels.py                         # 正解の集計
```

注意点:
- 那覇・名護は `hourly_s1`、周辺のアメダスは `hourly_a1` で取得する（名護は a1 が表示できない）。
- 「--」は降水なし（0mm）、「)」「]」付きの値は記号を外して数値にする。
- 時別値の hour h は、h-1時〜h時の値として扱っている（実データでの確認は未了。REPORT.md の限界）。

## 実行

```
.venv/bin/python src/run_phase1.py   # 基準（常に0・直近降水量・ロジスティック回帰）
.venv/bin/python src/run_phase2.py   # ExtraTrees と RandomForest の比較（10シード。約30分）
.venv/bin/python src/run_phase3.py   # 差の95%区間・警戒の出し方・校正（約3分）
.venv/bin/python src/run_phase4.py   # 特徴量重要度・群ごとの寄与・Platt補正（10〜20分）
.venv/bin/python src/run_phase5.py   # 2025年の最終評価（1回だけ）

# 追加実験: 風の収束量（先に風の追加データを取得する。約2時間・再開可能）
.venv/bin/python src/fetch_wind.py       # data/wind_hourly.csv を作る
.venv/bin/python src/run_phase6.py       # 2地点法
.venv/bin/python src/run_phase6b.py      # 三角形法・最小二乗法
.venv/bin/python src/run_phase7.py       # スキルスコア（基準からどれだけ進んだか）
.venv/bin/python src/run_phase8.py       # 基準を 10mm/h にした場合（参考）
```

結果は `outputs/phase*.md` に出る。`run_phase5.py` は `outputs/final/TEST_USED.json` を作り、以後は拒否する（結果を見て設定を変えないため）。
新しい検証データで評価し直すときは、`TEST_USED.json` を残したまま、新しい年のデータで別の実行を作る。

## ノートブック（動かして学ぶ）

`notebooks/` に、用語と手法を実データで学ぶ Jupyter が7本ある（00 データ / 01 評価 / 02 木と森 / 03 不確かさ / 04 重要度 / 05 風の収束量 / 06 校正）。
`.venv/bin/jupyter lab notebooks/` で開く。05 の一部だけ、`data/wind_hourly.csv` が必要。読む順と結果は [notebooks/README.md](notebooks/README.md)。
コミットするノートは出力を消してある。

## 設定の要点

- 説明変数は、予測時刻 t（10・11・12時）までの観測だけ。`tests/test_dataset.py` で、午後の観測を壊しても変わらないことを確認している。
- 学習は、当日6時〜t の最大風速が 10m/s 未満の標本だけ（台風の除外）。
- 全天日射量は那覇のみ（名護は観測なし）。
- 評価は、日単位（年単位）で分ける。同じ日の3標本は同じ側に入れる。

## 出典

気象庁「過去の気象データ検索」の時別値を加工して作成。
