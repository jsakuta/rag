# 未発見セクションの書式変更プラン

## 概要
`evaluate_revisions.py`の詳細シートにおいて、「未発見」セクションのデータ行をオレンジ色から白背景に変更する。

## 変更対象
- **ファイル**: `scripts/evaluate_revisions.py`
- **メソッド**: `_write_unfound_row` (841-847行)

## 現状
未発見セクションの全セル（ヘッダー・データ両方）がオレンジ色（`#FDE9D9`）で色付けされている。

## 変更内容
`_write_unfound_row`メソッドで、データセルの書式を`formats["unfound_cell"]`から`formats["cell"]`に変更する。

```python
# 変更前 (847行)
worksheet.write(row_num, start_col + i, value if value != "" else "", formats["unfound_cell"])

# 変更後
worksheet.write(row_num, start_col + i, value if value != "" else "", formats["cell"])
```

## 結果
- **ヘッダー行**（「未発見_未発見ID」「未発見_変更内容」等）: オレンジ色（維持）
- **データ行**: 白背景（通常セル書式）

## 検証方法
1. `python scripts/evaluate_revisions.py` を実行
2. 出力された `output/revision_evaluation_*.xlsx` を開く
3. 詳細シートの「未発見」セクションで、ヘッダーのみオレンジ色、データ行は白背景になっていることを確認
