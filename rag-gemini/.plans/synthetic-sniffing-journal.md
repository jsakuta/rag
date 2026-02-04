# 評価スクリプト改善計画

## 概要

`scripts/evaluate_revisions.py` に3つの変更を実施する。

---

## 変更1: カテゴリ列の削除

「マッチ種別」列（Both/Original_Only/LLM_Enhanced_Only/Keyword）のみを残し、「カテゴリ」列を削除。

### 変更箇所

| 行番号 | 変更内容 |
|--------|----------|
| 387 | `"カテゴリ": category,` を削除 |
| 436 | `"カテゴリ": category,` を削除 |
| 1025 | `result_headers` から「カテゴリ」を削除 |
| 1026 | `unfound_headers` から「カテゴリ」を削除 |
| 1096 | 列幅設定から「カテゴリ」列の幅(30)を削除 |
| 1110 | `keys` から「カテゴリ」を削除 |
| 1119 | `_write_unfound_row` の `keys` から「カテゴリ」を削除 |

---

## 変更2: ソース列が空の問題の修正

### 原因

類似検索（hybrid）時、`question.split(" > ")[0]` でLv1を抽出しているが、質問テキストに「 > 」が含まれていない場合、Lv1が空になりソースファイルが取得できない。

### 解決策

`multi_stage_orchestrator.py` の `_build_result_data` メソッドで `Hierarchy` キーを結果に追加し、`evaluate_revisions.py` でフォールバックとして使用する。

### 変更箇所

**ファイル: `src/core/search/multi_stage_orchestrator.py`**

`_build_result_data` メソッド内（行244付近）に `Hierarchy` キーを追加:
```python
SearchResultKeys.HIERARCHY: metadata.get(MetadataKeys.HIERARCHY, ''),
```

**ファイル: `src/types/search_types.py`**

`SearchResultKeys` クラスに `HIERARCHY` 定数を追加（既存の場合は確認のみ）

**ファイル: `scripts/evaluate_revisions.py`**

`_convert_result_to_dict` メソッド（行419-424）を修正:
```python
question = result.get(SearchResultKeys.SEARCH_RESULT_Q, "")
lv1 = ""
if " > " in question:
    lv1 = question.split(" > ")[0]
else:
    # フォールバック: Hierarchy から直接取得
    hierarchy = result.get(SearchResultKeys.HIERARCHY, "")
    if hierarchy:
        lv1 = hierarchy.split(" > ")[0]
source_file = self._get_source_file(revision, bot_name, lv1)
```

---

## 変更3: サマリーの色分け

AzureとVertexAIを比較して優劣を色で表示:
- **正解発見率**: 高い方を青字、低い方を赤字
- **必要確認件数**: 低い方を青字、高い方を赤字

### 変更箇所

**ファイル: `scripts/evaluate_revisions.py`**

1. `_create_excel_formats` メソッド（行816-834）に青字・赤字フォーマットを追加:
```python
"good_percent": workbook.add_format({
    **base_style, "num_format": "0.0%", "valign": "top", "font_color": "#0000FF"
}),
"good_cell": workbook.add_format({
    **base_style, "valign": "top", "font_color": "#0000FF"
}),
"bad_percent": workbook.add_format({
    **base_style, "num_format": "0.0%", "valign": "top", "font_color": "#FF0000"
}),
"bad_cell": workbook.add_format({
    **base_style, "valign": "top", "font_color": "#FF0000"
}),
```

2. `_write_summary_data` メソッド（行954-975）で比較ロジックを追加:
   - 正解発見率: `azure_rate > vertex_rate` なら Azure青・VertexAI赤
   - 必要確認件数: `azure_check < vertex_check` なら Azure青・VertexAI赤
   - 同値の場合は通常フォーマット
   - "-" の場合は比較対象外

---

## 変更対象ファイル

| ファイルパス | 変更内容 |
|-------------|----------|
| `scripts/evaluate_revisions.py` | カテゴリ列削除、ソース列修正、サマリー色分け |
| `src/core/search/multi_stage_orchestrator.py` | Hierarchyキー追加 |
| `src/types/search_types.py` | HIERARCHY定数確認/追加 |

---

## 検証方法

1. `python scripts/evaluate_revisions.py` を実行
2. 出力されたExcelファイル（`output/evaluation_YYYYMMDD_HHMMSS.xlsx`）を確認:
   - 詳細シートに「カテゴリ」列がないことを確認
   - 「ソースファイル」列にファイル名が表示されていることを確認
   - サマリーシートで正解発見率・必要確認件数の色分けを確認
