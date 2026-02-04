# 評価結果Excel出力の改善計画（追加修正）

## 概要
`evaluate_revisions.py`の出力Excelを以下のように改善する：

### 完了済み
1. **順位を追加**: Azure/VertexAIそれぞれの検索結果順位を表示
2. **カテゴリ→マッチ種別に名称変更**: Search_Category列の名称を変更

### 追加修正（今回）
3. **カテゴリ列の修正**: Lv1〜Lv4を結合した値を表示（例：「預金関連 > 普通預金 > 新規口座開設」）
4. **サマリーシート**: 事務改定内容の全文を出力（50文字プレビュー→全文）
5. **詳細シート**: 事務改定内容を全行に出力（1行目のみ→全行）

---

## 修正対象ファイル

**ファイル**: `scripts/evaluate_revisions.py`

### 修正1: カテゴリ抽出をLv1〜Lv4に戻す（_convert_result_to_dict）

現在の実装ではエリア名から日本語名を取得しているが、Search_Result_QからLv1〜Lv4を抽出する方式に戻す。

```python
# 現在（誤り）
category = self._extract_category_from_area(area)

# 修正後
# Search_Result_Qは「Lv1 > Lv2 > Lv3 > Lv4 > 質問」形式
# 最後の「 > 」より前を取得してカテゴリとする
search_result_q = result.get(SearchResultKeys.SEARCH_RESULT_Q, "")
parts = search_result_q.rsplit(" > ", 1)
category = parts[0] if len(parts) > 1 else ""
```

### 修正2: キーワード検索のカテゴリもLv1〜Lv4に戻す（_execute_keyword_filter_search）

```python
# 現在（誤り）
category = self._extract_category_from_area(area)

# 修正後
# カテゴリ（Lv1〜Lv4を結合）
category_parts = []
for col in ["Lv1", "Lv2", "Lv3", "Lv4"]:
    if col in df.columns and pd.notna(row.get(col)):
        category_parts.append(str(row[col]))
category = " > ".join(category_parts)
```

### 修正3: サマリーシートで全文表示（_write_summary_data）

```python
# 現在
content_preview = (
    revision_content[:50] + "..."
    if len(revision_content) > 50
    else revision_content
)

# 修正後
# 全文を表示
summary_data.append({
    ...
    "改定内容": revision_content,  # 全文
    ...
})
```

### 修正4: 詳細シートで全行に事務改定内容を表示（_write_single_detail_sheet）

```python
# 現在（1行目のみ）
if row_num == 1:
    worksheet.write(row_num, 1, data["revision_content"], formats["cell"])

# 修正後（全行に表示）
worksheet.write(row_num, 1, data["revision_content"], formats["cell"])
```

---

## 検証方法

1. `python scripts/evaluate_revisions.py` を実行
2. 出力された `output/revision_evaluation_*.xlsx` を開く
3. 確認項目:
   - **カテゴリ列**: 「預金関連 > 普通預金 > 新規口座開設」のようなLv1〜Lv4の結合値が表示される
   - **サマリーシート**: 改定内容列に全文が表示される
   - **詳細シート**: 全行に事務改定内容が表示される
