# 未発見シナリオのプロバイダー別分割 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Excel出力の未発見シナリオをAzure/VertexAI別に分割し、各プロバイダーでの未発見状況を個別に把握可能にする。

**Architecture:** `evaluation_result["by_area"]` に `unfound_scenarios`（統合）の代わりに `unfound_azure` / `unfound_vertex` を格納する。Excel出力のサマリーシート・詳細シートの未発見セクションをAzure/VertexAI別の列に拡張する。DB不在のプロバイダーは未発見列を空にする。

**Tech Stack:** Python, xlsxwriter (Excel出力)

---

## 現状の構造

### データフロー
```
evaluate_single_revision
  ├── keyword_filter パス → unfound_scenarios（1つ）
  └── hybrid パス → unfound_scenarios（AND演算の統合結果）
        ↓
by_area[area] = {
    "azure_results": [...],
    "vertex_results": [...],
    "correct_ids": [...],
    "unfound_scenarios": [...]   ← 統合（1つ）
}
        ↓
Excel出力
  サマリー: 未発見数(1列) | 未発見ID(1列)
  詳細:    未発見_未発見ID | 未発見_変更内容 | 未発見_ソースファイル | 未発見_質問 | 未発見_回答
```

### 変更後の構造
```
by_area[area] = {
    "azure_results": [...],
    "vertex_results": [...],
    "correct_ids": [...],
    "unfound_azure": [...]       ← Azure個別
    "unfound_vertex": [...]      ← VertexAI個別
}
        ↓
Excel出力
  サマリー: Azure未発見数 | Azure未発見ID | VertexAI未発見数 | VertexAI未発見ID
  詳細:    Azure未発見_{5列} | VertexAI未発見_{5列}
```

### DB不在時の動作
- Azure DB不在 → `unfound_azure = []`（空、検索未実行）
- VertexAI DB不在 → `unfound_vertex = []`（空、検索未実行）
- 両方DB有効 → 各プロバイダーのfound_idsで個別に未発見を算出

---

### Task 1: `evaluate_single_revision` — unfound をプロバイダー別に分離

**Files:**
- Modify: `apps/revision-ops/run_eval.py:669-681`（keyword_filterパス）
- Modify: `apps/revision-ops/run_eval.py:755-783`（hybridパス）

**Step 1: keyword_filterパス修正（行669-681）**

修正前:
```python
found_ids = self._collect_found_ids(keyword_results)
unfound_scenarios = self._build_unfound_scenarios(
    area_correct_ids, found_ids, area, revision, change_details_map
)

evaluation_result["by_area"][area] = {
    "azure_results": keyword_results if show_azure else [],
    "vertex_results": keyword_results if show_vertex else [],
    "correct_ids": area_correct_ids,
    "unfound_scenarios": unfound_scenarios,
}
```

修正後:
```python
found_ids = self._collect_found_ids(keyword_results)
# keyword_filterはプロバイダー非依存。DB存在側のみ未発見を記録
unfound = self._build_unfound_scenarios(
    area_correct_ids, found_ids, area, revision, change_details_map
)

evaluation_result["by_area"][area] = {
    "azure_results": keyword_results if show_azure else [],
    "vertex_results": keyword_results if show_vertex else [],
    "correct_ids": area_correct_ids,
    "unfound_azure": unfound if show_azure else [],
    "unfound_vertex": unfound if show_vertex else [],
}
```

**Step 2: hybridパス修正（行755-783）**

修正前:
```python
found_ids_azure = self._collect_found_ids(azure_results)
found_ids_vertex = self._collect_found_ids(vertex_results)
if providers == "both":
    azure_db = (VECTOR_DB_BASE / area / "azure_openai" / "chroma.sqlite3").exists()
    vertex_db = (VECTOR_DB_BASE / area / "vertex_ai" / "chroma.sqlite3").exists()
    if azure_db and vertex_db:
        found_ids_combined = found_ids_azure & found_ids_vertex
    elif azure_db:
        found_ids_combined = found_ids_azure
    elif vertex_db:
        found_ids_combined = found_ids_vertex
    else:
        found_ids_combined = set()
else:
    found_ids_combined = found_ids_azure | found_ids_vertex

unfound_scenarios = self._build_unfound_scenarios(
    area_correct_ids, found_ids_combined, area, revision, change_details_map
)

evaluation_result["by_area"][area] = {
    "azure_results": azure_results,
    "vertex_results": vertex_results,
    "correct_ids": area_correct_ids,
    "unfound_scenarios": unfound_scenarios,
}
```

修正後:
```python
found_ids_azure = self._collect_found_ids(azure_results)
found_ids_vertex = self._collect_found_ids(vertex_results)

# プロバイダー別の未発見を構築（DB不在のプロバイダーは空）
if providers == "both":
    azure_db = (VECTOR_DB_BASE / area / "azure_openai" / "chroma.sqlite3").exists()
    vertex_db = (VECTOR_DB_BASE / area / "vertex_ai" / "chroma.sqlite3").exists()
else:
    azure_db = providers == "azure"
    vertex_db = providers == "vertex"

unfound_azure = self._build_unfound_scenarios(
    area_correct_ids, found_ids_azure, area, revision, change_details_map
) if azure_db else []
unfound_vertex = self._build_unfound_scenarios(
    area_correct_ids, found_ids_vertex, area, revision, change_details_map
) if vertex_db else []

evaluation_result["by_area"][area] = {
    "azure_results": azure_results,
    "vertex_results": vertex_results,
    "correct_ids": area_correct_ids,
    "unfound_azure": unfound_azure,
    "unfound_vertex": unfound_vertex,
}
```

**Step 3: バッチ実行で正常終了を確認**

Run: `python apps/revision-ops/run_eval.py --provider both`
Expected: EXIT 0（Excel出力はまだ旧フォーマット参照でエラーになる可能性あり）

**注意**: この時点では `_write_summary_sheet` と `_write_single_detail_sheet` が `unfound_scenarios` を参照するためExcel出力がエラーになる。Task 2, 3で修正する。

---

### Task 2: サマリーシート — 未発見列をAzure/VertexAI別に拡張

**Files:**
- Modify: `apps/revision-ops/run_eval.py:910-930`（`_write_summary_sheet` 内データ構築）
- Modify: `apps/revision-ops/run_eval.py:948-966`（`_create_empty_summary_row`）
- Modify: `apps/revision-ops/run_eval.py:968-995`（`_write_summary_headers`）
- Modify: `apps/revision-ops/run_eval.py:1022-1057`（`_write_summary_data`）
- Modify: `apps/revision-ops/run_eval.py:939`（column_widths）

**Step 1: `_write_summary_sheet` 内のデータ構築修正（行910-930）**

修正前:
```python
unfound_scenarios = area_data.get("unfound_scenarios", [])
unfound_count = len(unfound_scenarios)
unfound_ids = ", ".join([s["シナリオID"] for s in unfound_scenarios])

summary_data.append({
    ...
    "未発見数": unfound_count,
    "未発見ID": unfound_ids,
})
```

修正後:
```python
unfound_azure = area_data.get("unfound_azure", [])
unfound_vertex = area_data.get("unfound_vertex", [])

summary_data.append({
    ...
    "Azure_未発見数": len(unfound_azure),
    "Azure_未発見ID": ", ".join([s["シナリオID"] for s in unfound_azure]),
    "VertexAI_未発見数": len(unfound_vertex),
    "VertexAI_未発見ID": ", ".join([s["シナリオID"] for s in unfound_vertex]),
})
```

**Step 2: `_create_empty_summary_row` 修正（行948-966）**

修正前:
```python
"未発見数": len(correct_ids),
"未発見ID": ", ".join(correct_ids),
```

修正後:
```python
"Azure_未発見数": len(correct_ids),
"Azure_未発見ID": ", ".join(correct_ids),
"VertexAI_未発見数": len(correct_ids),
"VertexAI_未発見ID": ", ".join(correct_ids),
```

**Step 3: `_write_summary_headers` 修正（行968-995）**

修正前:
```python
worksheet.merge_range("M1:N1", "未発見", unfound_fmt)

headers = [
    "改定番号", "エリア", "改定内容", "正解数",
    "候補数", "正解発見数", "正解発見率", "必要確認件数",
    "候補数", "正解発見数", "正解発見率", "必要確認件数",
    "未発見数", "未発見ID",
]
for col, header in enumerate(headers):
    if col < 4:
        fmt = header_fmt
    elif col < 8:
        fmt = azure_fmt
    elif col < 12:
        fmt = vertex_fmt
    else:
        fmt = unfound_fmt
    worksheet.write(1, col, header, fmt)
```

修正後:
```python
worksheet.merge_range("M1:N1", "Azure未発見", azure_fmt)
worksheet.merge_range("O1:P1", "VertexAI未発見", vertex_fmt)

headers = [
    "改定番号", "エリア", "改定内容", "正解数",
    "候補数", "正解発見数", "正解発見率", "必要確認件数",
    "候補数", "正解発見数", "正解発見率", "必要確認件数",
    "未発見数", "未発見ID",
    "未発見数", "未発見ID",
]
for col, header in enumerate(headers):
    if col < 4:
        fmt = header_fmt
    elif col < 8:
        fmt = azure_fmt
    elif col < 12:
        fmt = vertex_fmt
    elif col < 14:
        fmt = azure_fmt
    else:
        fmt = vertex_fmt
    worksheet.write(1, col, header, fmt)
```

**Step 4: `_write_summary_data` 修正（行1056-1057）**

修正前:
```python
worksheet.write(row_num, 12, row_data.get("未発見数", 0), cell_fmt)
worksheet.write(row_num, 13, row_data.get("未発見ID", ""), cell_fmt)
```

修正後:
```python
worksheet.write(row_num, 12, row_data.get("Azure_未発見数", 0), cell_fmt)
worksheet.write(row_num, 13, row_data.get("Azure_未発見ID", ""), cell_fmt)
worksheet.write(row_num, 14, row_data.get("VertexAI_未発見数", 0), cell_fmt)
worksheet.write(row_num, 15, row_data.get("VertexAI_未発見ID", ""), cell_fmt)
```

**Step 5: column_widths 修正（行939）**

修正前:
```python
column_widths = [10, 12, 15, 8, 8, 12, 12, 18, 8, 12, 12, 18, 10, 40]
```

修正後:
```python
column_widths = [10, 12, 15, 8, 8, 12, 12, 18, 8, 12, 12, 18, 10, 40, 10, 40]
```

**Step 6: autofilter の total_cols も自動で追従するため変更不要（len(column_widths)ベース）**

確認: 行944-945
```python
total_rows = len(summary_data) + 1
total_cols = len(column_widths) - 1
```
→ column_widths が16要素になるため total_cols=15 で自動追従。OK。

---

### Task 3: 詳細シート — 未発見セクションをAzure/VertexAI別に分割

**Files:**
- Modify: `apps/revision-ops/run_eval.py:1106-1178`（`_write_single_detail_sheet`）

**Step 1: ヘッダー修正（行1108, 1120-1122）**

修正前:
```python
unfound_headers = ["未発見ID", "変更内容", "ソースファイル", "質問", "回答"]
...
for header in unfound_headers:
    worksheet.write(0, col, f"未発見_{header}", formats["unfound_header"])
    col += 1
```

修正後:
```python
unfound_headers = ["未発見ID", "変更内容", "ソースファイル", "質問", "回答"]
...
for header in unfound_headers:
    worksheet.write(0, col, f"Azure未発見_{header}", formats["azure_header"])
    col += 1
for header in unfound_headers:
    worksheet.write(0, col, f"VertexAI未発見_{header}", formats["vertex_header"])
    col += 1
```

**Step 2: データ収集修正（行1124-1131）**

修正前:
```python
azure_results = []
vertex_results = []
unfound_scenarios = []
for area in data.get("areas", []):
    area_data = data.get("by_area", {}).get(area, {})
    azure_results.extend(area_data.get("azure_results", []))
    vertex_results.extend(area_data.get("vertex_results", []))
    unfound_scenarios.extend(area_data.get("unfound_scenarios", []))
```

修正後:
```python
azure_results = []
vertex_results = []
unfound_azure = []
unfound_vertex = []
for area in data.get("areas", []):
    area_data = data.get("by_area", {}).get(area, {})
    azure_results.extend(area_data.get("azure_results", []))
    vertex_results.extend(area_data.get("vertex_results", []))
    unfound_azure.extend(area_data.get("unfound_azure", []))
    unfound_vertex.extend(area_data.get("unfound_vertex", []))
```

**Step 3: zip_longest 修正（行1133-1136）**

修正前:
```python
max_rows = max(len(azure_results), len(vertex_results), len(unfound_scenarios), 1)

for row_num, (azure_row, vertex_row, unfound_row) in enumerate(
    zip_longest(azure_results, vertex_results, unfound_scenarios, fillvalue={}), start=1
):
```

修正後:
```python
max_rows = max(len(azure_results), len(vertex_results), len(unfound_azure), len(unfound_vertex), 1)

for row_num, (azure_row, vertex_row, unfound_az_row, unfound_vx_row) in enumerate(
    zip_longest(azure_results, vertex_results, unfound_azure, unfound_vertex, fillvalue={}), start=1
):
```

**Step 4: 未発見行書き込み修正（行1174）**

修正前:
```python
self._write_unfound_row(worksheet, row_num, col, unfound_row, formats)
```

修正後:
```python
self._write_unfound_row(worksheet, row_num, col, unfound_az_row, formats)
col += len(unfound_headers)
self._write_unfound_row(worksheet, row_num, col, unfound_vx_row, formats)
```

**Step 5: column_widths 修正（行1178）**

修正前:
```python
column_widths = [10, 60, 30, 50, 25, 15, 12] + [6, 18, 10, 15, 12, 50, 50, 15, 40, 40] * 2 + [18, 12, 40, 50, 50]
```

修正後:
```python
column_widths = [10, 60, 30, 50, 25, 15, 12] + [6, 18, 10, 15, 12, 50, 50, 15, 40, 40] * 2 + [18, 12, 40, 50, 50] * 2
```

---

### Task 4: 検証

**Step 1: 両方DB有効 + `--provider both` で実行**

Run: `python apps/revision-ops/run_eval.py --provider both`

確認:
- サマリーシートに Azure未発見数/ID と VertexAI未発見数/ID が別列で表示されること
- ③souzoku: Azure未発見=1 (`souzoku-bot_146`)、VertexAI未発見=0
- ⑤⑥ (keyword_filter): 両方に同一の未発見が表示されること（両方DB有効のため）

**Step 2: Azure DB無効 + `--provider both` で実行**

```bash
for dir in data/vector_db/*/azure_openai; do mv "$dir" "${dir}_bak"; done
python apps/revision-ops/run_eval.py --provider both
for dir in data/vector_db/*/azure_openai_bak; do mv "$dir" "${dir%_bak}"; done
```

確認:
- Azure未発見数/ID が全て空（DB不在で検索未実行）
- VertexAI未発見数/ID のみに値が入ること
- ⑤⑥: VertexAI未発見のみに表示

**Step 3: 詳細シートの確認**

Pythonスクリプトで③のシートのAzure未発見/VertexAI未発見列を読み取り、内容が正しいことを確認。

**Step 4: コミット**

```bash
git add apps/revision-ops/run_eval.py
git commit -m "feat: 未発見シナリオをAzure/VertexAI別に分割出力

サマリーシート・詳細シートの未発見セクションをプロバイダー別に分離。
DB不在のプロバイダーは未発見列を空にする（検索未実行として扱う）。
両方DB有効時は各プロバイダーの検索結果から個別に未発見を算出。"
```
