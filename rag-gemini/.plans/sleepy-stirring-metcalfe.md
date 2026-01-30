# サマリーシート改善計画（エリア別行出力対応）

## 概要

`scripts/evaluate_revisions.py`の出力Excelファイルのサマリーシートを改善する。
特に、事務改定③のように複数エリア（DB）を持つ改定については、エリアごとに1行ずつ出力する。

## 現状

事務改定③は4つのエリアを持つ：
```python
'③': ['rev03naibujimu', 'rev03smile', 'rev03souzoku', 'rev03torikaku']
```

**現在の出力（1行にまとめられている）**:
| 改定番号 | 改定内容 | 正解数 | Azure候補数 | ... |
|---------|---------|-------|------------|-----|
| ③ | 保険証→資格確認証... | 20 | 400 | ... |

**要望する出力（エリアごとに1行）**:
| 改定番号 | エリア | 改定内容 | 正解数 | Azure候補数 | ... |
|---------|--------|---------|-------|------------|-----|
| ③ | rev03naibujimu | 保険証→... | 5 | 100 | ... |
| ③ | rev03smile | 保険証→... | 8 | 100 | ... |
| ③ | rev03souzoku | 保険証→... | 4 | 100 | ... |
| ③ | rev03torikaku | 保険証→... | 3 | 100 | ... |

---

## 実装計画

### タスク1: `evaluate_revision`の戻り値を変更

**現在**: 全エリアの結果をまとめて返す
**変更後**: エリアごとの結果を辞書で返す

```python
# 現在の戻り値
{
    'azure_results': [全エリアの結果リスト],
    'vertex_results': [全エリアの結果リスト],
    ...
}

# 変更後の戻り値
{
    'areas': ['rev03naibujimu', 'rev03smile', ...],
    'by_area': {
        'rev03naibujimu': {
            'azure_results': [...],
            'vertex_results': [...],
            'correct_ids': [このエリアに属する正解ID],
        },
        'rev03smile': { ... },
        ...
    },
    'revision_content': '...',
    'llm_query': '...',
    'keywords': [...],
}
```

### タスク2: `search_revision_multi_stage`の戻り値を変更

**現在**: 全エリアの結果をまとめたリストを返す
**変更後**: エリアごとの結果を辞書で返す

```python
# 変更後の戻り値
{
    'rev03naibujimu': [検索結果リスト],
    'rev03smile': [検索結果リスト],
    ...
}
```

### タスク3: 正解IDのエリア別フィルタリング

正解IDのフォーマット: `{ボット名}_{Excel行番号}` (例: `smile-bot_129`)

エリアからボット名を抽出し、正解IDをフィルタリング：
```python
def _filter_correct_ids_by_area(self, correct_ids: List[str], area: str) -> List[str]:
    """エリアに属する正解IDのみをフィルタリング"""
    bot_name = self._extract_bot_name_from_area(area)
    return [id for id in correct_ids if id.startswith(f"{bot_name}_")]
```

### タスク4: `_write_summary_sheet`の修正

複数エリアを持つ改定については、エリアごとに行を出力：

**ヘッダー構造（2行ヘッダー）**:
```
| (空)     | (空)   | (空)         | (空)   |     Azure (4列結合)    |     VertexAI (4列結合)    |
|----------|--------|--------------|--------|------------------------|---------------------------|
| 改定番号 | エリア | 改定内容     | 正解数 | 候補数 | 正解発見数 | ... | 候補数 | 正解発見数 | ... |
```

**処理フロー**:
```python
for revision, data in results.items():
    areas = data.get('areas', [])
    for area in areas:
        area_data = data['by_area'][area]
        # エリアごとに1行出力
        worksheet.write(row_num, 0, revision, ...)
        worksheet.write(row_num, 1, area, ...)  # 新規追加
        ...
```

---

## 修正ファイル

| ファイル | 変更箇所 |
|---------|---------|
| `scripts/evaluate_revisions.py` | `search_revision_multi_stage`, `evaluate_revision`, `_write_summary_sheet`, 新規メソッド`_filter_correct_ids_by_area` |

---

## 詳細実装

### 1. `_filter_correct_ids_by_area`メソッド（新規追加）

```python
def _filter_correct_ids_by_area(self, correct_ids: List[str], area: str) -> List[str]:
    """エリアに属する正解IDのみをフィルタリング"""
    bot_name = self._extract_bot_name_from_area(area)
    return [id for id in correct_ids if id.startswith(f"{bot_name}_")]
```

### 2. `search_revision_multi_stage`の修正

戻り値をエリア別辞書に変更：
```python
def search_revision_multi_stage(...) -> Tuple[Dict[str, List[Dict]], str, List[str], str]:
    """
    Returns:
        (エリア別結果辞書, LLM強化クエリ, 抽出キーワード, 検索エリアリスト)
    """
    results_by_area = {}

    for area in areas:
        # 検索実行
        results = orchestrator.execute(...)

        # エリア別に格納
        results_by_area[area] = [
            self._convert_result_to_dict(result, correct_ids, area)
            for result in results
        ]

    return results_by_area, llm_query, keywords, ', '.join(searched_areas)
```

### 3. `evaluate_revision`の修正

エリア別の結果を保持：
```python
evaluation_result = {
    'revision': revision,
    'revision_content': revision_content,
    'areas': [],
    'by_area': {},
    'llm_query': '',
    'keywords': [],
}

# Azure検索
azure_results_by_area, llm_query, keywords, _ = self.search_revision_multi_stage(...)

# VertexAI検索
vertex_results_by_area, _, _, _ = self.search_revision_multi_stage(...)

# エリア別に整理
for area in azure_results_by_area.keys():
    area_correct_ids = self._filter_correct_ids_by_area(correct_ids, area)
    evaluation_result['areas'].append(area)
    evaluation_result['by_area'][area] = {
        'azure_results': azure_results_by_area.get(area, []),
        'vertex_results': vertex_results_by_area.get(area, []),
        'correct_ids': area_correct_ids,
    }
```

### 4. `_write_summary_sheet`の修正

エリアごとに行を出力：
```python
# ヘッダー（エリア列を追加）
headers_row2 = [
    '改定番号', 'エリア', '改定内容', '正解数',
    '候補数', '正解発見数', '正解発見率', '最終正解発見順位',
    '候補数', '正解発見数', '正解発見率', '最終正解発見順位'
]

# データ行
row_num = 2
for revision, data in results.items():
    for area in data['areas']:
        area_data = data['by_area'][area]

        # 指標計算
        azure_metrics = self._calculate_metrics(
            area_data['azure_results'],
            area_data['correct_ids']
        )
        vertex_metrics = self._calculate_metrics(
            area_data['vertex_results'],
            area_data['correct_ids']
        )

        # 行出力
        worksheet.write(row_num, 0, revision, ...)
        worksheet.write(row_num, 1, area, ...)  # エリア
        worksheet.write(row_num, 2, data['revision_content'][:50], ...)
        worksheet.write(row_num, 3, len(area_data['correct_ids']), ...)
        # ... Azure/VertexAI列

        row_num += 1
```

---

## 検証方法

```bash
# 評価スクリプトを実行
python scripts/evaluate_revisions.py

# 出力ファイルを確認
# output/revision_evaluation_{timestamp}.xlsx を開き、以下を確認:
# 1. サマリーシートに2行ヘッダーが表示される
# 2. 事務改定③が4行（エリアごと）に分かれている
# 3. 各エリアの正解数が正しくフィルタリングされている
# 4. 正解発見率が%表示（数値として扱える）
# 5. 全セルに格子線がある
# 6. フォントがMeiryo UI
```
