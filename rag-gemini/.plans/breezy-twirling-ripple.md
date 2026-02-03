# 改定番号別検索タイプ設定の実装計画

## 概要
事務改定評価スクリプト（evaluate_revisions.py）で、改定番号ごとに検索タイプ（類似検索/キーワード必須）とベクトル重みを個別に設定できるようにする。

## 現状
- `settings.yaml`の`revision_areas`で改定番号ごとに`vector_weight`は設定可能
- 検索タイプ（`search_type`）は未対応
- ⑤⑥は`vector_weight: 0.0`でキーワード重視になっているが、キーワードフィルタ（必須）とは異なる

## 方針
- **キーワード必須検索**: 元のシナリオExcelを直接キーワード検索（ベクトルDB不使用）
- **類似検索**: 従来通りベクトルDB（Azure/VertexAI）を使用

---

## 設定ファイル変更

### config/settings.yaml
```yaml
revision_areas:
  "①":
    areas:
      - rev01smile
    search_type: hybrid        # 類似検索（意味で探す）
    vector_weight: 0.9
  "⑤":
    areas:
      - rev05smile
    search_type: keyword_filter  # キーワード必須（Excel直接検索）
  "⑥":
    areas:
      - rev06smile
    search_type: keyword_filter
```

---

## コード変更

### 1. scripts/evaluate_revisions.py

#### 変更箇所1: 設定読み込み部分（L76-88）
```python
REVISION_SEARCH_TYPES = {}
for rev, config in _raw_revision_areas.items():
    if isinstance(config, dict):
        REVISION_SEARCH_TYPES[rev] = config.get("search_type", "hybrid")
```

#### 変更箇所2: シナリオExcel読み込みメソッド追加

**ファイル名パターン**: `{area}_シナリオデータ_{日付}.xlsx`
- 例: `rev05smile_シナリオデータ_20260203.xlsx`

```python
SCENARIO_DIR = PROJECT_ROOT / "reference" / "scenario"

def _load_scenario_excel(self, area: str) -> pd.DataFrame:
    """シナリオExcelを読み込み"""
    pattern = f"{area}_シナリオデータ_*.xlsx"
    files = list(SCENARIO_DIR.glob(pattern))
    if not files:
        logger.warning(f"シナリオファイルが見つかりません: {pattern}")
        return pd.DataFrame()
    # 最新ファイルを使用
    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    logger.info(f"シナリオExcel読み込み: {latest_file.name}")
    return pd.read_excel(latest_file)
```

#### 変更箇所3: キーワードフィルタ検索メソッド追加
```python
def _execute_keyword_filter_search(
    self, revision: str, query: str, correct_ids: List[str]
) -> Tuple[Dict[str, List[Dict]], str, List[str], List[str]]:
    """キーワード必須検索（Excel直接）"""
    areas = REVISION_TO_AREAS.get(revision, [])

    # キーワード抽出
    keyword_engine = KeywordSearchEngine(...)
    keywords = keyword_engine.extract_keywords(query)

    results_by_area = {}
    for area in areas:
        # シナリオExcel読み込み
        df = self._load_scenario_excel(area)

        # 各行に対してキーワードマッチング
        matched = []
        for idx, row in df.iterrows():
            text = f"{row['Lv1']} {row['Lv2']} ... {row['回答']}"
            match_count = sum(1 for kw in keywords if kw in text)
            if match_count > 0:
                matched.append((idx, row, match_count))

        # マッチ数順でソート
        matched.sort(key=lambda x: -x[2])

        # 結果をフォーマット
        results_by_area[area] = [self._format_excel_result(r, correct_ids, area) for r in matched]

    return results_by_area, "", keywords, list(areas)
```

#### 変更箇所4: search_revision_multi_stage メソッド分岐（L277）
```python
def search_revision_multi_stage(self, revision, query, correct_ids, provider):
    search_type = REVISION_SEARCH_TYPES.get(revision, "hybrid")

    if search_type == "keyword_filter":
        # キーワード必須検索（Excel直接）
        return self._execute_keyword_filter_search(revision, query, correct_ids)
    else:
        # 類似検索（従来通り）
        # 既存のコード...
```

#### 変更箇所5: 評価設定表示（L894-910）
- 改定番号別の検索タイプを表示

---

## 出力フォーマット変更

### キーワード必須検索時
- Azure/VertexAIの列は「-」または空欄（Excel直接検索のため）
- または「Excel検索」列を追加して結果を表示
- 詳細シートに「検索タイプ: キーワード必須」を表示

---

## 実装ファイル一覧

| ファイル | 変更内容 |
|---------|---------|
| config/settings.yaml | revision_areasに`search_type`を追加 |
| scripts/evaluate_revisions.py | キーワードフィルタ検索の分岐処理、Excel読み込み追加 |

---

## 検証方法
1. `python scripts/evaluate_revisions.py`を実行
2. 出力Excelで以下を確認:
   - ①〜④: 類似検索（Azure/VertexAI両方で検索）
   - ⑤⑥: キーワード必須（Excel直接検索、Azure/VertexAI列は空）
3. ログで検索タイプが正しく表示されること
