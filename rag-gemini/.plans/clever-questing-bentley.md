# 詳細シート分割 & 改定番号別ベクトル重み設定

## 概要
1. 詳細シートを事務改定③のようにエリアごとに分割
2. 改定番号ごとにベクトル重みを設定可能にする（⑤⑥はキーワード検索重視）

---

## 変更対象ファイル

| ファイル | 変更内容 |
|---------|---------|
| `config/settings.yaml` | 改定番号別`vector_weight`設定を追加 |
| `scripts/evaluate_revisions.py` | 詳細シート分割、改定番号別vector_weight適用 |

---

## 実装計画

### 1. settings.yaml の変更

**現状**:
```yaml
revision_areas:
  "①":
    - rev01smile
  "③":
    - rev03naibujimu
    - rev03smile
    ...
```

**変更後**:
```yaml
revision_areas:
  "①":
    areas:
      - rev01smile
    vector_weight: 0.9  # デフォルト（ベクトル重視）
  "②":
    areas:
      - rev02souzoku
    vector_weight: 0.9
  "③":
    areas:
      - rev03naibujimu
      - rev03smile
      - rev03souzoku
      - rev03torikaku
    vector_weight: 0.9
  "④":
    areas:
      - rev04naibujimu
    vector_weight: 0.9
  "⑤":
    areas:
      - rev05smile
    vector_weight: 0.3  # キーワード重視
  "⑥":
    areas:
      - rev06smile
    vector_weight: 0.3  # キーワード重視
```

---

### 2. evaluate_revisions.py の変更

#### 2.1 設定読み込み部分（行67-75付近）

**現状**:
```python
REVISION_TO_AREAS = _settings["revision_areas"]
VECTOR_WEIGHT = _settings["vector_weight"]
```

**変更後**:
```python
# 新しい形式に対応（areas/vector_weightを含む辞書）
_raw_revision_areas = _settings["revision_areas"]
REVISION_TO_AREAS = {}
REVISION_VECTOR_WEIGHTS = {}
DEFAULT_VECTOR_WEIGHT = _settings["vector_weight"]

for rev, config in _raw_revision_areas.items():
    if isinstance(config, dict):
        REVISION_TO_AREAS[rev] = config.get("areas", [])
        REVISION_VECTOR_WEIGHTS[rev] = config.get("vector_weight", DEFAULT_VECTOR_WEIGHT)
    else:
        # 旧形式（リスト直接指定）への後方互換性
        REVISION_TO_AREAS[rev] = config
        REVISION_VECTOR_WEIGHTS[rev] = DEFAULT_VECTOR_WEIGHT
```

#### 2.2 詳細シート分割（`_write_detail_sheet`メソッド、行615-672）

**現状**: 複数エリアの結果を統合して1シートに出力

**変更後**: エリアごとに別シートを作成

```python
def _write_detail_sheets(
    self,
    writer: pd.ExcelWriter,
    revision: str,
    data: Dict[str, Any],
    formats: Dict[str, Any],
) -> None:
    """複数エリアの場合、エリアごとに詳細シートを作成"""
    areas = data.get("areas", [])

    if len(areas) <= 1:
        # 単一エリアの場合は従来通り
        self._write_single_detail_sheet(writer, revision, data, formats)
    else:
        # 複数エリアの場合はエリアごとにシートを作成
        for area in areas:
            area_short = area.replace("rev", "").replace(revision.replace("③", "03"), "")
            sheet_name = f"{revision}_{area_short}"  # 例: ③_naibujimu

            area_data = {
                "revision_content": data["revision_content"],
                "correct_ids": self._filter_correct_ids_by_area(data["correct_ids"], area),
                "llm_query": data.get("llm_query", ""),
                "keywords": data.get("keywords", []),
                "areas": [area],
                "by_area": {area: data.get("by_area", {}).get(area, {})},
            }
            self._write_single_detail_sheet(writer, sheet_name, area_data, formats)
```

#### 2.3 検索時のvector_weight適用（`_create_orchestrator`メソッド）

**現状** (行169):
```python
vector_weight=VECTOR_WEIGHT,
```

**変更後**:
`search_revision_multi_stage`メソッドでrevision引数からvector_weightを取得し、orchestratorに渡す

```python
def _create_orchestrator(
    self,
    provider: str,
    area: str,
    reference_queries: List[str],
    vector_weight: float,  # 引数追加
) -> Optional[MultiStageOrchestrator]:
    ...
    return MultiStageOrchestrator(
        ...
        vector_weight=vector_weight,  # 改定番号別の値を使用
        ...
    )

def search_revision_multi_stage(
    self, revision: str, query: str, correct_ids: List[str], provider: str
) -> Tuple[Dict[str, List[Dict]], str, List[str], List[str]]:
    ...
    vector_weight = REVISION_VECTOR_WEIGHTS.get(revision, DEFAULT_VECTOR_WEIGHT)
    ...
    orchestrator = self._create_orchestrator(provider, area, reference_queries, vector_weight)
```

---

## 検証方法

1. **設定確認**:
   ```bash
   cd C:\VSCode\rag\rag-gemini
   python -c "from config import load_settings; s=load_settings('evaluation'); print(s['revision_areas'])"
   ```

2. **評価スクリプト実行**:
   ```bash
   python scripts/evaluate_revisions.py
   ```

3. **出力Excel確認**:
   - サマリーシート: ③が4行に分かれていること
   - 詳細シート: ③が4シート（③_naibujimu, ③_smile, ③_souzoku, ③_torikaku）に分かれていること
   - ベクトル重み列: ①②③④は0.9、⑤⑥は0.3が表示されること

---

## 実装順序

1. `config/settings.yaml` を更新（新形式に変更）
2. `evaluate_revisions.py` の設定読み込み部分を更新
3. `_create_orchestrator`にvector_weight引数追加
4. `search_revision_multi_stage`でrevision別vector_weightを渡す
5. `_write_detail_sheet`を`_write_detail_sheets`に改名・分割対応
6. 動作確認
