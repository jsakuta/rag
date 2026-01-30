# evaluate_revisions.py 機能拡張計画

## 概要

事務改定評価スクリプトに以下の3つの機能を追加する。

---

## 要件1: 日本語指標の追加

### 追加する指標

| 指標名 | 説明 | 計算方法 |
|--------|------|---------|
| **候補数** | 検索結果の総件数 | `len(results)` |
| **正解発見数** | 見つけた正解の件数 | 正解フラグ=TRUEの件数 |
| **正解発見率** | 正解ID数のうち何件見つけたか | `正解発見数 / 正解ID数 × 100%` |
| **最終正解の発見順位** | 最後の正解が何番目に出てきたか | 類似度順で最後の正解の順位 |

### 修正ファイル

**`scripts/evaluate_revisions.py`**

1. **`_calculate_metrics` メソッド追加**（新規）
   - 検索結果リストから上記指標を計算
   - 正解フラグがTRUEの行の最大順位を取得

2. **`_write_summary_sheet` メソッド修正**（行566-613）
   - サマリーシートに新指標列を追加
   - 列: `Azure_候補数`, `Azure_正解発見数`, `Azure_正解発見率`, `Azure_最終正解発見順位`
   - 同様に `VertexAI_` の列も追加

### 計算ロジック

```python
def _calculate_metrics(self, results: List[Dict], correct_ids: List[str]) -> Dict:
    """指標を計算"""
    candidate_count = len(results)
    found_correct_count = sum(1 for r in results if r.get('正解フラグ') == 'TRUE')

    # 正解発見率
    total_correct = len(correct_ids)
    discovery_rate = (found_correct_count / total_correct * 100) if total_correct > 0 else 0

    # 最終正解の発見順位（結果は類似度降順でソート済み前提）
    last_correct_rank = 0
    for i, r in enumerate(results, start=1):
        if r.get('正解フラグ') == 'TRUE':
            last_correct_rank = i

    return {
        '候補数': candidate_count,
        '正解発見数': found_correct_count,
        '正解発見率': f"{discovery_rate:.1f}%",
        '最終正解発見順位': last_correct_rank if last_correct_rank > 0 else '-'
    }
```

---

## 要件2: プロバイダー別閾値設定

### 設定値

| プロバイダー | 閾値 |
|-------------|------|
| Azure OpenAI | 0.40 |
| VertexAI | 0.50 |

### 修正ファイル

**`scripts/evaluate_revisions.py`**

1. **定数定義の変更**（行74-77）

```python
# 変更前
THRESHOLD = 0.45
VECTOR_WEIGHT = 0.9

# 変更後
THRESHOLD_BY_PROVIDER = {
    'azure_openai': 0.40,
    'vertex_ai': 0.50,
}
VECTOR_WEIGHT = 0.9
MAX_RESULTS = 100
```

2. **`_create_orchestrator` メソッド修正**（行197-205）

```python
# プロバイダー別の閾値を取得
threshold = THRESHOLD_BY_PROVIDER.get(provider, 0.45)

orchestrator = MultiStageOrchestrator(
    ...
    threshold=threshold,  # ← プロバイダー別
    ...
)
```

3. **`main` 関数修正**（行781-787）
   - `SearchConfig` 生成時の `multi_stage_threshold` をデフォルト値に変更

---

## 要件3: 修正案が出力されない問題の修正

### 原因

プロンプト `prompt/judgment_support.txt` の行13, 28で：
```
修正案: <関連ありの場合は具体的な修正提案、それ以外は「-」>
```
と指示しているため、「要確認」判定時はLLMが「-」を返している。

### 修正ファイル

**`prompt/judgment_support.txt`**

### 修正内容

行13を変更：
```
# 変更前
修正案: <関連ありの場合は具体的な修正提案、それ以外は「-」>

# 変更後
修正案: <関連あり/要確認の場合は具体的な修正提案や確認ポイント、明らかに無関係の場合は「-」>
```

行25-28（出力例2）を変更：
```
# 変更前
【出力例2: 要確認】
関連性: 要確認
根拠: 改定対象の商品カテゴリに部分的に該当する可能性がある
修正案: -

# 変更後
【出力例2: 要確認】
関連性: 要確認
根拠: 改定対象の商品カテゴリに部分的に該当する可能性がある
修正案: 商品カテゴリの適用範囲を確認し、該当する場合は関連箇所の文言を更新する
```

---

## 修正ファイル一覧

| ファイル | 修正内容 |
|---------|---------|
| `scripts/evaluate_revisions.py` | 日本語指標追加、プロバイダー別閾値 |
| `prompt/judgment_support.txt` | 修正案の出力条件変更 |

---

## 検証方法

1. **スクリプト実行**
   ```bash
   python scripts/evaluate_revisions.py
   ```

2. **出力ファイル確認** (`output/revision_evaluation_*.xlsx`)
   - サマリーシートに新指標が追加されていること
   - Azure閾値0.40、VertexAI閾値0.50で結果が変わっていること
   - 「修正案」列に「-」以外の内容が出力されていること

3. **検証ポイント**
   - ③の改定で、Azureの最終正解発見順位 < VertexAIの最終正解発見順位 となるか確認
   - 「要確認」判定時に修正案が提示されているか確認
