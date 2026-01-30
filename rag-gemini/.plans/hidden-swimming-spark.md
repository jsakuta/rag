# 検索設定ハードコード除去計画

## 問題点

`scripts/evaluate_revisions.py`でYAMLから設定を読み込むように変更されたが、`.get()`のデフォルト値としてハードコードが残っている。

### 発見された問題箇所

| 行 | 変数名 | 問題 |
|----|--------|------|
| 61-68 | `REVISION_TO_AREAS` | フォールバック値がハードコード |
| 70-77 | `AREA_TO_BOT` | フォールバック値がハードコード |
| 88-91 | `THRESHOLD_BY_PROVIDER` | **YAML: 0.40 vs コード: 0.30（不整合）** |
| 92 | `VECTOR_WEIGHT` | フォールバック値0.9（YAMLと一致） |
| 93 | `MAX_RESULTS` | フォールバック値100（YAMLと一致） |
| 1051 | `SearchConfig.multi_stage_threshold` | `0.45`がハードコード（使用されていないがコメントと矛盾） |

### 特に重要な不整合

```python
# コード（行88-91）
THRESHOLD_BY_PROVIDER = _eval_settings.get("thresholds", {
    'azure_openai': 0.30,  # ← ハードコード
    'vertex_ai': 0.50,
})

# YAML（config/settings.yaml 行116-118）
thresholds:
  azure_openai: 0.40  # ← 実際の設定値
  vertex_ai: 0.50
```

## 修正計画

### 修正ファイル
- [evaluate_revisions.py](scripts/evaluate_revisions.py)

### 修正内容

1. **フォールバック値をYAMLと同期**
   - `THRESHOLD_BY_PROVIDER`: `azure_openai: 0.30` → `0.40`

2. **または、フォールバック値を削除してYAML必須化**
   - `.get(key, default)` → `[key]`（YAMLが必須）
   - 設定がない場合は明確なエラーメッセージを表示

3. **SearchConfig初期化の整理**
   - `multi_stage_threshold=0.45`のコメントを修正、またはYAMLから値を取得

## 推奨アプローチ

フォールバック値をYAMLの値と一致させる（方式B）:
- YAMLファイルがない環境でも動作を維持
- コードとYAMLの値を同期することで整合性を保証

## 検証方法

```bash
cd c:\VSCode\rag\rag-gemini
python scripts/evaluate_revisions.py
```

実行時に表示される設定値を確認:
- 閾値 (Azure): 0.40
- 閾値 (VertexAI): 0.50
