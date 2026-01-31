# TOP-K モードで候補数が少ない問題の調査結果

## 問題の概要
TOP-K=130 に設定しているのに、一部の改定で候補数が130件より少ない。
- ① Azure: 43件
- ⑤ Azure: 2件
- ⑥ Azure: 18件

## 原因特定: 全角スペースによる設定ミス

**`config/settings.yaml` 122行目**:
```yaml
filter_mode: top_k　  # ← 末尾に全角スペース（U+3000）がある！
```

この**全角スペース**により：
1. `filter_mode` の実際の値 = `"top_k　"` （末尾に全角スペース）
2. コードの比較 `if self.filter_mode == "top_k":` が **False** になる
3. **else ブロック（閾値モード）が実行される**

### 閾値モードの動作
```python
# multi_stage_orchestrator.py:198-203
if self.filter_mode == "top_k":
    return all_results[:self.top_k]  # ← 実行されない
else:
    # 閾値モード（実際に実行される）
    return [r for r in all_results if r[SearchResultKeys.SIMILARITY] >= self.threshold]
```

適用される閾値（settings.yaml:132-134）:
- **Azure: 0.40** 以上のみ → ⑤⑥で候補数が少ない原因
- **VertexAI: 0.50** 以上のみ

## 解決策

### 修正内容
`config/settings.yaml` 122行目の全角スペースを削除:

```yaml
# 修正前
filter_mode: top_k　  # 末尾に全角スペース

# 修正後
filter_mode: top_k  # 末尾に半角スペースなし、または半角スペースのみ
```

### 修正対象ファイル
- `config/settings.yaml:122`

## 修正後の期待される結果

TOP-K=130 が正しく適用され、全ての改定で候補数が最大130件（または DB内のドキュメント数）になる:
- ① Azure: ~130件
- ⑤ Azure: ~130件
- ⑥ Azure: ~130件

## 検証方法

1. 設定ファイルを修正
2. スクリプトを再実行: `python scripts/evaluate_revisions.py`
3. 出力を確認し、候補数が130件前後になることを確認
