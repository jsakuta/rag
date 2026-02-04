# 多段階OR検索の件数制限修正計画

## 概要
多段階OR検索で原文検索+LLMクエリ検索のOR結合後、最終的に上位N件に絞る機能を追加。

### 現状の問題
- 原文検索でtop_k件、LLMクエリ検索でtop_k件を取得
- OR結合（重複除去）後、そのまま全件返している
- **期待**: OR結合後に上位top_k件に絞るべき

### UI版の問題
- 多段階検索時は閾値（threshold）は使用しない（top_kモードのみ）
- 閾値スライダーが不要、代わりに候補数を指定するUIが必要

## 対象ファイル

| ファイル | 変更内容 |
|---------|---------|
| `src/core/search/multi_stage_orchestrator.py` | OR結合後にtop_k件に絞る |
| `ui/chat.py` | 多段階検索時のUI調整 |

## 実装ステップ

### Step 1: MultiStageOrchestrator修正
**ファイル**: `src/core/search/multi_stage_orchestrator.py`

`_merge_results`メソッドの最後で、OR結合・ソート後にtop_k件に絞る:

```python
def _merge_results(...) -> List[MultiStageSearchResultDict]:
    # ... 既存のOR結合処理 ...

    # スコアでソート
    merged_results.sort(key=lambda x: x[SearchResultKeys.SIMILARITY], reverse=True)

    # 【追加】TOP-Kモードの場合は上位K件に絞る
    if self.filter_mode == "top_k":
        merged_results = merged_results[:self.top_k]

    return merged_results
```

**変更箇所**: Line 330-332付近

### Step 2: UI版の多段階検索時のUI調整
**ファイル**: `ui/chat.py`

#### 2-1: サイドバーの条件分岐修正
多段階検索（multi_stage）選択時:
- 閾値スライダーを非表示
- 候補数入力を表示

```python
# 多段階検索パラメータ（multi_stage時のみ表示）
if selected_mode == "multi_stage":
    # 候補数入力（top_k）
    st.session_state.config.top_k = st.number_input(
        "候補数", min_value=10, max_value=200,
        value=st.session_state.config.top_k, step=10
    )
    # LLM判断支援チェックボックスは維持
    st.session_state.config.multi_stage_enable_judgment_support = st.checkbox(
        "LLM判断支援",
        value=st.session_state.config.multi_stage_enable_judgment_support
    )
```

**変更箇所**: Line 760-769付近（現在のmulti_stageパラメータ部分）

#### 2-2: 両プロバイダー検索時の候補数パラメータ適用
**ファイル**: `ui/chat.py` `_search_with_provider`関数

オーケストレーター作成時に`top_k`パラメータを使用:
```python
orchestrator = MultiStageOrchestrator(
    ...
    filter_mode="top_k",  # 常にtop_kモード
    top_k=st.session_state.config.top_k,  # UI設定の候補数を使用
)
```

**変更箇所**: Line 455-466付近

## サイドバーUI設計（変更後）

```
┌─ 設定 ─────────────────────┐
│ 検索タイプ: ○類似検索 ○キーワード必須 │
│ 検索バランス: [====|====]      │  ← hybrid時のみ
│ 検索モード: [▼ multi_stage]   │
│ 検索対象: [▼ シナリオ+FAQ]    │  ← hybrid時のみ
│ 候補数: [100]                 │  ← multi_stage時のみ
│ ☑ LLM判断支援                │  ← multi_stage時のみ
│ 業務分野: [▼ 預金]           │
│ ──────────────────────────── │
│ 事務改定評価                   │
│ 改定番号: [▼ なし]            │
└─────────────────────────────┘
```

## 検証方法

1. **バッチ版の動作確認**
   ```bash
   python scripts/evaluate_revisions.py
   ```
   - top_k=100設定時、各エリアの結果が最大100件になることを確認

2. **UI版の動作確認**
   ```bash
   python main.py interactive
   ```
   - 検索モード「多段階OR検索」選択時:
     - 閾値スライダーが非表示
     - 候補数入力が表示される
   - 候補数を50に設定 → 検索結果が最大50件

## 影響範囲

- `MultiStageOrchestrator`: 全ての利用箇所で動作が変わる
  - `scripts/evaluate_revisions.py`: バッチ評価
  - `ui/chat.py`: UI検索（両プロバイダーモード）
  - `src/core/search_engine.py`: 通常検索（multi_stageモード時）
