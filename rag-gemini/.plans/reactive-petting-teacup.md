# UI版 事務改定評価モード バグ修正 + UI刷新計画

## 発見された問題

### 問題1: UNKNOWN-BOT / 正解バッジ表示されない
**根本原因**: rev系シナリオファイルのシート名が全て「Sheet1」
- 検索結果の`Sheet_Name` = "Sheet1"
- `CATEGORY_TO_AREA`辞書のキー = "スマイルタブレット", "内部事務" など
- **マッチしないため `unknown-bot` が返される**
- シナリオID = `unknown-bot_123` → 正解ID（`smile-bot_129`）とマッチしない

**解決策**: エリア名（rev01smile）からボット名を抽出する方式に変更
- 参考実装: `evaluate_revisions.py` の `_extract_bot_name_from_area()`

### 問題2: 20件制限
**原因**: chat.pyでハードコード `[:20]`
**解決策**: UI設定の`top_k`を使用して全件表示

### 問題3: UI整理
**現状**: 業務分野と事務改定評価が混在
**要望**:
- **業務分野検索**: 通常モード（正解なし、実践的）
- **事務改定評価**: 評価モード（正解あり、精度確認）

## 対象ファイル

| ファイル | 変更内容 |
|---------|---------|
| `ui/chat.py` | シナリオID構築修正、表示件数修正、UI刷新 |

## 実装ステップ

### Step 1: シナリオID構築ロジック修正

**問題**: `build_scenario_id()`が`Sheet_Name`からボット名を取得しようとしている

**修正方針**:
1. 両プロバイダー検索時は、検索対象エリア名からボット名を抽出
2. `check_if_correct()`関数にエリア名を渡す

```python
def extract_bot_name_from_area(area: str) -> str:
    """エリア名（rev01smile）からボット名を抽出"""
    area_lower = area.lower()
    for keyword, bot_name in AREA_TO_BOT.items():
        if keyword in area_lower:
            return bot_name
    return "unknown-bot"

def build_scenario_id_from_area(result: Dict, area: str) -> str:
    """エリア名を使用してシナリオIDを構築"""
    row_index = result.get("Row_Index", "")
    if row_index == "":
        return ""
    try:
        excel_row = int(row_index) + 2
        bot_name = extract_bot_name_from_area(area)
        return f"{bot_name}_{excel_row}"
    except (ValueError, TypeError):
        return ""
```

### Step 2: 表示件数制限の撤廃

**変更箇所**: chat.py Line 873, 889付近

```python
# 変更前
for idx, response in enumerate(azure_results[:20], 1):

# 変更後（スクロール可能なコンテナ内で全件表示）
with st.container(height=600):  # 固定高さでスクロール可能
    for idx, response in enumerate(azure_results, 1):
```

### Step 3: UI刷新 - 2モード分離

**新UI設計（タブベース）**:

```
┌─[通常検索]─┬─[事務改定評価]─┐
│                              │
│  【通常検索タブ】              │
│  ・業務分野選択               │
│  ・検索モード選択             │
│  ・候補数設定                │
│                              │
│  【事務改定評価タブ】          │
│  ・改定番号選択（①②③...）   │
│  ・正解ID: N件設定            │
│  ・Azure/VertexAI タブ切替    │
│  ・正解バッジ表示             │
└──────────────────────────────┘
```

**サイドバー構成**:
```
┌─ モード選択 ─────────────────┐
│ ○ 通常検索                   │
│ ● 事務改定評価               │
├──────────────────────────────┤
│ 【通常検索時】               │
│  検索タイプ: ○hybrid ○keyword │
│  検索モード: [▼ original]    │
│  業務分野: [▼ 預金]         │
│  候補数: [3]                │
├──────────────────────────────┤
│ 【事務改定評価時】           │
│  改定番号: [▼ ①]           │
│  → 正解ID: 5件              │
│  → 検索タイプ: 類似検索     │
│  候補数: [100]              │
└──────────────────────────────┘
```

### Step 4: 検索結果へのエリア情報追加

`_search_with_provider()`で結果にエリア情報を追加:

```python
for r in results:
    all_results.append({
        ...
        "_area": area,  # エリア情報を追加
    })
```

タブ表示時にエリア情報を使用してシナリオID構築:
```python
scenario_id = build_scenario_id_from_area(response, response.get("_area", ""))
```

## 検証方法

1. **シナリオID構築テスト**
   - 改定①選択 → 検索実行
   - シナリオIDが `smile-bot_XXX` 形式で表示されることを確認
   - `unknown-bot` が表示されないことを確認

2. **正解バッジテスト**
   - 改定①選択 → 該当クエリ検索
   - 正解IDとマッチする結果に緑バッジ「✓正解」表示

3. **全件表示テスト**
   - 候補数100を設定 → 検索実行
   - 100件全てがスクロールで確認可能

4. **UI切り替えテスト**
   - 通常検索モード ↔ 事務改定評価モード切り替え
   - 各モードで適切なUI要素が表示されること

## 既存コード再利用

| 機能 | 参照元 | 行番号 |
|------|--------|--------|
| エリア名→ボット名抽出 | evaluate_revisions.py | 175-180 |
| 正解フラグ判定 | evaluate_revisions.py | 425 |
