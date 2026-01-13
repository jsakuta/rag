# コンセンサスレポート: UI Multi-Stage Search Support

## レビュー完了状況

| レビュアー | 状態 | 指摘数 |
|------------|------|--------|
| Claude CLI | ✅ 完了 | 5件 |
| Claude Agent | ✅ 完了 | 8件 |
| Codex CLI | ❌ TTY問題 | - |

## 合意した問題点

### 3/3 一致 → 必須修正

該当なし（Codex CLIが動作しなかったため）

### 2/2 一致（Claude系）→ 推奨修正

| # | Severity | 問題 | 修正方針 |
|---|----------|------|---------|
| 1 | **HIGH** | XSS脆弱性: ユーザー入力がHTMLエスケープされていない | `html.escape()` を使用 |
| 2 | Medium | multi_stage_threshold/enable_llm_analysisがSearchConfigに存在しない場合AttributeError | `getattr()` でデフォルト値付きアクセス |
| 3 | Medium | search_mode変更時のユーザーフィードバックがない | `st.info()` で通知追加 |
| 4 | Low | category_colors/labelsが関数内で毎回定義される | モジュールレベル定数化 |

### 1/2 のみ → 参考情報

| # | Severity | 問題 | 対応 |
|---|----------|------|------|
| 5 | Medium | save_chat_history()でProcessor毎回生成 | session_state.processor再利用 |
| 6 | Medium | エラーメッセージに内部情報が漏洩 | 汎用エラーメッセージに変更 |
| 7 | Low | configの直接変更がバリデーションをバイパス | 後日検討 |

## 修正対象（優先度順）

### 1. XSS対策（HIGH）

```python
import html

def format_response_card(number, similarity, query, answer, category=None):
    query = html.escape(query)
    answer = html.escape(answer)
    # ...
```

### 2. 属性アクセス安全化（Medium）

```python
if selected_mode == "multi_stage":
    threshold = getattr(st.session_state.config, 'multi_stage_threshold', 0.5)
    st.session_state.config.multi_stage_threshold = st.slider("しきい値", 0.0, 1.0, threshold, 0.05)
```

## 結論

**XSS脆弱性が最重要**。これはセキュリティ問題のため即時修正が必要。

その他の問題は機能に影響しない改善点であり、後日対応可能。
