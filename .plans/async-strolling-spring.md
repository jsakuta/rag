# キャッシュクリアボタン削除

## Context

`eval_ui.py` のサイドバーに「キャッシュクリア」ボタンがある。`st.cache_resource.clear()` を呼ぶが、ChromaDB の内部シングルトンレジストリはクリアされないため、2回目の検索で `An instance of Chroma already exists ... with different settings` エラーが発生する。

DB や設定を変更した場合は Streamlit を再起動すれば済むので、このボタンは不要。削除して TROUBLESHOOTING.md に「DB変更後は Streamlit 再起動」の手順を明記する。

## 変更箇所

### 1. `rag-local/apps/revision-eval/ui/eval_ui.py` — ボタン削除

**883-885行目を削除:**
```python
        if st.button("キャッシュクリア", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()
```

回答支援AI (`apps/answer-support/ui/chat.py`) にはキャッシュクリアボタンは存在しない（grep 確認済み）。

### 2. `rag-local/docs/TROUBLESHOOTING.md` — 手順追記

「データベース関連」セクションに新項目を追加:

```
### ChromaDB エラー: An instance of Chroma already exists with different settings

**症状:**
DB再構築後や設定変更後に検索結果が0件になる、または上記エラーが表示される。

**原因:**
ChromaDBはプロセス内で同一パスに1つのクライアントしか許可しない。
DB再構築後も古いクライアントインスタンスがメモリに残っている。

**解決策:**
Streamlit を再起動する（Ctrl+C → 再度 `streamlit run ...`）。
UIにキャッシュクリア機能はないため、DB・設定変更後は必ず再起動が必要。
```

## 検証

- `eval_ui.py` のサイドバーに「キャッシュクリア」ボタンが表示されないことを確認
- Streamlit 再起動後に検索が正常動作することを確認
