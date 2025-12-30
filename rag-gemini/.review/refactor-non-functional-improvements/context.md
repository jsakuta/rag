# レビューコンテキスト

## 変更の背景・目的

rag-geminiプロジェクトの非機能改善（Phase 1-2）を実施。
機能は一切変更せず、コード品質・保守性・可読性の向上のみ。

## 実施した変更

### Phase 1: 即時対応（クリティカル）
1. **デッドコード削除** (searcher.py)
   - 未使用インポート削除: `ChatVertexAI`
   - 未使用変数削除: `reference_vectors`, `reference_texts`, `reference_df`
   - 未使用メソッド削除: `_is_data_unchanged`, `_check_file_timestamps`, `_get_file_timestamps`, `_save_cache_info`

2. **ハードコード修正** (searcher.py)
   - Vertex AI設定を `config` 参照に変更

3. **エラーハンドリング改善** (dynamic_db_manager.py)
   - 裸の `except:` を `except ValueError:` に変更

### Phase 2: 短期対応
1. **auth.py新規作成**
   - Google認証処理の共通化
   - `get_google_credentials()`, `initialize_vertex_ai()` を提供

2. **コード重複解消** (input_handler.py)
   - `_get_column_names()` を基底クラスに移動
   - `_build_combined_text()` を基底クラスに追加
   - サブクラスの重複メソッド削除

3. **マジックナンバー定数化** (config.py)
   - `EMBEDDING_BATCH_SIZE`, `VECTOR_DB_BATCH_SIZE`
   - `VECTOR_SEARCH_MULTIPLIER`, `POSITION_WEIGHT`
   - `QUERY_COLUMN_CANDIDATES`, `ANSWER_COLUMN_CANDIDATES`, `TAG_COLUMN_CANDIDATES`
   - `PRINCIPLE_MARKER`

4. **ロギング最適化** (logger.py)
   - 環境変数 `LOG_LEVEL` でログレベル制御
   - ハンドラ重複追加防止

5. **型ヒント統一** (searcher.py)
   - `list[str]` → `List[str]` (Python 3.7互換)

## 変更ファイル

| ファイル | 変更行数 | 内容 |
|----------|----------|------|
| searcher.py | -155 | デッドコード削除、型ヒント修正 |
| input_handler.py | +45/-55 | 重複解消、基底クラス拡張 |
| auth.py | +42 | 新規作成 |
| config.py | +18 | 定数追加 |
| logger.py | +10/-13 | ログレベル制御 |
| dynamic_db_manager.py | +2/-2 | エラーハンドリング |
| gemini_embedding.py | +5/-16 | 認証処理委譲 |

## 未実施（Phase 3）

- search()メソッド分割
- 依存性注入導入
- バックアップファイル整理
