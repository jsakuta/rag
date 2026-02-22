# トラブルシューティング

このドキュメントでは、RAG-Localシステムでよく発生する問題と解決方法を説明します。

## 目次

- [認証関連](#認証関連)
- [データベース関連](#データベース関連)
- [検索・処理関連](#検索処理関連)
- [API関連](#api関連)
- [UI関連](#ui関連)
- [パフォーマンス関連](#パフォーマンス関連)

---

## 認証関連

### Gemini 認証エラー

**症状:**
```
google.auth.exceptions.DefaultCredentialsError: Could not automatically determine credentials.
```

**原因:**
- 認証ファイル `gemini_credentials.json` が見つからない
- 環境変数が正しく設定されていない

**解決策:**

1. 認証ファイルの確認:
```bash
# ファイルが存在するか確認
ls -l C:\VSCode\rag\rag-local\gemini_credentials.json
```

2. `.env` ファイルの確認:
```env
GEMINI_CREDENTIALS_PATH=gemini_credentials.json
GEMINI_PROJECT_ID=your-project-id
GEMINI_LOCATION=us-central1
```

3. サービスアカウントの権限確認:
   - `Vertex AI User`
   - `AI Platform Admin`

詳細は [docs/GOOGLE_CLOUD_AUTH.md](./GOOGLE_CLOUD_AUTH.md) を参照してください。

### Azure OpenAI 認証エラー

**症状:**
```
openai.error.AuthenticationError: Incorrect API key provided
```

**原因:**
- API キーが無効
- エンドポイントURLが間違っている

**解決策:**

1. `.env` ファイルの確認:
```env
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large
```

2. Azure Portal で API キーを確認

3. リソース名とエンドポイントの一致を確認

---

## データベース関連

### ChromaDB エラー: コレクションが見つからない

**症状:**
```
ValueError: Collection 'deposit_DB' does not exist.
```

**原因:**
- ベクトルDBが初期化されていない
- DBファイルが破損している

**解決策:**

1. DBの再構築:
```bash
# 既存DBを削除（Streamlit停止後）
rm -rf data/vector_db/

# データ処理を実行（自動的にDB再構築）
python apps/answer-support/main.py
```

2. 特定のDBのみ再構築:
```bash
python scripts/rebuild_before_scenario_db.py
```

### ChromaDB エラー: Device or resource busy

**症状:**
```
rm: cannot remove 'data/vector_db/...': Device or resource busy
```

**原因:**
- Streamlit UI や他のPythonプロセスがDBを使用中

**解決策:**

1. 全てのPythonプロセスを停止:
```bash
# Windows
taskkill /F /IM python.exe

# Linux/Mac
pkill -f python
```

2. Streamlit UIを停止:
```bash
# Ctrl+C または
pkill -f streamlit
```

3. DBファイルのロック解除後に再実行

### タイムスタンプ検証エラー

**症状:**
- データを更新しても検索結果が変わらない

**原因:**
- タイムスタンプファイルが古い状態

**解決策:**

1. タイムスタンプファイルを削除:
```bash
rm data/vector_db/update_timestamps.json
```

2. データ処理を再実行:
```bash
python apps/answer-support/main.py
```

---

## 検索・処理関連

### メモリエラー

**症状:**
```
MemoryError: Unable to allocate array
```

**原因:**
- 大量のデータを一度に処理している
- システムメモリ不足

**解決策:**

1. バッチサイズを縮小:
```python
# src/utils/gemini_embedding.py
BATCH_SIZE = 5  # デフォルト値から減らす
```

2. システムメモリを増強（推奨: 16GB以上）

3. 処理データを分割:
```bash
# データを複数ファイルに分割して処理
python apps/answer-support/main.py --input input1.xlsx
python apps/answer-support/main.py --input input2.xlsx
```

### コレクション名エラー

**症状:**
```
ValueError: Collection name contains invalid characters
```

**原因:**
- 日本語文字がコレクション名に含まれている（過去の問題）

**解決策:**

現在は自動変換で対応済み。問題が発生した場合:

1. `src/utils/dynamic_db_manager.py` の変換マップを確認:
```python
COLLECTION_NAME_MAP = {
    "預金": "deposit_DB",
    "融資": "loan_DB",
    ...
}
```

2. 新しい業務領域を追加する場合は変換マップに登録

### キーワード抽出が空

**症状:**
```
WARNING - No keywords extracted from query
```

**原因:**
- 入力文字列が短すぎる
- Sudachi辞書が正しくインストールされていない

**解決策:**

1. Sudachi辞書の再インストール:
```bash
pip install --upgrade sudachipy sudachidict-core
```

2. 入力文字列の確認（最低5文字以上推奨）

---

## API関連

### API レート制限エラー

**症状:**
```
google.api_core.exceptions.ResourceExhausted: 429 Quota exceeded
```

**原因:**
- Gemini API の使用量制限に到達

**解決策:**

1. リトライ処理を待つ（自動リトライ機能あり）

2. バッチサイズを縮小:
```python
# config.py
BATCH_SIZE = 3  # 5 から減らす
```

3. Google Cloud Console でクォータ引き上げを申請

### LLM タイムアウト

**症状:**
```
TimeoutError: Request timed out
```

**原因:**
- LLM レスポンスが遅い
- ネットワーク問題

**解決策:**

1. タイムアウト値を増やす:
```python
# config.py
LLM_TIMEOUT = 60  # 30から増やす
```

2. LLM分析を無効化:
```bash
ENABLE_LLM_ANALYSIS=false python apps/revision-eval/evaluate_revisions.py
```

3. ネットワーク接続を確認

---

## UI関連

### Streamlit 起動エラー

**症状:**
```
ModuleNotFoundError: No module named 'streamlit'
```

**原因:**
- Streamlit がインストールされていない

**解決策:**

```bash
pip install streamlit
```

### ポート衝突エラー

**症状:**
```
OSError: [Errno 48] Address already in use
```

**原因:**
- ポート 8501 が既に使用されている

**解決策:**

1. 既存のプロセスを停止:
```bash
# Windows
netstat -ano | findstr :8501
taskkill /PID <pid> /F

# Linux/Mac
lsof -ti:8501 | xargs kill -9
```

2. 別のポートを使用:
```bash
streamlit run apps/answer-support/ui/chat.py --server.port 8502
```

---

## パフォーマンス関連

### 初回処理が遅い

**症状:**
- 初回実行時に5-10分かかる

**原因:**
- ベクトル化処理（正常な動作）

**解決策:**

1. 初回のみ時間がかかります（2回目以降は数秒）
2. タイムスタンプ検証により、不要な再ベクトル化を回避

### 検索が遅い

**症状:**
- 各クエリに数秒かかる

**原因:**
- LLM クエリ拡張が有効（API呼び出し）

**解決策:**

1. 原文検索モードに変更:
```python
# config.py
DEFAULT_SEARCH_MODE = "original"
DEFAULT_ENABLE_QUERY_ENHANCEMENT = False
```

2. キーワード重みを下げる:
```python
# config.py
KEYWORD_WEIGHT = 0.0  # キーワード検索を無効化
```

### ディスク容量不足

**症状:**
```
OSError: [Errno 28] No space left on device
```

**原因:**
- ベクトルDBが大量のディスク容量を使用

**解決策:**

1. 不要なDBを削除:
```bash
# 旧バージョンのDBを削除
rm -rf data/vector_db/old_version/
```

2. ログファイルをクリーンアップ:
```bash
rm logs/*.log
```

3. data/output/ ディレクトリの古いファイルを削除

---

## デバッグ方法

### ログレベルの変更

```bash
# DEBUG レベルで詳細ログを出力
export LOG_LEVEL=DEBUG
python apps/answer-support/main.py
```

### リアルタイムログ監視

```bash
tail -f logs/app.log
```

### DB 内容確認

```bash
python scripts/check_db_content.py
```

出力例:
```text
=== ChromaDB Content Analysis ===
Collection: deposit_DB
Total documents: 943
Unique documents: 943
Duplicate documents: 0

Source distribution:
  scenario: 816
  faq_data: 127
```

---

## よくある質問 (FAQ)

### Q: ベクトルDBを完全にリセットするには?

```bash
# 1. 全Pythonプロセスを停止
pkill -f python

# 2. DBディレクトリを削除
rm -rf data/vector_db/

# 3. データ処理を再実行（自動的にDB再構築）
python apps/answer-support/main.py
```

### Q: 複数の埋め込みモデルを同時に使用できる?

いいえ。同一コレクションに異なるモデルのベクトルは混在できません。
事務改定評価システムでは、プロバイダーごとに別々のDBディレクトリを使用しています。

```
data/vector_db/rev01smile/
├── azure_openai/    # Azure OpenAI専用
└── vertex_ai/       # VertexAI専用
```

### Q: LLM プロバイダーを変更するには?

`.env` ファイルを編集:
```env
DEFAULT_LLM_PROVIDER=anthropic  # gemini, anthropic, openai
DEFAULT_LLM_MODEL=claude-3-5-sonnet-20241022
```

---

## サポート

上記の方法で解決しない場合:

1. ログファイルを確認: `logs/app.log`
2. GitHub Issues で報告
3. [docs/ARCHITECTURE.md](./ARCHITECTURE.md) でシステム構成を確認

---

## 関連ドキュメント

- [README.md](../README.md) - 概要とセットアップ
- [docs/GOOGLE_CLOUD_AUTH.md](./GOOGLE_CLOUD_AUTH.md) - 認証設定
- [docs/SECURITY.md](./SECURITY.md) - セキュリティガイド
- [docs/CONFIGURATION.md](./CONFIGURATION.md) - 設定詳細
