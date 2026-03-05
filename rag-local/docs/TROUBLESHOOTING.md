# トラブルシューティング

このドキュメントでは、RAG-Localシステムでよく発生する問題と解決方法を説明します。

## 目次

- [認証関連](#認証関連)
- [データベース関連](#データベース関連)
- [検索・処理関連](#検索処理関連)
- [API関連](#api関連)
- [Streamlit / SDK 警告](#streamlit--sdk-警告)
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
# Windows
dir rag-local\gemini_credentials.json

# Linux/Mac
ls -l rag-local/gemini_credentials.json
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

### Azure OpenAI 認証エラー

**症状:**
```
openai.AuthenticationError: Incorrect API key provided
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

### Azure OpenAI 未設定で改定影響調査が失敗する

**症状:**
```
ValueError: AZURE_OPENAI_EMBEDDING_ENDPOINT is not set
```

**原因:**
- `run_eval.py` のデフォルト（`--provider both`）は Azure OpenAI と VertexAI の両方を使用する
- Azure OpenAI の環境変数（`AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`）が未設定

**解決策:**

VertexAI のみで実行する:
```bash
python apps/revision-ops/run_eval.py --provider vertex
```

Streamlit UI（`ops_ui.py`）は環境変数から利用可能なプロバイダーを自動検出するため、この問題は発生しません。

---

## データベース関連

### ChromaDB（検索用データベース）エラー: コレクションが見つからない

**症状:**
```
ValueError: Collection 'naibujimu' does not exist.
```

**原因:**
- 検索用データベース（ベクトルDB）が初期化されていない
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
python scripts/build_db.py --revisions-only
```

### 用途別の再構築コマンド

| 用途 | コマンド | 対象DB |
|------|---------|--------|
| 回答支援AI用のみ | `python scripts/build_db.py --no-revisions` | naibujimu, smile 等 |
| 改定影響調査用のみ | `python scripts/build_db.py --revisions-only` | rev01_smile, rev02_souzoku 等 |
| 両方 | `python scripts/build_db.py` | 全DB |

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

### ChromaDB エラー: An instance of Chroma already exists with different settings

**症状:**

DB再構築後や設定変更後に検索結果が0件になる、または以下のエラーが表示される:
```
An instance of Chroma already exists for ... with different settings
```

**原因:**
- ChromaDB はプロセス内で同一パスに1つのクライアントしか許可しない
- DB再構築後も古いクライアントインスタンスがメモリに残っている

**解決策:**

Streamlit を再起動する（Ctrl+C → 再度 `streamlit run ...`）。
DB・設定変更後は必ず再起動が必要。

### 業務分野が検出されない

**対処**:
1. `data/source/faq/latest/` と `data/source/scenarios/latest/` にファイルが存在するか確認
2. ファイル名が `{業務名}_履歴データ_{YYYYMMDD}.xlsx` / `{業務名}_シナリオデータ_{YYYYMMDD}.xlsx` の形式か確認
3. `config/business_areas.yaml` に業務名のマッピングが登録されているか確認

### DB構築後にUIで業務分野が表示されない

**対処**: `data/vector_db/{業務名}/azure_openai/chroma.sqlite3` が存在するか確認。存在しない場合は `python scripts/build_db.py --force` で再構築

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

改定影響調査に固有の問題は [docs/REVISION_OPS.md](./REVISION_OPS.md) の「既知の問題と注意事項」セクションを参照してください。

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
# config.py の SearchConfig クラス定数を変更
EMBEDDING_BATCH_SIZE: int = 50   # デフォルト250から縮小
VECTOR_DB_BATCH_SIZE: int = 50   # デフォルト100から縮小
```

2. システムメモリを増強（推奨: 16GB以上）

3. 処理データを分割:
```bash
# --limit オプションで処理件数を制限
python apps/answer-support/main.py --limit 50
python apps/answer-support/main.py --business naibujimu --limit 50
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

```python
# 業務分野名は自動的に英語変換されます
# 例: "預金" → "deposit" → naibujimu に統合済み
# 現在の主要コレクション: naibujimu, smile, rev{XX}_{bot}
```

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

### 検索結果が0件になる

**症状:**
- バッチ処理やUIで検索しても結果が返らない

**原因と対処:**

1. **DB が空**: `python scripts/check_db_content.py` で件数を確認。0件なら `python scripts/build_db.py --force` で再構築
2. **検索対象の設定ミスマッチ**: `settings.yaml` の `common.search_source` がDB内のデータ種別と一致しているか確認。`history_data` は問い合わせ履歴データ（FAQ）、`scenario` はシナリオデータを意味します
3. **文章変換モデル（埋め込みプロバイダー）の不一致**: `.env` の `DEFAULT_EMBEDDING_PROVIDER` と、構築済みDBのプロバイダーが一致しているか確認（例: vertex_ai で構築したDBに azure_openai でアクセスしていないか）。DB構築時と検索時で異なるモデルを使うと、文章から変換された数値の意味が変わるため、正しく類似検索ができません
4. **Streamlit キャッシュ**: DB再構築後はStreamlitを再起動（Ctrl+C → 再起動）

  Streamlit は `@st.cache_resource` でDBクライアントやLLMをキャッシュしているため、DB再構築後は必ず再起動が必要です（`session_state` のキャッシュ値も古いまま残ります）。

### バッチ処理で入力ファイルが見つからない

**症状:**
- バッチ実行時に結果が生成されない、またはスキップされる

**原因:**
- `data/input/` にファイルがない、またはファイル名が期待形式と異なる

**解決策:**

1. 入力ファイルの配置確認:
```bash
ls data/input/
```

2. ファイル名の確認: `{業務名}_YYYYMMDD.xlsx` 形式で、`{業務名}` は `config/business_areas.yaml` に登録されている日本語名（例: `スマイル_20250301.xlsx`）

### LLM 未初期化エラー

**症状:**
```
RuntimeError: LLM is not initialized. Set DEFAULT_LLM_PROVIDER and DEFAULT_LLM_MODEL.
```

**原因:**
- `DEFAULT_LLM_PROVIDER` / `DEFAULT_LLM_MODEL` が未設定（全モードで起動時に必須）
- `search_mode: llm_enhanced` 設定時に GCP 認証（`GEMINI_PROJECT_ID` + `gemini_credentials.json`）が未設定

**解決策:**

1. **回答支援AI（類似回答検索）** で LLM が不要な場合、`original` モードに戻す:
```yaml
# config/settings.yaml
common:
  search_mode: original
```

2. **運用保守効率化AI（改定影響調査）**（`run_eval.py`）では `search_mode` に関係なく LLM を常に初期化します。この場合は解決策 3 の環境変数設定が必須です。

3. LLM を使用する場合、環境変数を設定:
```env
DEFAULT_LLM_PROVIDER=gemini
DEFAULT_LLM_MODEL=gemini-2.5-flash-lite
GEMINI_PROJECT_ID=your-project-id
```

---

## API関連

### API レート制限エラー

**症状:**
```
google.genai.errors.ClientError: 429 Quota exceeded
```

**原因:**
- Gemini API の使用量制限に到達

**解決策:**

1. リトライ処理を待つ（自動リトライ機能あり: 3回、指数バックオフ）

2. バッチサイズを縮小:
```python
# config.py の SearchConfig クラス定数を変更
EMBEDDING_BATCH_SIZE: int = 50   # デフォルト250から縮小（Gemini は内部で最大100にキャップ）
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

1. 自動リトライを待つ（tenacity によるリトライ処理が組み込み済み）

2. LLM分析を無効化:
```bash
ENABLE_LLM_ANALYSIS=false python apps/revision-ops/run_eval.py
```

3. ネットワーク接続を確認

4. Google Cloud Console でクォータ状況を確認

### LLM分析が失敗する

**症状:**
```
ValueError: DEFAULT_LLM_PROVIDER環境変数が設定されていません
```

**原因:** `run_eval.py` はデフォルトで `enable_llm_analysis=True`（JudgmentSupport による関連性判定）。LLM 環境変数が未設定だと初期化エラー。

**解決策:**
1. LLM分析を無効化: `ENABLE_LLM_ANALYSIS=false python apps/revision-ops/run_eval.py`
2. LLM環境変数を設定: `DEFAULT_LLM_PROVIDER=gemini`, `DEFAULT_LLM_MODEL=gemini-2.5-flash-lite`, `GEMINI_PROJECT_ID`

## Streamlit / SDK 警告

### Streamlit 起動時にログが重複出力される

**症状:**
```
業務分野設定を読み込みました: 15件 + 事務改定9件
業務分野設定を読み込みました: 15件 + 事務改定9件
業務分野設定を読み込みました: 15件 + 事務改定9件
```

**原因:**
Streamlit はユーザー操作（業務分野選択、検索実行等）のたびにスクリプト全体を再実行する設計です。
`DynamicDBManager` → `BusinessAreaTranslator` が再実行ごとに新規インスタンス化され、設定読み込みログが複数回出力されます。

**影響:** なし（YAML読み込みはミリ秒単位のため、パフォーマンス影響は無視できる）。

---

### Vertex AI SDK 非推奨警告（移行済み）

**ステータス:** 移行完了（`google-cloud-aiplatform` → `google-genai` SDK）

`vertexai.language_models.TextEmbeddingModel` と `vertexai.init()` は `google-genai` SDK の `genai.Client.models.embed_content()` に移行済みです。非推奨警告は表示されません。

**移行内容:**
- `src/utils/auth.py`: `initialize_vertex_ai()` → `create_genai_client()` に置換
- `src/utils/gemini_embedding.py`: `TextEmbeddingModel` → `genai.Client` API に置換
- `requirements.txt`: `google-cloud-aiplatform[vertexai]` → `google-genai`

**参照:** https://cloud.google.com/vertex-ai/generative-ai/docs/deprecations/genai-vertexai-sdk

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
- 文章を検索用の数値データに変換する処理（ベクトル化）が初回に実行されるため（正常な動作）

**解決策:**

1. 初回のみ時間がかかります（2回目以降は数秒）
2. ファイルの更新日時を確認する仕組み（タイムスタンプ検証）により、変更のないデータの再変換を自動的に回避します

### 検索が遅い

**症状:**
- 各クエリに数秒かかる

**原因:**
- AIによる検索語の自動補強（LLMクエリ拡張）が有効になっている。この機能はユーザーの入力をAIモデルに送信して関連語句を追加するため、AIモデルへの問い合わせが発生し時間がかかる

**解決策:**

1. 原文検索モードに変更:
```yaml
# config/settings.yaml の common セクション
common:
  search_mode: original
```

2. キーワード重みを下げる:
```yaml
# config/settings.yaml の common セクション
common:
  vector_weight: 1.0  # keyword_weight は 1.0 - vector_weight で自動計算（= 0.0）
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
Collection: naibujimu
Total documents: 11439
Unique documents: 11439
Duplicate documents: 0

Source distribution:
  scenario: 1384        ← シナリオデータの件数
  faq_data: 10055       ← 問い合わせ履歴データ（FAQ）の件数
```

---

## よくある質問

### Q: ベクトルDBを完全にリセットするには?

```bash
# 1. 全Pythonプロセスを停止
pkill -f python

# 2. DBディレクトリを削除
rm -rf data/vector_db/

# 3. データ処理を再実行（自動的にDB再構築）
python apps/answer-support/main.py
```

### Q: 複数の文章を数値に変換するAIモデル（埋め込みモデル）を同時に使用できる?

いいえ。同一のデータの格納単位（コレクション）に異なるモデルの数値データ（ベクトル）は混在できません。
運用保守効率化AI（改定影響調査）では、プロバイダーごとに別々のDBディレクトリを使用しています。

```
data/vector_db/rev01_smile/
├── azure_openai/    # Azure OpenAI専用
└── vertex_ai/       # VertexAI専用
```

### Q: LLM プロバイダーを変更するには?

現在は Gemini のみサポートしています。`.env` でモデルを変更できます:
```env
DEFAULT_LLM_PROVIDER=gemini
DEFAULT_LLM_MODEL=gemini-2.5-flash-lite  # gemini-2.5-flash, gemini-2.5-pro も利用可能
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
- [docs/ANSWER_SUPPORT.md](./ANSWER_SUPPORT.md) - 回答支援AI詳細
- [docs/REVISION_OPS.md](./REVISION_OPS.md) - 改定影響調査詳細
- [docs/CONFIGURATION.md](./CONFIGURATION.md) - 設定リファレンス
- [docs/ARCHITECTURE.md](./ARCHITECTURE.md) - アーキテクチャ
