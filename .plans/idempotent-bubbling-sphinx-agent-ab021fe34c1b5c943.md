# Vertex AI SDK -> google-genai SDK 移行調査レポート

## 概要

`vertexai.language_models.TextEmbeddingModel` と `vertexai.init()` は **2025年6月24日に非推奨** となり、**2026年6月24日に削除予定**。移行先は `google-genai` SDK。

---

## 1. 新SDK (`google-genai`) の API 仕様

### 1.1 インストール

```bash
pip install google-genai
```

`google-cloud-aiplatform[vertexai]` の代替。`google-genai` は独立した軽量パッケージ。

### 1.2 クライアント初期化（サービスアカウント認証）

```python
from google import genai
from google.oauth2 import service_account

credentials = service_account.Credentials.from_service_account_file(
    '/path/to/key.json',
    scopes=['https://www.googleapis.com/auth/cloud-platform']
)

client = genai.Client(
    vertexai=True,
    project='your-project-id',
    location='us-central1',
    credentials=credentials,
)
```

**現行コードとの対応:**
- `vertexai.init(project=..., location=..., credentials=...)` → `genai.Client(vertexai=True, project=..., location=..., credentials=...)`
- `google.oauth2.service_account` はそのまま使用可能（`google-auth` 継続必要）
- Key Vault 経由の `from_service_account_info()` も同様に使用可能

### 1.3 Embedding API

**現行（非推奨）:**
```python
from vertexai.language_models import TextEmbeddingModel
model = TextEmbeddingModel.from_pretrained("gemini-embedding-001")
embeddings = model.get_embeddings(batch_texts)
for emb in embeddings:
    vector = emb.values  # List[float]
```

**移行先:**
```python
from google import genai
from google.genai.types import EmbedContentConfig

response = client.models.embed_content(
    model="gemini-embedding-001",
    contents=["text1", "text2", "text3"],  # str or List[str]
    config=EmbedContentConfig(
        output_dimensionality=3072,
        # task_type="RETRIEVAL_DOCUMENT",  # オプション
    ),
)

for embedding in response.embeddings:
    vector = embedding.values  # List[float]
```

### 1.4 バッチ Embedding

- `contents` パラメータに `List[str]` を渡すことでバッチ処理可能
- **API バッチサイズ上限: 100件**（現行の `EMBEDDING_BATCH_SIZE=250` を要変更）
- レスポンス: `response.embeddings` は `List[ContentEmbedding]`、各要素に `.values` (List[float])

### 1.5 エラー/例外クラス

```python
from google.genai.errors import APIError, ClientError, ServerError
```

| クラス | HTTP範囲 | 対応する旧例外 |
|--------|----------|---------------|
| `ClientError(APIError)` | 4xx | `PermissionDenied(403)`, `InvalidArgument(400)`, `TooManyRequests(429)` |
| `ServerError(APIError)` | 5xx | `ServiceUnavailable(503)`, `DeadlineExceeded` |
| `APIError(Exception)` | その他 | ベースクラス |

**重要な違い:**
- `google.api_core.exceptions` の個別例外（ServiceUnavailable, TooManyRequests等）は**使用不可**
- `ClientError` / `ServerError` の `.code` プロパティで HTTP ステータスコードを判別する必要がある
- `google.api_core.retry.if_transient_error` は `google.genai` のエラーをキャッチ**しない**

**リトライロジックの変更:**
```python
# 現行
@retry(retry=retry_if_exception_type((ServiceUnavailable, TooManyRequests, DeadlineExceeded)))

# 移行先（案）
def _is_retryable(error):
    if isinstance(error, ServerError):
        return True  # 5xx は全てリトライ
    if isinstance(error, ClientError) and error.code == 429:
        return True  # Rate limit のみリトライ
    return False

@retry(retry=retry_if_exception(lambda e: _is_retryable(e)))
```

### 1.6 EmbedContentConfig のオプション

| パラメータ | 型 | 説明 |
|-----------|------|------|
| `task_type` | str | SEMANTIC_SIMILARITY, RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY 等 |
| `output_dimensionality` | int | 128-3072（デフォルト3072） |
| `title` | str | RETRIEVAL_DOCUMENT 使用時のドキュメントタイトル |

---

## 2. LangChain との互換性

### 2.1 langchain-google-genai 4.0.0 の変更

- `langchain-google-genai>=4.0.0` は**内部的に `google-genai` SDK を使用**（旧 `google-ai-generativelanguage` から移行済み）
- `vertexai` SDK への直接依存は**ない**
- **つまり LangChain 側の移行は不要**

### 2.2 ChatGoogleGenerativeAI のパラメータ

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    project="your-project-id",
    location="us-central1",
    credentials=credentials,       # google.auth.credentials.Credentials
    # vertexai=True,               # credentials/project 指定時は自動判定
)
```

- `project` パラメータ指定 → 自動的に Vertex AI バックエンドを使用
- `credentials` パラメータ指定 → 自動的に Vertex AI バックエンドを使用
- **現行コードの `create_llm()` は API 変更なしで動作する可能性が高い**

### 2.3 langchain-google-vertexai の状況

- `ChatVertexAI` は `ChatGoogleGenerativeAI` に統合・非推奨化の方向
- 本プロジェクトでは `langchain-google-vertexai` は使用していないため影響なし

### 2.4 Breaking Changes (langchain-google-genai 4.0.0)

- gRPC トランスポート削除（REST のみ）
- `with_structured_output()` のデフォルトが `method="json_schema"` に変更
- Embeddings のモデルパラメータ: `model_name` → `model` にリネーム
- `GoogleVectorStore` モジュール削除

---

## 3. 現在の依存関係の変更

### 3.1 requirements.txt の変更

```diff
 # LangChain（LLM拡張検索機能用）
 langchain>=0.1.0
 langchain-google-genai>=4.0.0

 # Google Cloud / Vertex AI（LLM + 埋め込みモデル）
-google-cloud-aiplatform[vertexai]>=1.35.0
+google-genai>=1.0.0
 google-auth>=2.17.0
```

### 3.2 依存関係の分析

| パッケージ | 状態 | 理由 |
|-----------|------|------|
| `google-cloud-aiplatform[vertexai]` | **削除** | `google-genai` で代替 |
| `google-genai` | **追加** | Embedding API の直接使用 |
| `google-auth` | **継続** | サービスアカウント認証 (`google.oauth2.service_account`) |
| `langchain-google-genai>=4.0.0` | **継続** | 内部で `google-genai` を使用済み |
| `tenacity` | **継続** | リトライロジック（例外クラスの変更のみ） |

### 3.3 注意: langchain-google-genai の推移的依存

`langchain-google-genai>=4.0.0` は内部で `google-genai` を依存に持つため、
`google-genai` の明示的な追加は厳密には不要かもしれないが、
`gemini_embedding.py` が直接 `from google import genai` するため、**明示的に記載すべき**。

---

## 4. テストへの影響

### 4.1 テスト内の vertexai 参照

テストで `vertexai` を直接 import しているファイルは**なし**。

唯一の関連箇所:
- `tests/unit/test_timestamp_migration.py:17` — `config.VALID_EMBEDDING_PROVIDERS = ("vertex_ai", "azure_openai")` を設定（文字列定数のみ、SDKインポートなし）

### 4.2 テストへの影響サマリ

| テストファイル | 影響 |
|---------------|------|
| `test_run_eval_cache.py` | `create_llm` を `patch` — API不変のため影響なし |
| `test_timestamp_migration.py` | 文字列 `"vertex_ai"` のみ — 影響なし |
| `test_ui_shared.py` | `_create_llm_analysis_section` のテスト — API不変のため影響なし |

**結論: 既存テストの修正は不要。ただし `gemini_embedding.py` のリトライロジック変更に対応する新テストが望ましい。**

---

## 5. 移行対象ファイルと変更点サマリ

### 5.1 `src/utils/auth.py`

| 変更箇所 | 内容 |
|----------|------|
| `_load_vertex_ai_modules()` | `import vertexai` → `from google import genai` + `genai.Client` 生成 |
| `initialize_vertex_ai()` | `vertexai.init(...)` → `genai.Client(vertexai=True, ...)` を返す方式に変更 |
| `create_llm()` | `ChatGoogleGenerativeAI` の呼び出しは変更不要（パラメータ互換） |
| モジュールレベル変数 | `vertexai = None` → `genai_client = None` |

**設計判断ポイント:**
- `initialize_vertex_ai()` は現在 `None` を返す（副作用のみ）
- 新SDKでは `genai.Client` インスタンスが必要 → 返り値として返すか、モジュール変数で保持するか

### 5.2 `src/utils/gemini_embedding.py`

| 変更箇所 | 内容 |
|----------|------|
| import | `from vertexai.language_models import TextEmbeddingModel` → `from google.genai.types import EmbedContentConfig` |
| 例外 import | `from google.api_core.exceptions import ...` → `from google.genai.errors import ClientError, ServerError` |
| `_setup_model()` | `TextEmbeddingModel.from_pretrained(...)` → `genai.Client` を受け取る方式に変更 |
| `_get_embeddings_with_retry()` | `model.get_embeddings(batch_texts)` → `client.models.embed_content(...)` |
| `encode()` | `embedding.values` → `embedding.values`（同名だが型が変わる可能性） |
| リトライ対象例外 | `(ServiceUnavailable, TooManyRequests, DeadlineExceeded)` → `ServerError` + `ClientError(429)` |
| `EMBEDDING_BATCH_SIZE` | 250 → **100** に変更（API上限の変更） |

### 5.3 `config.py`

| 変更箇所 | 内容 |
|----------|------|
| `EMBEDDING_BATCH_SIZE` | デフォルト値 `250` → `100`、バリデーション上限も変更 |

### 5.4 `requirements.txt`

上記 3.1 の通り。

### 5.5 `docs/TROUBLESHOOTING.md`

移行完了後にステータスを更新。

---

## 6. リスクと注意事項

1. **バッチサイズ上限の縮小 (250→100)**: DB構築時間が約2.5倍に延びる可能性
2. **エラーハンドリングの粒度低下**: 個別例外→ HTTP ステータスコード判定に変更
3. **`genai.Client` のライフサイクル管理**: シングルトン化が望ましい（現行の `vertexai.init()` はグローバルステート）
4. **Embedding レスポンス互換性**: `.values` プロパティ名は同じだが、型の完全一致は要検証
5. **langchain-google-genai のバージョン**: 4.0.0 が `google-genai` に内部移行済みだが、`vertexai` パラメータの自動検出挙動は要検証
