# Vertex AI SDK 移行 + 起動速度改善

## Context

2つの問題を同時に解決する:

1. **SDK非推奨**: `vertexai.language_models.TextEmbeddingModel` と `vertexai.init()` が2026年6月24日に削除予定
   - 参照: https://cloud.google.com/vertex-ai/generative-ai/docs/deprecations/genai-vertexai-sdk
2. **起動が数分かかる**: `_load_vertex_ai_modules()` が `ChatGoogleGenerativeAI`（langchain）を一括インポートし、langchain_core → transformers → torch → sympy という巨大チェーンが走る。embedding 初期化時にLLM不要なのに重いインポートが発生。

**根本原因**: `_load_vertex_ai_modules()` が3つの無関係な依存（`vertexai`, `service_account`, `ChatGoogleGenerativeAI`）を1関数に詰め込んでいる。

---

## 修正一覧

### 1. `src/utils/auth.py` — `_load_vertex_ai_modules()` 分解 + SDK移行

**Before**: 1つの遅延ローダーに3つ詰め込み
```python
vertexai = None
service_account = None
ChatGoogleGenerativeAI = None

def _load_vertex_ai_modules():
    global vertexai, service_account, ChatGoogleGenerativeAI
    if vertexai is None:
        import vertexai as _vertexai                                    # SDK移行で不要
        from google.oauth2 import service_account as _service_account   # 認証用（軽量）
        from langchain_google_genai import ChatGoogleGenerativeAI ...   # LLM用（重い）
```

**After**: 用途別に分離
```python
service_account = None

def _load_auth_modules():
    """認証モジュールのみ遅延ロード（軽量・高速）"""
    global service_account
    if service_account is None:
        from google.oauth2 import service_account as _sa
        service_account = _sa

def create_genai_client(config, credentials=None):
    """google-genai Client を生成（vertexai.init() の代替）"""
    from google import genai  # 軽量
    if credentials is None:
        credentials = get_google_credentials(config)
    return genai.Client(
        vertexai=True,
        project=config.gemini_project_id,
        location=config.gemini_location,
        credentials=credentials,
    )

def create_llm(config):
    """LLM生成時にのみ langchain をインポート"""
    from langchain_google_genai import ChatGoogleGenerativeAI  # ここでのみインポート
    credentials = get_google_credentials(config)
    return ChatGoogleGenerativeAI(
        model=config.llm_model, temperature=0,
        project=config.gemini_project_id,
        location=config.gemini_location,
        credentials=credentials,
    )
```

**呼び出し元の変更:**
- `_get_credentials_local()`, `_get_credentials_key_vault()`: `_load_vertex_ai_modules()` → `_load_auth_modules()`
- `initialize_vertex_ai()`: **関数ごと削除**（`create_genai_client()` で代替）
- `create_llm()`: インライン import に変更（遅延ローダー不要）
- モジュール変数 `vertexai`, `ChatGoogleGenerativeAI`: 削除

### 2. `src/utils/gemini_embedding.py` — Client API + 例外クラス

import:
```python
# Before
from src.utils.auth import initialize_vertex_ai
from vertexai.language_models import TextEmbeddingModel
from google.api_core.exceptions import ServiceUnavailable, TooManyRequests, ...

# After
from google.genai.errors import ClientError, ServerError
```

`_setup_model()`:
```python
# Before
initialize_vertex_ai(self.config)
model = TextEmbeddingModel.from_pretrained(model_name)
return model

# After
from src.utils.auth import create_genai_client
self._client = create_genai_client(self.config)
self._model_name = model_name
return self._client
```

`_get_embeddings_with_retry()`:
```python
# Before
return self.model.get_embeddings(batch_texts)

# After
from google.genai.types import EmbedContentConfig
response = self._client.models.embed_content(
    model=self._model_name,
    contents=batch_texts,
    config=EmbedContentConfig(output_dimensionality=self.EMBEDDING_DIM),
)
return response.embeddings
```

リトライ例外:
```python
# Before: 個別例外5種
retry=retry_if_exception_type((ServiceUnavailable, TooManyRequests, DeadlineExceeded))

# After: 2クラス + ステータスコード判定
def _is_retryable(error):
    if isinstance(error, ServerError): return True      # 5xx
    if isinstance(error, ClientError) and getattr(error, 'code', 0) == 429: return True  # Rate limit
    return False
```

`VERTEX_AI_AVAILABLE` フラグ: `from vertexai...` → `from google.genai.errors import ...` に判定変更。
フォールバック例外: `ServiceUnavailable` 等5種 → `ClientError`/`ServerError` 2種に。

### 3. `config.py`

```python
# Before
EMBEDDING_BATCH_SIZE: int = 250  # Vertex AI API上限
# Validation: > 250

# After
EMBEDDING_BATCH_SIZE: int = 100  # google-genai SDK 上限
# Validation: > 100
```

### 4. `requirements.txt`

```diff
-google-cloud-aiplatform[vertexai]>=1.35.0
+google-genai>=1.0.0
```

### 5. 孤立パッケージ削除

`sentence-transformers`（→ `transformers` → `torch` → `sympy`）は requirements.txt に含まれず、プロジェクトコードでも import されていない孤立パッケージ。`pip uninstall sentence-transformers transformers torch` で削除。

### 6. `docs/TROUBLESHOOTING.md`

非推奨警告セクションを「移行済み」に更新。起動速度セクションの追記は不要（問題が解消されるため）。

---

## 修正対象ファイル一覧

| # | ファイル | 変更内容 |
|---|---------|---------|
| 1 | `src/utils/auth.py` | `_load_vertex_ai_modules` 分解 + `create_genai_client` 新設 + `initialize_vertex_ai` 削除 |
| 2 | `src/utils/gemini_embedding.py` | Client API + 例外クラス全面書き換え |
| 3 | `config.py` | `EMBEDDING_BATCH_SIZE` 250→100 |
| 4 | `requirements.txt` | `google-cloud-aiplatform` → `google-genai` |
| 5 | `docs/TROUBLESHOOTING.md` | 移行完了ステータス更新 |

---

## 起動速度改善の効果

**Before**: embedding 初期化 → `_load_vertex_ai_modules()` → `ChatGoogleGenerativeAI` → langchain_core → transformers → torch → sympy（数分）

**After**: embedding 初期化 → `create_genai_client()` → `from google import genai`（数秒）。langchain は `create_llm()` 呼び出し時（LLM検索モード使用時）にのみインポートされる。

---

## 検証方法

1. `pip install google-genai && pip uninstall sentence-transformers transformers torch -y`
2. `pytest tests/ -x` で既存141テスト全パス
3. 回答支援AI起動 → 非推奨警告が消えていること + **起動が数秒で完了すること** + 検索動作
4. 運用保守AI起動 → VertexAI プロバイダーで検索動作
5. LLM検索モード（llm_enhanced）→ langchain がオンデマンドでロードされ、クエリ拡張が動作すること
