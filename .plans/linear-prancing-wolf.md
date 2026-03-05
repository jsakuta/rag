# Vertex AI SDK 移行 + 起動速度改善 — 最終実装計画

## Context

2つの問題を同時に解決する:

1. **SDK非推奨**: `vertexai.language_models.TextEmbeddingModel` と `vertexai.init()` が 2026-06-24 に削除予定。移行先: `google-genai` SDK
2. **起動が数分かかる**: `_load_vertex_ai_modules()` が `ChatGoogleGenerativeAI`（langchain）を一括 import し、langchain_core → transformers → torch → sympy という巨大チェーンが走る

**根本原因**: `_load_vertex_ai_modules()` が 3つの無関係な依存を 1 関数に詰め込んでいる。

**レビュー指摘（6件）を全て反映済み**。元計画: `.plans/idempotent-bubbling-sphinx.md`

---

## 修正ファイル一覧

| # | ファイル | 変更概要 |
|---|---------|---------|
| 1 | `src/utils/auth.py` | 遅延ローダー分解 + `create_genai_client` 新設 + `initialize_vertex_ai` 削除 |
| 2 | `src/utils/gemini_embedding.py` | Client API + 例外クラス + バッチサイズ内部キャップ |
| 3 | `requirements.txt` | `google-cloud-aiplatform` → `google-genai` |
| 4 | `docs/TROUBLESHOOTING.md` | 移行完了ステータス更新 |

**変更しないファイル**: `config.py`（`EMBEDDING_BATCH_SIZE` は Azure と共用のため維持）

---

## Step 1: `src/utils/auth.py`

### 1a. グローバル変数とインポート

```python
# Before (L9-12)
# 遅延インポート: Vertex AI関連（インストールされていない場合もエラーにしない）
vertexai = None
service_account = None
ChatGoogleGenerativeAI = None

# After
# 遅延インポート: 認証モジュール
service_account = None
```

- `vertexai` 変数: 削除（`google.genai.Client` に置換）
- `ChatGoogleGenerativeAI` 変数: 削除（`create_llm()` 内でインライン import）

### 1b. `_load_vertex_ai_modules()` → `_load_auth_modules()`

```python
# Before (L21-30)
def _load_vertex_ai_modules():
    """Vertex AI関連モジュールを遅延ロード"""
    global vertexai, service_account, ChatGoogleGenerativeAI
    if vertexai is None:
        import vertexai as _vertexai
        from google.oauth2 import service_account as _service_account
        from langchain_google_genai import ChatGoogleGenerativeAI as _ChatGoogleGenerativeAI
        vertexai = _vertexai
        service_account = _service_account
        ChatGoogleGenerativeAI = _ChatGoogleGenerativeAI

# After
def _load_auth_modules():
    """認証モジュールのみ遅延ロード（軽量・高速）"""
    global service_account
    if service_account is None:
        from google.oauth2 import service_account as _sa
        service_account = _sa
```

### 1c. `_get_credentials_local()` — 呼び出し変更

```python
# L45: _load_vertex_ai_modules() → _load_auth_modules()
```

### 1d. `_get_credentials_key_vault()` — 呼び出し変更

```python
# L72: _load_vertex_ai_modules() → _load_auth_modules()
```

### 1e. `create_genai_client()` — 新設（`initialize_vertex_ai()` の代替）

```python
def create_genai_client(config: 'SearchConfig', credentials=None):
    """google-genai Client を生成（旧 initialize_vertex_ai の代替）

    Args:
        config: SearchConfig インスタンス
        credentials: 認証情報（省略時は自動取得）

    Returns:
        genai.Client インスタンス
    """
    from google import genai
    if credentials is None:
        credentials = get_google_credentials(config)
    client = genai.Client(
        vertexai=True,
        project=config.gemini_project_id,
        location=config.gemini_location,
        credentials=credentials,
    )
    logger.info("google-genai Client initialized successfully")
    return client
```

### 1f. `initialize_vertex_ai()` — 削除

L116-138 の関数を丸ごと削除。

### 1g. `create_llm()` — インライン import に変更

```python
# Before (L141-170)
def create_llm(config: 'SearchConfig'):
    ...
    _load_vertex_ai_modules()
    ...
    credentials = get_google_credentials(config)
    return ChatGoogleGenerativeAI(...)

# After
def create_llm(config: 'SearchConfig'):
    """Gemini LLMインスタンスを生成"""
    if config.llm_provider != "gemini":
        raise ValueError(
            f"Unsupported LLM provider: {config.llm_provider}. "
            f"Currently only 'gemini' is supported"
        )

    if not config.gemini_project_id:
        raise ValueError("GEMINI_PROJECT_ID environment variable is not set")

    from langchain_google_genai import ChatGoogleGenerativeAI  # ここでのみインポート
    credentials = get_google_credentials(config)
    return ChatGoogleGenerativeAI(
        model=config.llm_model,
        temperature=0,
        project=config.gemini_project_id,
        location=config.gemini_location,
        credentials=credentials,
    )
```

### 1h. `create_embedding_model()` — 変更なし

ファクトリー関数はそのまま。内部の `GeminiEmbeddingModel` が変わるだけ。

---

## Step 2: `src/utils/gemini_embedding.py`

### 2a. import セクション

```python
# Before (L1-15)
import threading
import numpy as np
from typing import List, Union, Optional, TYPE_CHECKING
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.utils.base_embedding import BaseEmbeddingModel
from src.utils.logger import setup_logger
from src.utils.auth import initialize_vertex_ai

# After
import threading
import numpy as np
from typing import List, Union, Optional, TYPE_CHECKING
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

from src.utils.base_embedding import BaseEmbeddingModel
from src.utils.logger import setup_logger
```

**変更点:**
- `retry_if_exception_type` → `retry_if_exception`
- `from src.utils.auth import initialize_vertex_ai` → 削除（`_setup_model()` 内でインライン import）

### 2b. SDK import + フォールバック例外クラス

```python
# Before (L17-47)
try:
    from vertexai.language_models import TextEmbeddingModel
    from google.api_core.exceptions import (
        ServiceUnavailable, TooManyRequests, DeadlineExceeded,
        PermissionDenied, InvalidArgument
    )
    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False
    class ServiceUnavailable(Exception): pass
    class TooManyRequests(Exception): pass
    class DeadlineExceeded(Exception): pass
    class PermissionDenied(Exception): pass
    class InvalidArgument(Exception): pass
    logger.warning("Vertex AI SDK not installed. Run: pip install google-cloud-aiplatform")

# After
try:
    from google.genai.errors import ClientError, ServerError
    from google.genai.types import EmbedContentConfig
    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False
    class ClientError(Exception):
        """google-genai SDK not available - placeholder"""
        def __init__(self, *args, code=None, **kwargs):
            self.code = code
            super().__init__(*args, **kwargs)
    class ServerError(Exception):
        """google-genai SDK not available - placeholder"""
        pass
    EmbedContentConfig = None
    logger.warning("google-genai SDK not installed. Run: pip install google-genai")
```

**レビュー指摘反映:**
- H-1: フォールバック `ClientError` に `code` 属性を追加（`_is_retryable()` で使用するため）
- M-1: エラーメッセージを `google-genai` に更新

### 2c. `_is_retryable()` — モジュールレベル関数として新設

```python
def _is_retryable(error):
    """リトライ対象の例外かどうかを判定"""
    if isinstance(error, ServerError):
        return True       # 5xx は全てリトライ
    if isinstance(error, ClientError) and getattr(error, 'code', 0) == 429:
        return True       # Rate limit のみリトライ
    return False
```

**配置場所**: クラス定義の直前（デコレータがクラス定義時に評価されるため、モジュールレベルに置く必要がある）

### 2d. `_setup_model()` — Client API に変更

```python
# Before (L89-109)
def _setup_model(self):
    if not VERTEX_AI_AVAILABLE:
        raise ImportError(
            "Vertex AI SDK is not installed. "
            "Run: pip install google-cloud-aiplatform"
        )
    try:
        initialize_vertex_ai(self.config)
        model_name = self.config.embedding_model or "gemini-embedding-001"
        model = TextEmbeddingModel.from_pretrained(model_name)
        logger.debug(f"Gemini Embedding API initialized successfully (model: {model_name})")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize Gemini Embedding API: {e}")
        raise

# After
def _setup_model(self):
    if not VERTEX_AI_AVAILABLE:
        raise ImportError(
            "google-genai SDK is not installed. "
            "Run: pip install google-genai"
        )
    try:
        from src.utils.auth import create_genai_client
        self._model_name = self.config.embedding_model or "gemini-embedding-001"
        client = create_genai_client(self.config)
        logger.debug(f"Gemini Embedding API initialized (model: {self._model_name})")
        return client  # → base class が self.model に格納
    except Exception as e:
        logger.error(f"Failed to initialize Gemini Embedding API: {e}")
        raise
```

**レビュー指摘反映:**
- C-2: `self._client` を使わず、基底クラスの `self.model` を使用
- M-1: エラーメッセージを `google-genai` に更新

### 2e. `_get_embeddings_with_retry()` — API呼び出し変更

```python
# Before (L111-140)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    retry=retry_if_exception_type((ServiceUnavailable, TooManyRequests, DeadlineExceeded)),
    reraise=True
)
def _get_embeddings_with_retry(self, batch_texts: List[str]):
    try:
        return self.model.get_embeddings(batch_texts)
    except (PermissionDenied, InvalidArgument) as e:
        logger.error(f"回復不可能なAPIエラー: {type(e).__name__}")
        raise

# After
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    retry=retry_if_exception(_is_retryable),
    reraise=True
)
def _get_embeddings_with_retry(self, batch_texts: List[str]):
    """リトライ付きで Embedding API を呼び出す

    リトライ対象:
    - ServerError (5xx): サービス一時不可
    - ClientError (429): レート制限/クォータ超過

    リトライ対象外（即座に失敗）:
    - ClientError (403): 権限エラー
    - ClientError (400): 無効なリクエスト
    """
    try:
        response = self.model.models.embed_content(
            model=self._model_name,
            contents=batch_texts,
            config=EmbedContentConfig(output_dimensionality=self.EMBEDDING_DIM),
        )
        return response.embeddings
    except ClientError as e:
        if getattr(e, 'code', 0) != 429:
            logger.error(f"回復不可能なAPIエラー (HTTP {getattr(e, 'code', '?')}): {e}")
        raise
```

**レビュー指摘反映:**
- M-2: `retry_if_exception_type` → `retry_if_exception(_is_retryable)`
- H-2: 非リトライ `ClientError`（403, 400 等）の明示ログを維持

### 2f. `encode()` — バッチサイズ内部キャップ

```python
# Before (L158)
batch_size = self.config.EMBEDDING_BATCH_SIZE

# After
GENAI_SDK_BATCH_LIMIT = 100
batch_size = min(self.config.EMBEDDING_BATCH_SIZE, GENAI_SDK_BATCH_LIMIT)
```

**レビュー指摘反映:**
- C-1: config.py は変更せず、gemini_embedding.py 内部でキャップ。Azure 側は 250 のまま維持。

**`encode()` のそれ以外の部分は変更なし**。`embedding.values` のアクセスパターンは新 SDK でも同じ。

### 2g. 定数の配置

`GENAI_SDK_BATCH_LIMIT = 100` はクラス定数として定義:

```python
class GeminiEmbeddingModel(BaseEmbeddingModel):
    EMBEDDING_DIM = 3072
    GENAI_SDK_BATCH_LIMIT = 100  # google-genai SDK のバッチ上限
```

---

## Step 3: `requirements.txt`

```diff
-google-cloud-aiplatform[vertexai]>=1.35.0
+google-genai>=1.0.0
```

他の行は変更なし。

---

## Step 4: `docs/TROUBLESHOOTING.md`

非推奨警告セクションを「移行済み」に更新。具体的な内容は実装時にファイルを読んで確認する。

---

## 変更しないファイル（確認済み）

| ファイル | 理由 |
|---------|------|
| `config.py` | `EMBEDDING_BATCH_SIZE` は Azure と共用。Gemini 側は内部キャップで対応 |
| `src/core/searcher.py` | `create_llm()` / `create_embedding_model()` の API 署名不変 |
| `src/core/judgment_support.py` | 同上 |
| `apps/revision-ops/run_eval.py` | 同上 |
| `apps/revision-ops/ui/ops_ui.py` | 同上 |
| `src/utils/dynamic_db_manager.py` | `create_embedding_model()` 経由。内部変更なし |
| `src/utils/azure_embedding.py` | Azure 経路は変更なし |
| `tests/` | mock は `create_llm` のパスベース。API 署名不変 |
| `docs/ARCHITECTURE.md` | サービスは Vertex AI のまま（SDK が変わっただけ） |

---

## 検証方法

### 自動テスト
```bash
cd rag-local
pip install google-genai
pytest tests/ -x
```

### 手動テスト

1. **Vertex AI 経路**: `DEFAULT_EMBEDDING_PROVIDER=vertex_ai` で回答支援AI起動
   - 非推奨警告が消えていること
   - 起動が数秒で完了すること（数分→数秒）
   - 検索が正常動作すること

2. **Azure 経路**: `DEFAULT_EMBEDDING_PROVIDER=azure_openai` で起動
   - バッチサイズが 250 のまま維持されていること（ログで確認）
   - 検索が正常動作すること

3. **LLM 検索モード**: `search_mode=llm_enhanced` で検索
   - langchain がオンデマンドロードされること
   - クエリ拡張が動作すること

4. **SDK 未インストール時**: `pip uninstall google-genai -y` → Azure のみで起動
   - `VERTEX_AI_AVAILABLE=False` で Azure 経路が正常動作すること

5. **環境クリーンアップ**（オプション）:
   ```bash
   pip uninstall google-cloud-aiplatform sentence-transformers transformers torch -y
   ```
   - 削除後もテスト全パスすること
