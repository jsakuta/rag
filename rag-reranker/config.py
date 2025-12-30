import os

# ベースディレクトリの設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# システム全体で使用する定数の定義
DOCS_DIRECTORY = os.path.join(BASE_DIR, "manuals")
CACHE_DIRECTORY = os.path.join(BASE_DIR, "cache")
PROMPT_DIRECTORY = os.path.join(BASE_DIR, "prompts")

# モデル関連の設定
MODEL_PROVIDER = "anthropic"  # 使用するモデルプロバイダー
MODEL_NAME = "claude-3-5-sonnet-20240620"  # 使用するモデル名
EMBEDDING_PROVIDER = "sentence_transformers"  # 使用する埋め込みモデルプロバイダー
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"  # 使用する埋め込みモデル名
RERANKER_MODEL = "hotchpotch/japanese-reranker-cross-encoder-large-v1"  # 使用するリランカーモデル名

# キャッシュファイルのパス
VECTORS_FILE = os.path.join(CACHE_DIRECTORY, "vectors.pkl")
CHUNKS_FILE = os.path.join(CACHE_DIRECTORY, "chunks.pkl")
METADATA_FILE = os.path.join(CACHE_DIRECTORY, "metadata.json")
PARAMS_FILE = os.path.join(CACHE_DIRECTORY, "search_params.json")

# デフォルトのプロンプトテンプレートファイル
DEFAULT_PROMPT_FILE = os.path.join(PROMPT_DIRECTORY, "v1.txt")

# 必要なディレクトリの作成
for directory in [DOCS_DIRECTORY, CACHE_DIRECTORY, PROMPT_DIRECTORY]:
    if not os.path.exists(directory):
        os.makedirs(directory)