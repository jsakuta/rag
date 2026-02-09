# --- config.py ---
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import yaml
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def load_settings(section: Optional[str] = None) -> Dict[str, Any]:
    """settings.yamlを読み込み、指定セクションの設定を返す

    Args:
        section: 読み込むセクション名 ("ui", "batch", "evaluation")
                 Noneの場合は全設定を返す

    Returns:
        commonセクションと指定セクションをマージした辞書
        sectionがNoneの場合は全設定辞書
    """
    settings_path = Path(__file__).parent / "config" / "settings.yaml"

    if not settings_path.exists():
        logger.warning(f"設定ファイルが見つかりません: {settings_path}")
        return {}

    with open(settings_path, "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)

    if settings is None:
        return {}

    if section is None:
        return settings

    # commonセクションと指定セクションをマージ
    common = settings.get("common", {})
    section_settings = settings.get(section, {})

    # 深いマージを行う（セクション設定がcommonを上書き）
    merged = _deep_merge(common.copy(), section_settings)
    return merged


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """辞書を再帰的にマージ（overrideがbaseを上書き）"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


# バッチ処理用のデフォルト設定をYAMLから読み込み
_batch_settings = load_settings("batch")
_common_settings = load_settings("common") if load_settings("common") else {}

@dataclass
class SearchConfig:
    """
    検索設定を管理するデータクラス
    """
    # デフォルト設定（YAMLから読み込み、フォールバック値付き）
    DEFAULT_TOP_K: int = _batch_settings.get("top_k", 4)
    DEFAULT_MODEL_NAME: str = "intfloat/multilingual-e5-base"
    DEFAULT_VECTOR_WEIGHT: float = _batch_settings.get("vector_weight", 0.9)  # バッチ処理用

    # 検索タイプ設定（類似検索 / キーワード必須）
    DEFAULT_SEARCH_TYPE: str = _common_settings.get("search_type", "hybrid")
    VALID_SEARCH_TYPES: Tuple[str, ...] = ("hybrid", "keyword_filter")

    # 検索方式設定（LLM拡張検索対応）
    DEFAULT_SEARCH_MODE: str = _common_settings.get("search_mode", "original")

    # 検索対象設定
    DEFAULT_SEARCH_SOURCE: str = _common_settings.get("search_source", "all")
    VALID_SEARCH_SOURCES: Tuple[str, ...] = ("all", "scenario", "history_data")

    # 多段階検索設定
    MULTI_STAGE_THRESHOLD: float = 0.45        # 統合スコアのしきい値
    MULTI_STAGE_MAX_RESULTS: int = 100        # 各検索の最大結果数

    # 両プロバイダー比較モード（多段階検索時のみ有効）
    DEFAULT_DUAL_PROVIDER_MODE: bool = False

    # 正解ID列の候補（YAMLから読み込み）
    CORRECT_ID_COLUMNS: Tuple[str, ...] = tuple(
        _common_settings.get("columns", {}).get("correct_id", ['正解ID', '正解', 'CorrectID', 'Expected'])
    )
    
    # 埋め込みモデル設定
    # 有効なプロバイダー: "vertex_ai" (Gemini), "azure_openai" (text-embedding-3-large)
    # 環境変数から読み込み（必須）
    VALID_EMBEDDING_PROVIDERS: Tuple[str, ...] = ("vertex_ai", "azure_openai")

    # 検索モードの有効な値（__post_init__で参照されるためクラス定数として先頭付近に配置）
    VALID_SEARCH_MODES: Tuple[str, ...] = ("original", "llm_enhanced", "multi_stage")
    
    # 動的DB管理設定
    DEFAULT_FORCE_DB_UPDATE: bool = False  # 強制DB更新フラグ

    # バッチサイズ設定
    # EMBEDDING_BATCH_SIZE: Vertex AI Embedding APIの1回あたりの最大テキスト数。
    # APIの上限は250件。パフォーマンス向上のため最大値に設定。
    EMBEDDING_BATCH_SIZE: int = 250
    # VECTOR_DB_BATCH_SIZE: ChromaDBへの一括書き込み時のバッチサイズ。
    # 大きすぎるとメモリエラーの原因になります。100-500が推奨範囲。
    VECTOR_DB_BATCH_SIZE: int = 100

    # 検索設定
    # VECTOR_SEARCH_MULTIPLIER: top_k に対する取得倍率。
    # リランキング用に多めに取得します。2-3が推奨。
    VECTOR_SEARCH_MULTIPLIER: int = 2
    # POSITION_WEIGHT: キーワードがテキスト前半にある場合の重み係数（YAMLから読み込み）
    POSITION_WEIGHT: float = _common_settings.get("keyword", {}).get("position_weight", 1.2)
    # STOP_WORDS: キーワード抽出時に除外する一般的な単語（YAMLから読み込み）
    STOP_WORDS: Tuple[str, ...] = tuple(
        _common_settings.get("keyword", {}).get("stop_words", ['こと', 'もの', 'これ', 'それ', 'ところ', '方', 'する', 'ある', 'いる', 'れる', 'られる', 'なる', 'その'])
    )

    # 列名候補（YAMLから読み込み）
    # Excel/CSVファイルから質問・回答・タグ列を自動検出する際の候補名。
    QUERY_COLUMN_CANDIDATES: Tuple[str, ...] = tuple(
        _common_settings.get("columns", {}).get("query", ['分割後質問', '問合せ内容', '質問内容', '問い合わせ', '質問', 'query', 'Query'])
    )
    ANSWER_COLUMN_CANDIDATES: Tuple[str, ...] = tuple(
        _common_settings.get("columns", {}).get("answer", ['分割後回答', '回答', '既存回答', 'answer', 'Answer'])
    )
    TAG_COLUMN_CANDIDATES: Tuple[str, ...] = tuple(
        _common_settings.get("columns", {}).get("tag", ['タグ付け', 'タグ', '分類', 'category', 'Category', 'tag', 'Tag'])
    )

    # 原則文判定マーカー
    # このテキストを含む回答は「原則文」として特別扱いされます。
    PRINCIPLE_MARKER: str = "以下の選択肢から選んでください"

    # ファイル名パターン（既存ファイル対応版）
    REFERENCE_FILE_PATTERN: str = r".*?([^_]+).*?(履歴データ|シナリオデータ).*?(\d{8})?.*?\.xlsx$"
    INPUT_FILE_PATTERN: str = r"^([^_]+)_(\d{8})\.xlsx$"

    top_k: int = DEFAULT_TOP_K
    model_name: str = DEFAULT_MODEL_NAME
    llm_provider: str = field(default_factory=lambda: os.getenv("DEFAULT_LLM_PROVIDER", ""))
    llm_model: str = field(default_factory=lambda: os.getenv("DEFAULT_LLM_MODEL", ""))
    vector_weight: float = DEFAULT_VECTOR_WEIGHT
    keyword_weight: float = field(init=False)  # keyword_weight は vector_weight から自動計算
    base_dir: str = "."
    input_type: str = "excel"  # 新規: 入力ファイル形式
    output_type: str = "excel" # 新規: 出力ファイル形式
    input_config: Dict[str, Any] = field(default_factory=dict)  # 新規: 入力設定
    output_config: Dict[str, Any] = field(default_factory=dict) # 新規: 出力設定
    
    # 検索タイプ設定
    search_type: str = DEFAULT_SEARCH_TYPE

    # 検索方式設定
    search_mode: str = DEFAULT_SEARCH_MODE

    # 検索対象設定
    search_source: str = DEFAULT_SEARCH_SOURCE

    # 多段階検索設定（インスタンス変数）
    multi_stage_threshold: float = MULTI_STAGE_THRESHOLD
    multi_stage_max_results: int = MULTI_STAGE_MAX_RESULTS
    multi_stage_enable_judgment_support: bool = True  # LLM判断支援の有効化
    judgment_support_prompt_path: str = "prompt/judgment_support.txt"  # 判断支援プロンプトファイル
    dual_provider_mode: bool = DEFAULT_DUAL_PROVIDER_MODE  # 両プロバイダー比較モード
    
    # 埋め込みモデル設定（環境変数から読み込み、未設定時はエラー）
    embedding_provider: str = field(default_factory=lambda: os.getenv("DEFAULT_EMBEDDING_PROVIDER", ""))
    embedding_model: str = field(default_factory=lambda: os.getenv("DEFAULT_EMBEDDING_MODEL", ""))
    
    # 動的DB管理設定
    force_db_update: bool = DEFAULT_FORCE_DB_UPDATE  # 強制DB更新フラグ
    
    # 参照データ形式設定
    reference_type: str = "multi_folder"  # "excel", "hierarchical_excel", "multi_folder"
    include_hierarchy_in_vector: bool = True  # 階層情報をベクトル化に含めるかどうか
    
    # Vertex AI設定（セキュリティ向上: ハードコードされたプロジェクトIDを削除）
    gemini_credentials_path: str = field(default_factory=lambda: os.getenv("GEMINI_CREDENTIALS_PATH", "gemini_credentials.json"))
    gemini_project_id: str = field(default_factory=lambda: os.getenv("GEMINI_PROJECT_ID", ""))
    gemini_location: str = field(default_factory=lambda: os.getenv("GEMINI_LOCATION", "us-central1"))
    azure_key_vault_url: str = field(default_factory=lambda: os.getenv("AZURE_KEY_VAULT_URL", ""))
    azure_key_vault_scopes: str = field(default_factory=lambda: os.getenv("AZURE_KEY_VAULT_SCOPES", "https://www.googleapis.com/auth/cloud-platform"))

    # Azure OpenAI Embedding 設定
    azure_openai_embedding_endpoint: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_ENDPOINT", ""))
    azure_openai_embedding_api_key: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_API_KEY", ""))
    azure_openai_embedding_deployment: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"))
    azure_openai_embedding_api_version: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"))

    def __post_init__(self):
        """パラメータの検証とkeyword_weightの計算"""
        # Input Validation: 数値パラメータの範囲検証
        if not 0 <= self.vector_weight <= 1:
            raise ValueError("vector_weight must be between 0 and 1")
        self.keyword_weight = 1.0 - self.vector_weight
        self.base_dir = os.path.abspath(self.base_dir)

        # Input Validation: top_k（1以上の整数）
        if not isinstance(self.top_k, int) or self.top_k < 1:
            raise ValueError(f"top_k must be a positive integer, got: {self.top_k}")
        if self.top_k > 100:
            logger.warning(f"top_k={self.top_k} is very large, this may impact performance")

        # Input Validation: 多段階検索パラメータ
        if not 0 <= self.multi_stage_threshold <= 1:
            raise ValueError(f"multi_stage_threshold must be between 0 and 1, got: {self.multi_stage_threshold}")
        if not isinstance(self.multi_stage_max_results, int) or self.multi_stage_max_results < 1:
            raise ValueError(f"multi_stage_max_results must be a positive integer, got: {self.multi_stage_max_results}")
        if self.multi_stage_max_results > 1000:
            logger.warning(f"multi_stage_max_results={self.multi_stage_max_results} is very large, this may impact performance")

        # Input Validation: バッチサイズ
        if self.EMBEDDING_BATCH_SIZE < 1 or self.EMBEDDING_BATCH_SIZE > 250:
            raise ValueError(f"EMBEDDING_BATCH_SIZE must be between 1 and 250, got: {self.EMBEDDING_BATCH_SIZE}")
        if self.VECTOR_DB_BATCH_SIZE < 1 or self.VECTOR_DB_BATCH_SIZE > 1000:
            raise ValueError(f"VECTOR_DB_BATCH_SIZE must be between 1 and 1000, got: {self.VECTOR_DB_BATCH_SIZE}")

        # Input Validation: 検索設定
        if self.VECTOR_SEARCH_MULTIPLIER < 1:
            raise ValueError(f"VECTOR_SEARCH_MULTIPLIER must be at least 1, got: {self.VECTOR_SEARCH_MULTIPLIER}")

        # 検索タイプの検証
        if self.search_type not in self.VALID_SEARCH_TYPES:
            raise ValueError(f"search_type must be one of {self.VALID_SEARCH_TYPES}")

        # 検索方式の検証
        if self.search_mode not in self.VALID_SEARCH_MODES:
            raise ValueError(f"search_mode must be one of {self.VALID_SEARCH_MODES}")

        # 検索対象の検証
        if self.search_source not in self.VALID_SEARCH_SOURCES:
            raise ValueError(f"search_source must be one of {self.VALID_SEARCH_SOURCES}")

        # 埋め込み設定の必須チェック（環境変数未設定時はエラー）
        if not self.embedding_provider:
            raise ValueError("DEFAULT_EMBEDDING_PROVIDER環境変数が設定されていません")
        if not self.embedding_model:
            raise ValueError("DEFAULT_EMBEDDING_MODEL環境変数が設定されていません")

        # 埋め込みプロバイダーの検証
        if self.embedding_provider not in self.VALID_EMBEDDING_PROVIDERS:
            raise ValueError(f"embedding_provider must be one of {self.VALID_EMBEDDING_PROVIDERS}")

        # LLM設定の必須チェック（環境変数未設定時はエラー）
        if not self.llm_provider:
            raise ValueError("DEFAULT_LLM_PROVIDER環境変数が設定されていません（gemini / anthropic / openai）")
        if not self.llm_model:
            raise ValueError("DEFAULT_LLM_MODEL環境変数が設定されていません")

        self._validate_embedding_config()  # 埋め込み設定の検証

    def _validate_embedding_config(self):
        """埋め込みモデル設定の検証（Vertex AI / Azure OpenAI）"""
        if self.embedding_provider == "vertex_ai":
            # Vertex AI: 認証情報ファイルの存在確認
            credentials_path = os.path.join(self.base_dir, self.gemini_credentials_path)
            if not os.path.exists(credentials_path):
                logger.warning(f"Vertex AI credentials file not found: {credentials_path}")
                logger.info("Please ensure GEMINI_CREDENTIALS_PATH is set correctly in .env file")

        elif self.embedding_provider == "azure_openai":
            # Azure OpenAI: 必須環境変数の確認
            if not self.azure_openai_embedding_endpoint:
                logger.warning("AZURE_OPENAI_ENDPOINT is not set")
                logger.info("Please set AZURE_OPENAI_ENDPOINT in .env file")
            if not self.azure_openai_embedding_api_key:
                logger.warning("AZURE_OPENAI_API_KEY is not set")
                logger.info("Please set AZURE_OPENAI_API_KEY in .env file")

    # 検索モードのフラグマッピング
    SEARCH_MODE_FLAGS = {
        "multi_stage": "ms",
        "llm_enhanced": "llm",
        "original": "orig"
    }

    def get_param_summary(self) -> str:
        """パラメータのサマリー文字列を生成（LLM拡張検索対応）"""
        hierarchy_flag = "h" if self.include_hierarchy_in_vector else "nh"
        search_flag = self.SEARCH_MODE_FLAGS.get(self.search_mode, "orig")
        return f"v{self.vector_weight:.1f}_k{self.keyword_weight:.1f}_{hierarchy_flag}_{search_flag}"
