# --- config.py ---
from dataclasses import dataclass, field
import os
from typing import Dict, Any, Tuple
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class SearchConfig:
    """
    検索設定を管理するデータクラス
    """
    # デフォルト設定
    DEFAULT_TOP_K: int = 4
    DEFAULT_MODEL_NAME: str = "intfloat/multilingual-e5-base"
    DEFAULT_LLM_PROVIDER: str = "anthropic"
    DEFAULT_LLM_MODEL: str = "claude-3-5-sonnet-20241022"
    DEFAULT_VECTOR_WEIGHT: float = 0.9  # バッチ処理用のデフォルト値
    DEFAULT_UI_VECTOR_WEIGHT: float = 0.7 # UI用のデフォルト値
    
    # 検索方式設定（LLM拡張検索対応）
    DEFAULT_SEARCH_MODE: str = "original"  # "original" | "llm_enhanced" | "multi_stage"
    DEFAULT_ENABLE_QUERY_ENHANCEMENT: bool = False

    # 多段階検索設定
    MULTI_STAGE_THRESHOLD: float = 0.5        # 統合スコアのしきい値
    MULTI_STAGE_MAX_RESULTS: int = 100        # 各検索の最大結果数
    
    # 埋め込みモデル設定
    DEFAULT_EMBEDDING_PROVIDER: str = "vertex_ai"
    DEFAULT_EMBEDDING_MODEL: str = "gemini-embedding-001"
    
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
    # POSITION_WEIGHT: キーワードがテキスト前半にある場合の重み係数。
    # 1.0より大きいと前半マッチを優先。1.0-1.5が推奨範囲。
    POSITION_WEIGHT: float = 1.2
    # STOP_WORDS: キーワード抽出時に除外する一般的な単語
    STOP_WORDS: Tuple[str, ...] = ('こと', 'もの', 'これ', 'それ', 'ところ', '方', 'する', 'ある', 'いる', 'れる', 'られる', 'なる', 'その')

    # 列名候補
    # Excel/CSVファイルから質問・回答・タグ列を自動検出する際の候補名。
    # 優先順位順に検索され、最初にマッチした列が使用されます。
    QUERY_COLUMN_CANDIDATES: Tuple[str, ...] = ('分割後質問', '問合せ内容', '質問内容', '問い合わせ', '質問', 'query', 'Query')
    ANSWER_COLUMN_CANDIDATES: Tuple[str, ...] = ('分割後回答', '回答', '既存回答', 'answer', 'Answer')
    TAG_COLUMN_CANDIDATES: Tuple[str, ...] = ('タグ付け', 'タグ', '分類', 'category', 'Category', 'tag', 'Tag')

    # 原則文判定マーカー
    # このテキストを含む回答は「原則文」として特別扱いされます。
    PRINCIPLE_MARKER: str = "以下の選択肢から選んでください"

    # ファイル名パターン（既存ファイル対応版）
    REFERENCE_FILE_PATTERN: str = r".*?([^_]+).*?(履歴データ|シナリオデータ).*?(\d{8})?.*?\.xlsx$"
    INPUT_FILE_PATTERN: str = r"^([^_]+)_(\d{8})\.xlsx$"

    top_k: int = DEFAULT_TOP_K
    model_name: str = DEFAULT_MODEL_NAME
    llm_provider: str = DEFAULT_LLM_PROVIDER
    llm_model: str = DEFAULT_LLM_MODEL
    vector_weight: float = DEFAULT_VECTOR_WEIGHT
    keyword_weight: float = field(init=False)  # keyword_weight は vector_weight から自動計算
    base_dir: str = "."
    input_type: str = "excel"  # 新規: 入力ファイル形式
    output_type: str = "excel" # 新規: 出力ファイル形式
    input_config: Dict[str, Any] = field(default_factory=dict)  # 新規: 入力設定
    output_config: Dict[str, Any] = field(default_factory=dict) # 新規: 出力設定
    
    # 検索方式設定
    search_mode: str = DEFAULT_SEARCH_MODE
    enable_query_enhancement: bool = DEFAULT_ENABLE_QUERY_ENHANCEMENT

    # 多段階検索設定（インスタンス変数）
    multi_stage_threshold: float = MULTI_STAGE_THRESHOLD
    multi_stage_max_results: int = MULTI_STAGE_MAX_RESULTS
    multi_stage_enable_llm_analysis: bool = True  # LLM影響分析の有効化
    
    # 埋め込みモデル設定
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    
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

    def __post_init__(self):
        """パラメータの検証とkeyword_weightの計算"""
        if not 0 <= self.vector_weight <= 1:
            raise ValueError("vector_weight must be between 0 and 1")
        self.keyword_weight = 1.0 - self.vector_weight
        self.base_dir = os.path.abspath(self.base_dir)
        
        # 検索方式の検証
        if self.search_mode not in ["original", "llm_enhanced", "multi_stage"]:
            raise ValueError("search_mode must be 'original', 'llm_enhanced', or 'multi_stage'")
            
        self._validate_vertex_ai_config() # 新規追加: Vertex AI設定の検証

    def _validate_vertex_ai_config(self):
        """Vertex AI設定の検証"""
        if self.embedding_provider == "vertex_ai":
            # 認証情報ファイルの存在確認
            credentials_path = os.path.join(self.base_dir, self.gemini_credentials_path)
            if not os.path.exists(credentials_path):
                logger.warning(f"Vertex AI credentials file not found: {credentials_path}")
                logger.info("Please ensure GEMINI_CREDENTIALS_PATH is set correctly in .env file")

    def get_param_summary(self) -> str:
        """パラメータのサマリー文字列を生成（LLM拡張検索対応）"""
        hierarchy_flag = "h" if self.include_hierarchy_in_vector else "nh"
        if self.search_mode == "multi_stage":
            search_flag = "ms"
        elif self.search_mode == "llm_enhanced":
            search_flag = "llm"
        else:
            search_flag = "orig"
        return f"v{self.vector_weight:.1f}_k{self.keyword_weight:.1f}_{hierarchy_flag}_{search_flag}"
