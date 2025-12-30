from dataclasses import dataclass
import os

@dataclass
class SearchConfig:
    """検索設定を管理するデータクラス"""
    # デフォルト設定の一元管理
    DEFAULT_TOP_K: int = 4
    DEFAULT_MODEL_NAME: str = "intfloat/multilingual-e5-base"
    DEFAULT_LLM_PROVIDER: str = "anthropic"
    DEFAULT_LLM_MODEL: str = "claude-3-5-sonnet-20241022"
    DEFAULT_VECTOR_WEIGHT: float = 0.9  # バッチ処理用のデフォルト値
    DEFAULT_UI_VECTOR_WEIGHT: float = 0.9 # UI用のデフォルト値

    top_k: int = DEFAULT_TOP_K
    model_name: str = DEFAULT_MODEL_NAME
    llm_provider: str = DEFAULT_LLM_PROVIDER
    llm_model: str = DEFAULT_LLM_MODEL
    vector_weight: float = DEFAULT_VECTOR_WEIGHT
    base_dir: str = "."
    is_interactive: bool = False  # モード設定を追加

    def __post_init__(self):
        """パラメータの検証"""
        if not 0 <= self.vector_weight <= 1:
            raise ValueError("vector_weight must be between 0 and 1")
        self.base_dir = os.path.abspath(self.base_dir)

    @property
    def keyword_weight(self) -> float:
        """キーワード検索の重みを自動計算"""
        return 1.0 - self.vector_weight

    def get_param_summary(self) -> str:
        """パラメータのサマリー文字列を生成"""
        mode = "chat" if self.is_interactive else "batch"
        return f"{mode}_v{self.vector_weight:.1f}_k{self.keyword_weight:.1f}"