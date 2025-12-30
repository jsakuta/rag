# --- config.py ---
from dataclasses import dataclass, field
import os
from typing import Dict, Any
import yaml
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class SearchConfig:
    """検索設定を管理するデータクラス"""
    # デフォルト設定
    DEFAULT_TOP_K: int = 4
    DEFAULT_MODEL_NAME: str = "intfloat/multilingual-e5-base"
    DEFAULT_LLM_PROVIDER: str = "anthropic"
    DEFAULT_LLM_MODEL: str = "claude-3-5-sonnet-20241022"
    DEFAULT_VECTOR_WEIGHT: float = 0.9  # バッチ処理用のデフォルト値
    DEFAULT_UI_VECTOR_WEIGHT: float = 0.7 # UI用のデフォルト値

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

    def __post_init__(self):
        """パラメータの検証とkeyword_weightの計算"""
        if not 0 <= self.vector_weight <= 1:
            raise ValueError("vector_weight must be between 0 and 1")
        self.keyword_weight = 1.0 - self.vector_weight
        self.base_dir = os.path.abspath(self.base_dir)
        self._load_external_config() # 外部設定の読み込み

    def _load_external_config(self):
        """外部設定ファイル（config.yaml）から設定を読み込む"""
        config_file = os.path.join(self.base_dir, "config.yaml")
        if os.path.exists(config_file):
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    external_config = yaml.safe_load(f)

                # 外部設定で上書き
                for key, value in external_config.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
                        logger.info(f"Overriding config setting '{key}' with value from config.yaml")
            except Exception as e:
                logger.error(f"Error loading external config: {e}")

    def get_param_summary(self) -> str:
        """パラメータのサマリー文字列を生成"""
        return f"v{self.vector_weight:.1f}_k{self.keyword_weight:.1f}"