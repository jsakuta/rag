# 関連ファイル一覧

## 修正対象

### rag-batch/config.py (全文)

```python
# --- config.py ---
from dataclasses import dataclass, field
import os
from typing import Dict, Any
import yaml
from utils.logger import setup_logger  # ← 問題: src.utils.logger に修正必要

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
```

## 参照ファイル（インポートパス確認用）

### rag-batch/main.py

```python
# --- main.py ---
import sys
import os
from dotenv import load_dotenv
from config import SearchConfig
from src.core.processor import Processor  # ← src.* 形式を使用
from src.utils.logger import setup_logger  # ← src.* 形式を使用

# 環境変数の読み込み
load_dotenv()
logger = setup_logger(__name__)

def main():
    # 設定の初期化
    config = SearchConfig(base_dir=os.path.dirname(os.path.abspath(__file__)))

    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        logger.info("Starting in interactive mode")
        config.is_interactive = True
        config.vector_weight = config.DEFAULT_UI_VECTOR_WEIGHT
        mode = "chat"
        os.system("streamlit run ui/chat.py")
    else:
        logger.info("Starting in batch mode")
        config.is_interactive = False
        mode = "batch"
        processor = Processor(config)
        processor.process_data(mode=mode)

if __name__ == "__main__":
    main()
```

### rag-batch/src/handlers/input_handler.py (関連部分)

```python
class InputHandler:
    def __init__(self, config: SearchConfig):
        self.config = config
        self.input_dir = os.path.join(config.base_dir, "input")  # ← input/ が必要
        self.reference_dir = os.path.join(config.base_dir, "reference")
```

## 比較対象（問題なしのファイル）

### rag-streamlit/config.py

```python
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
    DEFAULT_VECTOR_WEIGHT: float = 0.9
    DEFAULT_UI_VECTOR_WEIGHT: float = 0.9

    top_k: int = DEFAULT_TOP_K
    model_name: str = DEFAULT_MODEL_NAME
    llm_provider: str = DEFAULT_LLM_PROVIDER
    llm_model: str = DEFAULT_LLM_MODEL
    vector_weight: float = DEFAULT_VECTOR_WEIGHT
    base_dir: str = "."
    is_interactive: bool = False

    def __post_init__(self):
        if not 0 <= self.vector_weight <= 1:
            raise ValueError("vector_weight must be between 0 and 1")
        self.base_dir = os.path.abspath(self.base_dir)

    @property
    def keyword_weight(self) -> float:
        return 1.0 - self.vector_weight

    def get_param_summary(self) -> str:
        mode = "chat" if self.is_interactive else "batch"
        return f"{mode}_v{self.vector_weight:.1f}_k{self.keyword_weight:.1f}"
```

## フォルダ構造

```
rag/
├── rag-batch/
│   ├── config.py          # 修正対象
│   ├── main.py
│   ├── input/             # 作成が必要（現在存在しない）
│   ├── output/
│   ├── reference/
│   └── src/
│       ├── core/
│       ├── handlers/
│       └── utils/
│           └── logger.py
├── rag-streamlit/         # 問題なし
├── rag-reranker/          # Deprecated
├── rag-gemini/            # 改善済み
└── _archive/              # 旧フォルダ
```
