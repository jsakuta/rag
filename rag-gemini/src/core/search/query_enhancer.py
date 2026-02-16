# --- src/core/search/query_enhancer.py ---
"""クエリ拡張エンジン

LLMを使用した検索クエリの拡張・要約。
"""

import os
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class QueryEnhancer:
    """クエリ拡張エンジン

    LLMを使用して検索クエリを要約・拡張する。

    Attributes:
        llm: LangChain LLMインスタンス
        prompt_cache: プロンプトのキャッシュ
    """

    def __init__(self, llm, base_dir: str = "."):
        """QueryEnhancerを初期化

        Args:
            llm: LangChain LLMインスタンス
            base_dir: プロンプトファイルの基準ディレクトリ
        """
        self.llm = llm
        self.base_dir = base_dir
        self._prompt_cache: Optional[str] = None

        logger.info("QueryEnhancerを初期化しました")

    def _load_prompt(self) -> str:
        """プロンプトファイルを読み込む（キャッシュ対応）

        Returns:
            str: プロンプト文字列

        Raises:
            FileNotFoundError: プロンプトファイルが見つからない場合
        """
        if self._prompt_cache is not None:
            return self._prompt_cache

        from pathlib import Path

        prompt_dir = os.path.join(self.base_dir, "prompt")
        prompt_dir_resolved = Path(prompt_dir).resolve()
        summarize_prompt_file = (prompt_dir_resolved / "summarize_v1.0.txt").resolve()

        # セキュリティ: パストラバーサル防止
        try:
            summarize_prompt_file.relative_to(prompt_dir_resolved)
        except ValueError:
            raise ValueError("Path traversal attempt blocked: summarize_v1.0.txt")

        if not summarize_prompt_file.exists():
            raise FileNotFoundError(f"Summarize prompt file not found: {summarize_prompt_file}")

        logger.info("Using summarize prompt file: summarize_v1.0.txt")

        with open(summarize_prompt_file, 'r', encoding='utf-8') as f:
            self._prompt_cache = f.read()

        return self._prompt_cache

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def _invoke_llm_with_retry(self, messages: list):
        """LLM呼び出しをリトライロジック付きで実行

        Args:
            messages: LLMに送信するメッセージリスト

        Returns:
            LLMレスポンス

        Raises:
            Exception: 3回のリトライ後も失敗した場合
        """
        return self.llm.invoke(messages)

    def enhance(self, text: str, fallback_on_error: bool = True) -> str:
        """テキストを要約して検索クエリを生成

        LLMを使用して入力テキストを要約し、検索に適したクエリを生成。
        リトライロジック付きで一時的なAPI障害に対応。

        Args:
            text: 要約対象のテキスト
            fallback_on_error: Trueの場合、エラー時に元のテキストを返す

        Returns:
            str: 要約されたテキスト、またはエラー時は元のテキスト

        Raises:
            RuntimeError: LLMが初期化されていない場合
            Exception: fallback_on_error=Falseでエラーが発生した場合
        """
        if self.llm is None:
            raise RuntimeError("LLM is not initialized.")

        prompt_template = self._load_prompt()

        messages = [
            SystemMessage(content=prompt_template),
            HumanMessage(content=text)
        ]

        try:
            response = self._invoke_llm_with_retry(messages)
            enhanced_query = response.content.strip()
            logger.info(f"  Generated search query: {enhanced_query[:100]}...")
            return enhanced_query
        except Exception as e:
            logger.error(f"Error during query enhancement after retries: {str(e)}")
            if fallback_on_error:
                logger.warning("LLM API error - falling back to original text")
                return text
            else:
                logger.info("LLM API error - stopping processing as configured")
                raise

