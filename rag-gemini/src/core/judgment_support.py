# --- judgment_support.py ---
"""LLMによる判断支援モジュール（人間の意思決定を支援）"""
import os
from typing import List, Dict, Any, Optional
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import retry, stop_after_attempt, wait_exponential

from config import SearchConfig
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

LLM_PROVIDERS = {
    "anthropic": ("ANTHROPIC_API_KEY", ChatAnthropic, "anthropic_api_key"),
    "openai": ("OPENAI_API_KEY", ChatOpenAI, "api_key"),
    "gemini": ("GOOGLE_API_KEY", ChatGoogleGenerativeAI, "google_api_key"),
}


class JudgmentSupport:
    """LLMを使用して人間の判断を支援するクラス（関連性判定・根拠・修正案提示）"""

    def __init__(self, config: SearchConfig):
        self.config = config
        self._prompt_cache: Optional[str] = None

        if config.multi_stage_enable_judgment_support:
            self.llm = self._setup_llm()
            logger.info("JudgmentSupport: LLM initialized")
        else:
            self.llm = None
            logger.info("JudgmentSupport: LLM judgment support disabled")

    def _setup_llm(self):
        """LLM設定メソッド"""
        provider = self.config.llm_provider
        if provider not in LLM_PROVIDERS:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        env_key, llm_class, api_param = LLM_PROVIDERS[provider]
        api_key = os.getenv(env_key)
        if not api_key:
            raise ValueError(f"{env_key} environment variable is not set")

        return llm_class(**{api_param: api_key, "model": self.config.llm_model, "temperature": 0})

    def _load_prompt(self) -> str:
        """判断支援用プロンプトを読み込む（キャッシュ対応）"""
        if self._prompt_cache is not None:
            return self._prompt_cache

        prompt_path = os.path.join(
            self.config.base_dir,
            self.config.judgment_support_prompt_path
        )
        if not os.path.exists(prompt_path):
            raise FileNotFoundError(
                f"判断支援プロンプトファイルが見つかりません: {prompt_path}\n"
                "prompt/judgment_support.txt を作成してください。"
            )

        with open(prompt_path, 'r', encoding='utf-8') as f:
            self._prompt_cache = f.read()
        logger.info(f"Loaded prompt from: {prompt_path}")
        return self._prompt_cache

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    def _invoke_llm_with_retry(self, messages: list):
        """LLM呼び出しをリトライロジック付きで実行"""
        return self.llm.invoke(messages)

    def evaluate(
        self, revision_content: str, search_result_q: str, search_result_a: str
    ) -> Dict[str, str]:
        """単一の検索結果に対する関連性評価を実行"""
        if self.llm is None:
            return {
                "relevance_judgment": "判断支援無効",
                "judgment_reason": "",
                "modification_suggestion": ""
            }

        user_message = f"""【改定内容】
{revision_content}

【検索結果（既存QA）】
質問: {search_result_q}
回答: {search_result_a}

上記の改定内容と検索結果の関連性を判定し、判断材料を提供してください。"""

        messages = [
            SystemMessage(content=self._load_prompt()),
            HumanMessage(content=user_message)
        ]

        try:
            response = self._invoke_llm_with_retry(messages)
            return self._parse_response(response.content)
        except Exception as e:
            logger.error(f"Judgment support error: {e}")
            return {
                "relevance_judgment": "エラー",
                "judgment_reason": f"評価エラー: {str(e)[:50]}",
                "modification_suggestion": ""
            }

    def _parse_response(self, response_text: str) -> Dict[str, str]:
        """LLMレスポンスをパース"""
        result = {
            "relevance_judgment": "",
            "judgment_reason": "",
            "modification_suggestion": ""
        }
        field_map = {
            "関連性:": "relevance_judgment",
            "根拠:": "judgment_reason",
            "修正案:": "modification_suggestion"
        }

        current_field = None
        current_content = []

        for line in response_text.strip().split('\n'):
            line = line.strip()
            matched = False
            for prefix, field_name in field_map.items():
                if line.startswith(prefix):
                    if current_field and current_content:
                        result[current_field] = "\n".join(current_content).strip()
                    current_field = field_name
                    current_content = [line.replace(prefix, "").strip()]
                    matched = True
                    break
            if not matched and current_field:
                current_content.append(line)

        if current_field and current_content:
            result[current_field] = "\n".join(current_content).strip()

        return result

    def evaluate_batch(
        self, results: List[Dict[str, Any]], revision_content: str
    ) -> List[Dict[str, Any]]:
        """バッチで関連性評価を実行"""
        if not self.config.multi_stage_enable_judgment_support:
            logger.info("LLM判断支援は無効です。スキップします。")
            for result in results:
                result['Relevance_Judgment'] = ""
                result['Judgment_Reason'] = ""
                result['Modification_Suggestion'] = ""
            return results

        logger.info(f"=== LLM判断支援開始 ({len(results)}件) ===")

        for i, result in enumerate(results):
            logger.debug(f"  評価中: {i+1}/{len(results)}")
            evaluation = self.evaluate(
                revision_content,
                result.get('Search_Result_Q', ''),
                result.get('Search_Result_A', '')
            )
            result['Relevance_Judgment'] = evaluation['relevance_judgment']
            result['Judgment_Reason'] = evaluation['judgment_reason']
            result['Modification_Suggestion'] = evaluation['modification_suggestion']

        logger.info("=== LLM判断支援完了 ===")
        return results
