# --- impact_analyzer.py ---
"""LLMによる影響分析モジュール"""
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


class ImpactAnalyzer:
    """LLMを使用して改定内容の影響分析を行うクラス"""

    def __init__(self, config: SearchConfig):
        self.config = config
        self._prompt_cache: Optional[str] = None

        if config.multi_stage_enable_llm_analysis:
            self.llm = self._setup_llm()
            logger.info("ImpactAnalyzer: LLM initialized")
        else:
            self.llm = None
            logger.info("ImpactAnalyzer: LLM analysis disabled")

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
        """影響分析用プロンプトを読み込む（キャッシュ対応）"""
        if self._prompt_cache is not None:
            return self._prompt_cache

        prompt_path = os.path.join(self.config.base_dir, "prompt", "impact_analysis_v1.0.txt")
        if not os.path.exists(prompt_path):
            logger.warning(f"Impact analysis prompt not found: {prompt_path}")
            return self._get_default_prompt()

        with open(prompt_path, 'r', encoding='utf-8') as f:
            self._prompt_cache = f.read()
        logger.info(f"Loaded prompt from: {prompt_path}")
        return self._prompt_cache

    def _get_default_prompt(self) -> str:
        """デフォルトの影響分析プロンプト"""
        return """あなたは業務システムの影響分析エキスパートです。
改定内容が既存のFAQやシナリオに与える影響を分析し、修正が必要な場合は具体的な修正案を提示します。

【出力形式】
影響の根拠: <影響を与える理由を1-2文で説明>
修正案: <回答の修正案を具体的に提案>

【注意】影響がない場合は「影響の根拠: 影響なし」「修正案: 修正不要」と出力すること
"""

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    def _invoke_llm_with_retry(self, messages: list):
        """LLM呼び出しをリトライロジック付きで実行"""
        return self.llm.invoke(messages)

    def analyze_impact(
        self, revision_content: str, search_result_q: str, search_result_a: str
    ) -> Dict[str, str]:
        """単一の検索結果に対する影響分析を実行"""
        if self.llm is None:
            return {"impact_reason": "LLM分析無効", "modification_suggestion": ""}

        user_message = f"""【改定内容】
{revision_content}

【検索結果（既存QA）】
質問: {search_result_q}
回答: {search_result_a}

上記の改定内容が、この検索結果に影響を与えるか分析してください。"""

        messages = [
            SystemMessage(content=self._load_prompt()),
            HumanMessage(content=user_message)
        ]

        try:
            response = self._invoke_llm_with_retry(messages)
            return self._parse_response(response.content)
        except Exception as e:
            logger.error(f"Impact analysis error: {e}")
            return {"impact_reason": f"分析エラー: {str(e)[:50]}", "modification_suggestion": ""}

    def _parse_response(self, response_text: str) -> Dict[str, str]:
        """LLMレスポンスをパース"""
        result = {"impact_reason": "", "modification_suggestion": ""}
        field_map = {"影響の根拠:": "impact_reason", "修正案:": "modification_suggestion"}

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

    def analyze_batch(
        self, results: List[Dict[str, Any]], revision_content: str
    ) -> List[Dict[str, Any]]:
        """バッチで影響分析を実行"""
        if not self.config.multi_stage_enable_llm_analysis:
            logger.info("LLM影響分析は無効です。スキップします。")
            for result in results:
                result['Impact_Reason'] = ""
                result['Modification_Suggestion'] = ""
            return results

        logger.info(f"=== LLM影響分析開始 ({len(results)}件) ===")

        for i, result in enumerate(results):
            logger.debug(f"  分析中: {i+1}/{len(results)}")
            analysis = self.analyze_impact(
                revision_content,
                result.get('Search_Result_Q', ''),
                result.get('Search_Result_A', '')
            )
            result['Impact_Reason'] = analysis['impact_reason']
            result['Modification_Suggestion'] = analysis['modification_suggestion']

        logger.info("=== LLM影響分析完了 ===")
        return results
