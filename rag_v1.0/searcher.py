# --- searcher.py ---
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sudachipy import Dictionary, tokenizer
from collections import Counter
from config import SearchConfig
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
import os
from utils.logger import setup_logger
from langchain.schema import HumanMessage, SystemMessage
import json # 追加
from datetime import datetime

logger = setup_logger(__name__)

class Searcher:
    def __init__(self, config: SearchConfig):
        self.config = config
        self.tokenizer = Dictionary().create()
        self.mode = tokenizer.Tokenizer.SplitMode.C
        self.model = SentenceTransformer(config.model_name)
        self.llm = self._setup_llm()
        self.reference_vectors = None
        self.reference_texts = None
        self.reference_df = None # processor.pyから移動

    def _setup_llm(self):
        if self.config.llm_provider == "anthropic":
            return ChatAnthropic(
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
                model=self.config.llm_model,
                temperature=0
            )
        elif self.config.llm_provider == "openai":
            return ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                model=self.config.llm_model,
                temperature=0
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.llm_provider}")

    def _extract_keywords(self, text: str, top_k: int = 5) -> list[str]:
        morphemes = self.tokenizer.tokenize(text, self.mode)
        keywords = []
        for m in morphemes:
            if m.part_of_speech()[0] == '名詞':
                important_types = ['固有名詞', '一般']
                weight = 2 if m.part_of_speech()[1] in important_types else 1
                word = m.dictionary_form()
                if len(word) > 1:
                    keywords.extend([word] * weight)

        stop_words = {'こと', 'もの', 'これ', 'それ', 'ところ', '方','する', 'ある', 'いる', 'れる', 'られる', 'なる', 'その', 'これ', 'それ'}
        filtered_words = {word: count for word, count in Counter(keywords).items() if word not in stop_words}
        return [word for word, _ in Counter(filtered_words).most_common(top_k)]

    def _calculate_keyword_similarity(self, query_keywords: list[str], reference_text: str) -> float:
        ref_keywords = set(self._extract_keywords(reference_text))
        query_keywords_set = set(query_keywords)
        if not ref_keywords or not query_keywords_set:
            return 0.0

        intersection = ref_keywords.intersection(query_keywords_set)
        union = ref_keywords.union(query_keywords_set)
        position_weight = 1.2
        weighted_score = sum(position_weight if reference_text.find(kw) < len(reference_text) // 2 else 1 for kw in intersection)
        normalized_score = weighted_score / (len(union) * position_weight)
        return min(normalized_score, 1.0)

    def _load_latest_prompt(self) -> str:
        """最新のプロンプトファイルを読み込む"""
        prompt_dir = os.path.join(self.config.base_dir, "prompt")
        prompt_files = [f for f in os.listdir(prompt_dir) if os.path.isfile(os.path.join(prompt_dir, f))]
        if not prompt_files:
            raise FileNotFoundError(f"No prompt files found in {prompt_dir}")

        latest_prompt_file = max(prompt_files, key=lambda f: os.path.getctime(os.path.join(prompt_dir, f)))
        logger.info(f"Using prompt file: {latest_prompt_file}")

        with open(os.path.join(prompt_dir, latest_prompt_file), 'r', encoding='utf-8') as f:
            return f.read()

    def summarize_text(self, text: str) -> str:
        """LLMを使用してテキストを要約"""
        prompt_template = self._load_latest_prompt()
        messages = [
            SystemMessage(content=prompt_template),
            HumanMessage(content=text)
        ]

        try:
            response = self.llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error during summarization: {str(e)}")
            return text

    def prepare_search(self, reference_data):
        """検索の準備（ベクトル化、キャッシュ）"""
        import pandas as pd
        self.reference_texts = reference_data['queries']
        self.reference_df = pd.DataFrame({'問合せ内容': reference_data['queries'], '回答': reference_data['answers']}) # processorから移動
        
        # キャッシュディレクトリの作成（存在しない場合）
        cache_dir = os.path.join(self.config.base_dir, "reference", "vector_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        cache_file = os.path.join(cache_dir, "cache.json")

        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                if cache_data['texts'] == self.reference_texts:
                    self.reference_vectors = np.array(cache_data['vectors'])
                    logger.info("Loaded cached vectors.")
                    return
                else:
                    logger.info("Reference texts have changed. Regenerating vectors.")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        else:
            logger.info("No cache file found. Creating new cache.")

        # キャッシュがない場合や有効でない場合はベクトル化
        self.reference_vectors = self.model.encode(self.reference_texts, normalize_embeddings=True)
        cache_data = {
            'vectors': self.reference_vectors.tolist(),
            'texts': self.reference_texts,
            'timestamp': datetime.now().isoformat()
        }
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Cached reference vectors to {cache_file}")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}. Check permissions or disk space.")

    def search(self, input_number: str, query_text: str, original_answer: str) -> list:
        """ハイブリッド検索を実行"""
        summarized_text = self.summarize_text(query_text)
        logger.info(f"Row (No.{input_number}):")
        logger.info(f"  Original query: {query_text[:100]}...")
        logger.info(f"  Original answer: {original_answer[:100]}..." if original_answer else "  No original answer")
        logger.info(f"  Summarized query: {summarized_text}")
        keywords = self._extract_keywords(query_text)
        logger.info(f"  Extracted keywords: {keywords}")

        keyword_similarities = [
            self._calculate_keyword_similarity(keywords, ref_text)
            for ref_text in self.reference_texts
        ]

        query_vector = self.model.encode([summarized_text], normalize_embeddings=True)
        vector_similarities = cosine_similarity(query_vector, self.reference_vectors)[0]

        combined_scores = (
            self.config.vector_weight * vector_similarities +
            self.config.keyword_weight * np.array(keyword_similarities)
        )

        top_indices = np.argsort(combined_scores)[-self.config.top_k:][::-1]
        results = []
        for i, ref_idx in enumerate(top_indices):
            similarity = combined_scores[ref_idx]
            results.append({
                'Input_Number': input_number if i == 0 else '',
                'Original_Query': query_text if i == 0 else '',
                'Original_Answer': original_answer if i == 0 else '',
                'Summarized_Query': summarized_text if i == 0 else '',
                'Search_Result_Q': self.reference_df.iloc[ref_idx]['問合せ内容'],
                'Search_Result_A': self.reference_df.iloc[ref_idx]['回答'],
                'Similarity': similarity,
                'Vector_Weight': self.config.vector_weight,
                'Top_K': self.config.top_k
            })
            logger.info(f"  Added result with similarity: {similarity:.4f}")

        return results