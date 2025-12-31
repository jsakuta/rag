import logging
from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sudachipy import Dictionary
from sudachipy import tokenizer
from collections import Counter
from config import SearchConfig
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
import os
from src.utils.utils import setup_logger

logger = setup_logger(__name__)

class HybridSearchMixin:
    def __init__(self, config: SearchConfig):
        self.config = config
        # Sudachiの初期化
        self.tokenizer = Dictionary().create()
        # デフォルトでモードCを使用（最も長い単位で分割）
        self.mode = tokenizer.Tokenizer.SplitMode.C
        self.model = SentenceTransformer(config.model_name)
        self.llm = self._setup_llm()

    def _setup_llm(self):
        """LLMの設定"""
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
    
    def _extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
        """Sudachiを使用してテキストから重要なキーワードを抽出"""
        # 形態素解析の実行
        morphemes = self.tokenizer.tokenize(text, self.mode)

        # 重要な品詞のみを抽出
        keywords = []
        for m in morphemes:
            pos = m.part_of_speech()
            # 名詞のみを抽出対象とする
            if pos[0] == '名詞':
                # 重要な名詞の種類を定義
                important_types = ['固有名詞', '一般']
                if pos[1] in important_types:
                    weight = 2  # 重要な名詞は重み付けを2倍に
                else:
                    weight = 1
                
                word = m.dictionary_form()
                if len(word) > 1:  # 1文字の名詞は除外
                    keywords.extend([word] * weight)
        
        # TF-IDFスコアを考慮したキーワード抽出
        word_counts = Counter(keywords)
        
        # ストップワードの除去（名詞用にカスタマイズ）
        stop_words = {'こと', 'もの', 'これ', 'それ', 'ところ', '方','する', 'ある', 'いる', 'れる', 'られる', 'なる', 'その', 'これ', 'それ'}
        filtered_words = {
            word: count for word, count in word_counts.items()
            if word not in stop_words
        }
        
        return [word for word, _ in Counter(filtered_words).most_common(top_k)]

    def _calculate_keyword_similarity(self, query_keywords: List[str], reference_text: str) -> float:
        """キーワードベースの類似度を計算（改良版）"""
        # リファレンステキストからキーワードを抽出
        ref_keywords = set(self._extract_keywords(reference_text))
        query_keywords_set = set(query_keywords)
        
        if not ref_keywords or not query_keywords_set:
            return 0.0
        
        # 重み付きJaccard類似度の計算
        intersection = ref_keywords.intersection(query_keywords_set)
        union = ref_keywords.union(query_keywords_set)
        
        # キーワードの位置も考慮した重み付け
        position_weight = 1.2  # 文章の前半に出現するキーワードの重み
        
        weighted_score = 0
        for keyword in intersection:
            # キーワードが文章の前半に出現する場合、重みを増やす
            if reference_text.find(keyword) < len(reference_text) // 2:
                weighted_score += position_weight
            else:
                weighted_score += 1
        
        normalized_score = weighted_score / (len(union) * position_weight)
        return min(normalized_score, 1.0)  # スコアを0-1の範囲に正規化

    def _get_hybrid_search_results(self, 
                                 query_text: str, 
                                 summarized_text: str,
                                 reference_texts: List[str], 
                                 reference_vectors: np.ndarray,
                                 top_k: int = 3) -> List[Tuple[int, float]]:
        """ハイブリッド検索を実行し、結果を返す（改良版）"""
        # キーワード抽出（元のクエリテキストから）
        keywords = self._extract_keywords(query_text)
        logger.info(f"  Extracted keywords: {keywords}")
        
        # キーワードベースの類似度を計算
        keyword_similarities = [
            self._calculate_keyword_similarity(keywords, ref_text)
            for ref_text in reference_texts
        ]
        
        # ベクトルベースの類似度を計算（要約テキストを使用）
        query_vector = self.model.encode([summarized_text], normalize_embeddings=True)
        vector_similarities = cosine_similarity(query_vector, reference_vectors)[0]
        
        # 設定された重みでスコアを組み合わせ
        combined_scores = (
            self.config.vector_weight * vector_similarities + 
            self.config.keyword_weight * np.array(keyword_similarities)
        )
        
        # 上位k件のインデックスとスコアを取得
        top_indices = np.argsort(combined_scores)[-top_k:][::-1]
        results = [(idx, combined_scores[idx]) for idx in top_indices]
        
        logger.info(f"  Top {top_k} hybrid similarities: {[score for _, score in results]}")
        logger.info(f"  Using weights - Vector: {self.config.vector_weight:.2f}, Keyword: {self.config.keyword_weight:.2f}")
        
        return results