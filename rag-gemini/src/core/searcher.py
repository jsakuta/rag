# --- searcher.py ---
import os
import re
import json
import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from sudachipy import Dictionary, tokenizer
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
import vertexai
from google.oauth2 import service_account
from langchain.schema import HumanMessage, SystemMessage

from config import SearchConfig
from src.utils.logger import setup_logger
from src.utils.dynamic_db_manager import DynamicDBManager, DynamicDBError
from src.utils.auth import initialize_vertex_ai

logger = setup_logger(__name__)

class Searcher:
    def __init__(self, config: SearchConfig):
        self.config = config
        self.tokenizer = Dictionary().create()
        self.mode = tokenizer.Tokenizer.SplitMode.C
        from src.utils.gemini_embedding import GeminiEmbeddingModel
        self.model = GeminiEmbeddingModel(config)
        
        # 動的DB管理システムの初期化
        self.db_manager = DynamicDBManager(config)
        self.current_db_path = None
        self.current_business_area = None
        logger.info("動的DB管理システムを初期化しました")
        
        # LLM初期化（条件付き：LLM拡張検索が有効な場合のみ）
        if self.config.search_mode == "llm_enhanced" and self.config.enable_query_enhancement:
            self.llm = self._setup_llm()
            logger.info("LLM initialized for enhanced search mode")
        else:
            self.llm = None
            logger.info("LLM not initialized - using original search mode")

    def _setup_llm(self):
        """LLM設定メソッド（LLM拡張検索用）"""
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
        elif self.config.llm_provider == "gemini":
            # Vertex AI Gemini用の認証設定
            try:
                initialize_vertex_ai(self.config)

                # 正しく動作している環境の方式でモデル作成
                from vertexai.generative_models import GenerativeModel
                model = GenerativeModel(self.config.llm_model)
                chat = model.start_chat()
                
                logger.info("Gemini authentication configured successfully")
                return chat
            except Exception as e:
                logger.error(f"Failed to configure Gemini authentication: {e}")
                raise
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.llm_provider}")

    def _extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
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

    def _calculate_keyword_similarity(self, query_keywords: List[str], reference_text: str) -> float:
        ref_keywords = set(self._extract_keywords(reference_text))
        query_keywords_set = set(query_keywords)
        if not ref_keywords or not query_keywords_set:
            return 0.0

        intersection = ref_keywords.intersection(query_keywords_set)
        union = ref_keywords.union(query_keywords_set)
        position_weight = self.config.POSITION_WEIGHT
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
    
    def _load_summarize_prompt(self) -> str:
        """検索クエリ生成用のプロンプトファイル（summarize_v1.0.txt）を読み込む"""
        prompt_dir = os.path.join(self.config.base_dir, "prompt")
        summarize_prompt_file = os.path.join(prompt_dir, "summarize_v1.0.txt")
        
        if not os.path.exists(summarize_prompt_file):
            raise FileNotFoundError(f"Summarize prompt file not found: {summarize_prompt_file}")
        
        logger.info(f"Using summarize prompt file: summarize_v1.0.txt")
        
        with open(summarize_prompt_file, 'r', encoding='utf-8') as f:
            return f.read()

    def summarize_text(self, text: str) -> str:
        """LLMを使用してテキストを要約して検索クエリを生成"""
        if self.llm is None:
            raise RuntimeError("LLM is not initialized. Set search_mode to 'llm_enhanced' in config.")
        
        prompt_template = self._load_summarize_prompt()
        
        # Vertex AI Gemini用のメッセージ形式調整
        if self.config.llm_provider == "gemini":
            # Vertex AIは単一テキスト形式
            full_prompt = f"{prompt_template}\n\n{text}"
            try:
                response = self.llm.send_message(full_prompt)
                return response.text.strip()
            except Exception as e:
                logger.error(f"Error during summarization: {str(e)}")
                logger.info("LLM API error - stopping processing as configured")
                raise
        else:
            # 他のLLM用
            messages = [
                SystemMessage(content=prompt_template),
                HumanMessage(content=text)
            ]
            try:
                response = self.llm.invoke(messages)
                return response.content.strip()
            except Exception as e:
                logger.error(f"Error during summarization: {str(e)}")
                logger.info("LLM API error - stopping processing as configured")
                raise


    def prepare_search(self, reference_data):
        """検索の準備（メタデータ対応ベクトルDB + キャッシュ）"""
        from utils.vector_db import MetadataVectorDB
        
        self.reference_texts = reference_data['combined_texts']  # 結合テキストをベクトル化対象に
        self.reference_queries = reference_data['queries']  # 個別の質問（表示用）
        self.reference_answers = reference_data['answers']
        self.reference_metadatas = reference_data.get('metadatas', [])
        
        # デバッグ: 空のテキストをチェック
        logger.info(f"Total reference texts: {len(self.reference_texts)}")
        empty_texts = []
        for i, text in enumerate(self.reference_texts):
            if not text or not text.strip():
                empty_texts.append(i)
                logger.warning(f"Empty text found at index {i}: '{text}'")
        
        if empty_texts:
            logger.error(f"Found {len(empty_texts)} empty texts at indices: {empty_texts}")
            # 空のテキストを除外
            filtered_texts = []
            filtered_queries = []
            filtered_answers = []
            filtered_metadatas = []
            
            for i in range(len(self.reference_texts)):
                if i not in empty_texts:
                    filtered_texts.append(self.reference_texts[i])
                    filtered_queries.append(self.reference_queries[i])
                    filtered_answers.append(self.reference_answers[i])
                    filtered_metadatas.append(self.reference_metadatas[i])
            
            self.reference_texts = filtered_texts
            self.reference_queries = filtered_queries
            self.reference_answers = filtered_answers
            self.reference_metadatas = filtered_metadatas
            
            logger.info(f"Filtered to {len(self.reference_texts)} valid texts")
        
        # メタデータ対応ベクトルDBの初期化
        # 動的DB管理システムでは、初期化時にコレクションを指定しない
        # 実際のDB選択は search メソッドで行われる
        self.vector_db = None  # 初期化時はNone、検索時に適切なコレクションを選択
        logger.info("動的DB管理システム用に初期化（コレクションは検索時に選択）")
        
        # データ変更チェックは動的DB管理システムで行われるため、ここではスキップ
        logger.info("データ変更チェックは動的DB管理システムで実行済み")
        
        # データ変更チェックは動的DB管理システムで行われるため、ここではスキップ
        logger.info("ベクトル化処理は動的DB管理システムで実行済み")

    def parse_enhanced_combined_text(self, combined_text: str) -> dict:
        """階層構造を含む結合テキストを解析（新形式：ラベル付き）"""
        # 新形式の解析：「分類: 階層 | 質問: 質問内容 | 回答: 回答内容」
        hierarchy = ""
        query = ""
        answer = ""
        
        # 「|」で分割
        parts = combined_text.split(" | ")
        
        for part in parts:
            part = part.strip()
            if part.startswith("分類: "):
                hierarchy = part[3:].strip()  # "分類: "を除去
            elif part.startswith("質問: "):
                query = part[3:].strip()  # "質問: "を除去
            elif part.startswith("回答: "):
                answer = part[3:].strip()  # "回答: "を除去
        
        return {
            'hierarchy': hierarchy,
            'query': query,
            'answer': answer
        }

    # _build_filter_metadataメソッドを削除（タグレス対応）

    def search(self, input_number: str, query_text: str, original_answer: str, input_file: str = None) -> list:
        """メタデータ対応ハイブリッド検索を実行（LLM拡張検索対応・動的DB選択対応）"""
        
        # 動的DB選択（入力ファイルが指定されている場合）
        if input_file:
            try:
                business_area = self.db_manager.extract_business_area_from_input(input_file)
                self._select_db_for_business(business_area)
                logger.info(f"  Selected DB for business area: {business_area}")
            except DynamicDBError as e:
                logger.error(f"  DB選択エラー: {e}")
                raise
        
        # 検索方式の詳細ログ
        logger.info(f"Row (No.{input_number}):")
        logger.info(f"  Search mode: {self.config.search_mode}")
        logger.info(f"  Query enhancement enabled: {self.config.enable_query_enhancement}")
        logger.info(f"  Original query: {query_text[:100]}...")
        logger.info(f"  Original answer: {original_answer[:100]}..." if original_answer else "  No original answer")
        
        # 検索クエリの生成（検索方式による分岐）
        if self.config.search_mode == "llm_enhanced" and self.config.enable_query_enhancement:
            # LLM拡張検索：LLMで検索クエリを生成
            logger.info("  Using LLM-enhanced search mode")
            search_query = self.summarize_text(query_text)
            logger.info(f"  Generated search query: {search_query}")
            query_for_vector = search_query
        else:
            # 原文検索：質問文をそのまま使用
            logger.info("  Using original search mode")
            search_query = query_text
            query_for_vector = query_text
        
        # キーワード抽出（原文から抽出）
        keywords = self._extract_keywords(query_text)
        logger.info(f"  Extracted keywords: {keywords}")
        
        # メタデータフィルタリングを無効化（タグレス対応）
        filter_metadata = None
        
        # ベクトル検索（第1段階）- 検索クエリをベクトル化
        query_vector = self.model.encode([query_for_vector], normalize_embeddings=True)[0]
        search_results = self.vector_db.search(
            query_embedding=query_vector,
            n_results=self.config.top_k * self.config.VECTOR_SEARCH_MULTIPLIER,
            filter_metadata=filter_metadata
        )
        logger.info(f"  Vector search returned {len(search_results)} results")
        
        # 検索結果のソース分布
        source_counts = {}
        for result in search_results:
            source = result['metadata'].get('source', 'unknown')
            source_counts[source] = source_counts.get(source, 0) + 1
        logger.info(f"  Search results by source: {source_counts}")
        
        # キーワード類似度計算（検索結果に対して）
        keyword_similarities = []
        for search_result in search_results:
            # 元のインデックスを取得
            doc_id = search_result['id']
            if doc_id.startswith('doc_'):
                original_idx = int(doc_id.split('_')[1])
            else:
                original_idx = int(doc_id)
            
            ref_query = self.reference_queries[original_idx]
            keyword_sim = self._calculate_keyword_similarity(keywords, ref_query)
            keyword_similarities.append(keyword_sim)
        
        # 結果の統合とスコアリング（第1段階）
        results = []
        max_similarity = 0.0
        
        logger.info(f"  === 検索結果処理開始 ===")
        logger.info(f"  検索結果数: {len(search_results)}")
        
        for i, search_result in enumerate(search_results):
            logger.info(f"  処理中: ループカウンタ i={i}, 総検索結果数={len(search_results)}")
            doc_id = search_result['id']
            if doc_id.startswith('doc_'):
                original_idx = int(doc_id.split('_')[1])
            else:
                original_idx = int(doc_id)
            
            keyword_sim = keyword_similarities[i]
            vector_sim = search_result['similarity']
            combined_score = (
                self.config.vector_weight * vector_sim +
                self.config.keyword_weight * keyword_sim
            )
            max_similarity = max(max_similarity, combined_score)
            
            metadata = search_result['metadata']
            tags = metadata.get('tags', "")
            # tagsはDB保存時に " | " 連結された文字列。表示用に階層' > 'へ整形
            if isinstance(tags, str) and tags.strip():
                # Lv0:... Lv1:... のLvを取り出して ' > ' で連結
                lv_parts = []
                for part in tags.split(" "):
                    part = part.strip()
                    if part.startswith("Lv") and ":" in part:
                        lv_parts.append(part.split(":", 1)[1])
                category_info = " > ".join(lv_parts) if lv_parts else ""
            else:
                category_info = ""
            
            combined_text = search_result['document']
            parsed_text = self.parse_enhanced_combined_text(combined_text)
            
            # E列: 階層構造 + 質問を表示
            if metadata.get('source') == 'scenario':
                # シナリオデータの場合、階層構造 + 質問を表示
                hierarchy = metadata.get('hierarchy', '')
                query = parsed_text['query']
                if hierarchy and query:
                    search_result_query = f"{hierarchy} > {query}"
                elif hierarchy:
                    search_result_query = hierarchy
                else:
                    search_result_query = query
                search_result_answer = parsed_text['answer']
            else:
                # 履歴データの場合、質問のみを表示
                search_result_query = parsed_text['query']
                search_result_answer = parsed_text['answer']
            
            # 結果データの作成（scenarioとhistory_data共通）
            result_data = {
                'Input_Number': '',  # ソート後に1位のみ設定
                'Original_Query': '',  # ソート後に1位のみ設定
                'Original_Answer': '',  # ソート後に1位のみ設定
                'Search_Query': '',  # ソート後に1位のみ設定
                'Search_Result_Q': search_result_query,
                'Search_Result_A': search_result_answer,
                'Similarity': combined_score,
                'Vector_Weight': self.config.vector_weight,
                'Top_K': self.config.top_k
            }
            
            # 詳細ログ出力（デバッグレベル）
            logger.debug(f"  【結果{i+1}】(i={i})")
            logger.debug(f"    Input_Number: {result_data['Input_Number']}")
            logger.debug(f"    Original_Query: {result_data['Original_Query'][:50]}...")
            logger.debug(f"    Original_Answer: {result_data['Original_Answer'][:50]}...")
            logger.debug(f"    Search_Query: {result_data['Search_Query'][:50]}...")
            logger.debug(f"    Search_Result_Q: {result_data['Search_Result_Q'][:50]}...")
            logger.debug(f"    Search_Result_A: {result_data['Search_Result_A'][:50]}...")
            logger.debug(f"    Similarity: {result_data['Similarity']}")
            logger.debug(f"    i == 0? {i == 0}")
            
            results.append(result_data)
        
        logger.info(f"  === 検索結果処理完了 ===")
        logger.info(f"  作成された結果数: {len(results)}")
        logger.info(f"  最大類似度: {max_similarity}")
        
        # 第1段階の結果を評価
        logger.info(f"  【第1段階】最大類似度: {max_similarity:.4f}")
        
        # 段階的フィルタリングを削除（タグレス対応）
        # 第2段階処理は削除済み

        
        # スコアでソートして上位を返す
        results.sort(key=lambda x: x['Similarity'], reverse=True)
        
        # top_k件に制限
        logger.info(f"  制限前の結果数: {len(results)}")
        results = results[:self.config.top_k]
        logger.info(f"  制限後の結果数: {len(results)}")
        
        # ソート後、1位のみに質問情報を設定（関数引数を直接使用）
        if results:
            results[0]['Input_Number'] = input_number
            results[0]['Original_Query'] = query_text
            results[0]['Original_Answer'] = original_answer
            results[0]['Search_Query'] = search_query  # 実際に使用した検索クエリを設定
            logger.info(f"  1位の結果に質問情報を設定: Input_Number={input_number}")
            logger.info(f"  Search_Query set to: {search_query[:50]}...")
        
        logger.info(f"  Final results: {len(results)} items (limited to top_k={self.config.top_k})")
        
        # 各結果のInput_Numberを確認
        for j, result in enumerate(results):
            logger.info(f"    最終結果{j+1}: Input_Number='{result.get('Input_Number', 'MISSING')}'")
        return results

    def _select_db_for_business(self, business_area: str):
        """業務分野に対応するDBを選択"""
        if not self.db_manager:
            return
        
        try:
            db_path = self.db_manager.get_db_path_for_business(business_area)
            
            # 日本語の業務分野名を英語に変換
            english_name = self.db_manager._translate_business_area(business_area)
            
            # ChromaDBクライアントの切り替え
            from utils.vector_db import MetadataVectorDB
            self.vector_db = MetadataVectorDB(
                base_dir=self.config.base_dir,
                collection_name=f"{english_name}_DB"
            )
            
            self.current_db_path = db_path
            self.current_business_area = business_area
            
            logger.info(f"DB切り替え完了: {english_name}_DB (業務分野: {business_area})")
            
        except DynamicDBError as e:
            logger.error(f"DB選択エラー: {e}")
            raise