# --- searcher.py ---
import os
from typing import List, Dict, Any, Optional
from sudachipy import Dictionary, tokenizer
from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import retry, stop_after_attempt, wait_exponential

from config import SearchConfig
from src.utils.logger import setup_logger
from src.utils.auth import create_llm, create_embedding_model
from src.utils.dynamic_db_manager import DynamicDBManager, DynamicDBError
from src.utils.base_embedding import BaseEmbeddingModel

logger = setup_logger(__name__)

class Searcher:
    """メタデータ対応ハイブリッド検索クラス

    依存性注入により、テスト時にモックを注入可能。
    """

    # パフォーマンス: Sudachi辞書をクラス変数として共有（メモリ節約）
    # スレッドセーフ: ロックをクラス定義時に初期化（Race Condition防止）
    import threading as _threading
    _shared_tokenizer = None
    _tokenizer_lock = _threading.Lock()

    @classmethod
    def _get_shared_tokenizer(cls):
        """スレッドセーフな共有トークナイザーを取得"""
        if cls._shared_tokenizer is None:
            with cls._tokenizer_lock:
                if cls._shared_tokenizer is None:  # Double-checked locking
                    cls._shared_tokenizer = Dictionary().create()
                    logger.info("Sudachi辞書を共有インスタンスとして初期化")
        return cls._shared_tokenizer

    def __init__(
        self,
        config: SearchConfig,
        db_manager: Optional[DynamicDBManager] = None,
        embedding_model: Optional[BaseEmbeddingModel] = None
    ):
        """Searcherを初期化

        Args:
            config: 検索設定
            db_manager: 動的DB管理システム（省略時は自動生成）
            embedding_model: 埋め込みモデル（省略時は設定に応じて自動生成）
        """
        self.config = config
        # パフォーマンス: 共有トークナイザーを使用
        self.tokenizer = self._get_shared_tokenizer()
        self.mode = tokenizer.Tokenizer.SplitMode.C

        # 依存性注入: 外部から渡されなければ設定に応じて自動生成
        self.model = embedding_model or create_embedding_model(config)
        self.db_manager = db_manager or DynamicDBManager(config)

        self.current_db_path = None
        self.current_business_area = None

        # プロンプトファイルのキャッシュ（パフォーマンス向上）
        self._summarize_prompt_cache: Optional[str] = None

        # パフォーマンス: キーワードキャッシュ（N+1問題解消）
        self._reference_keywords_cache: Dict[int, set] = {}

        logger.info("Searcherを初期化しました（依存性注入対応）")

        # LLM初期化（条件付き：LLM拡張検索または多段階検索が有効な場合）
        needs_llm = self.config.search_mode in ["llm_enhanced", "multi_stage"]
        if needs_llm:
            self.llm = create_llm(self.config)
            logger.info(f"LLM initialized for {self.config.search_mode} search mode")
        else:
            self.llm = None
            logger.info("LLM not initialized - using original search mode")

    def _extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
        """キーワード抽出（KeywordSearchEngineに委譲）

        Deprecated: 新規コードはKeywordSearchEngineを直接使用してください。
        後方互換性のため残していますが、内部でKeywordSearchEngineを使用します。
        """
        if not hasattr(self, '_keyword_engine'):
            from src.core.search.keyword_search_engine import KeywordSearchEngine
            self._keyword_engine = KeywordSearchEngine(
                stop_words=self.config.STOP_WORDS,
                position_weight=self.config.POSITION_WEIGHT
            )
        return self._keyword_engine.extract_keywords(text, top_k)

    def _calculate_keyword_similarity(self, query_keywords: List[str], reference_text: str) -> float:
        """キーワード類似度を計算（KeywordSearchEngineに委譲）

        Deprecated: 新規コードはKeywordSearchEngineを直接使用してください。

        Args:
            query_keywords: クエリから抽出されたキーワードリスト
            reference_text: 参照テキスト

        Returns:
            float: 0.0〜1.0の類似度スコア
        """
        if not hasattr(self, '_keyword_engine'):
            from src.core.search.keyword_search_engine import KeywordSearchEngine
            self._keyword_engine = KeywordSearchEngine(
                stop_words=self.config.STOP_WORDS,
                position_weight=self.config.POSITION_WEIGHT
            )
        return self._keyword_engine.calculate_similarity(query_keywords, reference_text)

    def _load_summarize_prompt(self) -> str:
        """検索クエリ生成用のプロンプトファイルを読み込む（キャッシュ対応・パストラバーサル防止）"""
        if self._summarize_prompt_cache is not None:
            return self._summarize_prompt_cache

        from pathlib import Path

        prompt_dir = os.path.join(self.config.base_dir, "prompt")
        prompt_dir_resolved = Path(prompt_dir).resolve()
        summarize_prompt_file = (prompt_dir_resolved / "summarize_v1.0.txt").resolve()

        # セキュリティ: パストラバーサル防止（Path.relative_to() で検証）
        try:
            summarize_prompt_file.relative_to(prompt_dir_resolved)
        except ValueError:
            raise ValueError(f"Path traversal attempt blocked: summarize_v1.0.txt")

        if not summarize_prompt_file.exists():
            raise FileNotFoundError(f"Summarize prompt file not found: {summarize_prompt_file}")

        logger.info(f"Using summarize prompt file: summarize_v1.0.txt")

        with open(summarize_prompt_file, 'r', encoding='utf-8') as f:
            self._summarize_prompt_cache = f.read()
        return self._summarize_prompt_cache

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

    def summarize_text(self, text: str, fallback_on_error: bool = True) -> str:
        """LLMを使用してテキストを要約して検索クエリを生成

        すべてのLLMプロバイダーでLangchainの統一されたinvoke()メソッドを使用。
        リトライロジック付きで一時的なAPI障害に対応。

        Args:
            text: 要約対象のテキスト
            fallback_on_error: Trueの場合、エラー時に元のテキストを返す（デフォルト: True）

        Returns:
            str: 要約されたテキスト、またはエラー時は元のテキスト
        """
        if self.llm is None:
            raise RuntimeError("LLM is not initialized. Set search_mode to 'llm_enhanced' in config.")

        prompt_template = self._load_summarize_prompt()

        # 統一されたLangchainインターフェースを使用
        messages = [
            SystemMessage(content=prompt_template),
            HumanMessage(content=text)
        ]
        try:
            response = self._invoke_llm_with_retry(messages)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error during summarization after retries: {str(e)}")
            if fallback_on_error:
                # 品質: LLMエラー時に元のテキストを返すフォールバック
                logger.warning(f"LLM API error - falling back to original text")
                return text
            else:
                logger.info("LLM API error - stopping processing as configured")
                raise


    def prepare_search(self, reference_data):
        """検索の準備（メタデータ対応ベクトルDB + キャッシュ）"""
        from src.utils.vector_db import MetadataVectorDB

        self.reference_texts = reference_data['combined_texts']  # 結合テキストをベクトル化対象に
        self.reference_queries = reference_data['queries']  # 個別の質問（表示用）
        self.reference_answers = reference_data['answers']
        self.reference_metadatas = reference_data.get('metadatas', [])

        # 安全性: reference_metadatasの長さがreference_textsと一致するか確認（Index Out of Bounds防止）
        if len(self.reference_metadatas) < len(self.reference_texts):
            missing_count = len(self.reference_texts) - len(self.reference_metadatas)
            logger.warning(f"reference_metadatas is shorter than reference_texts by {missing_count} items. Padding with empty dicts.")
            self.reference_metadatas.extend([{} for _ in range(missing_count)])

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
                    # 安全性: インデックスチェック付きでメタデータにアクセス
                    if i < len(self.reference_metadatas):
                        filtered_metadatas.append(self.reference_metadatas[i])
                    else:
                        filtered_metadatas.append({})

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

        # ベクトル化処理は検索時に必要な業務分野のみ実行される
        # （_select_db_for_business内でneeds_updateをチェック）

        # パフォーマンス: キーワードを事前計算してキャッシュ（N+1問題解消）
        logger.info("キーワードキャッシュを構築中...")
        self._reference_keywords_cache = {}
        for i, query in enumerate(self.reference_queries):
            self._reference_keywords_cache[i] = set(self._extract_keywords(query))
        logger.info(f"キーワードキャッシュ構築完了: {len(self._reference_keywords_cache)}件")

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
        """検索を実行（SearchStrategyパターンに委譲）

        Args:
            input_number: 入力番号
            query_text: 検索クエリテキスト
            original_answer: 元の回答
            input_file: 入力ファイル名（動的DB選択用）

        Returns:
            list: 検索結果のリスト
        """
        # 動的DB選択（キーワードフィルタ以外）
        if self.config.search_type != "keyword_filter":
            self._select_db_if_needed(input_file)

        # SearchStrategyに委譲
        from src.core.search.search_strategy import create_strategy
        strategy = create_strategy(self)
        return strategy.execute(input_number, query_text, original_answer)

    def _select_db_if_needed(self, input_file: Optional[str]) -> None:
        """入力ファイルに基づいて動的にDBを選択

        Args:
            input_file: 入力ファイル名

        Raises:
            DynamicDBError: input_fileがNoneでvector_dbも未初期化の場合
        """
        if not input_file:
            if self.vector_db is None:
                raise DynamicDBError(
                    "VectorDB not initialized. input_file is required for dynamic DB selection, "
                    "or prepare_search() must be called first."
                )
            logger.info("  No input_file provided, using existing DB.")
            return

        try:
            business_area = self.db_manager.extract_business_area_from_input(input_file)

            # 前回と同じ業務分野ならDB再選択をスキップ
            if business_area == self.current_business_area and self.vector_db is not None:
                logger.info(f"  Selected DB for business area: {business_area}")
                return

            self._select_db_for_business(business_area)
            self.current_business_area = business_area
            logger.info(f"  Selected DB for business area: {business_area}")
        except DynamicDBError as e:
            logger.error(f"  DB選択エラー: {e}")
            raise

    def _build_source_filter(self) -> Dict[str, str]:
        """検索対象設定に基づいてソースフィルタを構築

        Returns:
            Dict: ソースフィルタ
        """
        return {"source": self.config.search_source}

    def _execute_vector_search(self, query_for_vector: str) -> List[Dict[str, Any]]:
        """ベクトル検索を実行

        Args:
            query_for_vector: ベクトル化する検索クエリ

        Returns:
            List[Dict]: 検索結果のリスト

        Raises:
            DynamicDBError: VectorDBが初期化されていない場合
        """
        if self.vector_db is None:
            raise DynamicDBError("VectorDB not initialized. Call prepare_search() and _select_db_for_business() first.")

        # 検索対象フィルタを構築
        filter_metadata = self._build_source_filter()
        if filter_metadata:
            logger.info(f"  Search source filter: {filter_metadata}")

        query_vector = self.model.encode([query_for_vector], normalize_embeddings=True)[0]
        search_results = self.vector_db.search(
            query_embedding=query_vector,
            n_results=self.config.top_k * self.config.VECTOR_SEARCH_MULTIPLIER,
            filter_metadata=filter_metadata
        )
        logger.info(f"  Vector search returned {len(search_results)} results")

        # 検索結果のソース分布
        source_counts: Dict[str, int] = {}
        for result in search_results:
            source = result['metadata'].get('source', 'unknown')
            source_counts[source] = source_counts.get(source, 0) + 1
        logger.info(f"  Search results by source: {source_counts}")

        return search_results

    def _calculate_keyword_similarities(
        self, search_results: List[Dict[str, Any]], keywords: List[str]
    ) -> List[float]:
        """検索結果に対するキーワード類似度を計算

        Args:
            search_results: 検索結果のリスト
            keywords: 抽出されたキーワード

        Returns:
            List[float]: 各結果のキーワード類似度
        """
        keyword_similarities = []
        query_keywords_set = set(keywords)

        for search_result in search_results:
            doc_id = search_result['id']
            if doc_id.startswith('doc_'):
                original_idx = int(doc_id.split('_')[1])
            else:
                original_idx = int(doc_id)

            # パフォーマンス: キーワードキャッシュを活用（N+1問題解消）
            if original_idx in self._reference_keywords_cache:
                ref_keywords = self._reference_keywords_cache[original_idx]
                # キャッシュを使用した高速なJaccard類似度計算
                if not ref_keywords or not query_keywords_set:
                    keyword_sim = 0.0
                else:
                    intersection = ref_keywords.intersection(query_keywords_set)
                    union = ref_keywords.union(query_keywords_set)
                    keyword_sim = len(intersection) / len(union) if union else 0.0
            else:
                # キャッシュミス時はフォールバック（警告付き）
                logger.warning(f"Keyword cache miss for index {original_idx}")
                if original_idx < len(self.reference_queries):
                    ref_query = self.reference_queries[original_idx]
                    keyword_sim = self._calculate_keyword_similarity(keywords, ref_query)
                else:
                    # DB/参照データ不整合: ドキュメントテキストから直接キーワード抽出して類似度計算
                    logger.warning(
                        f"Vector DB document {doc_id} (index {original_idx}) exceeds "
                        f"reference data ({len(self.reference_queries)} items). "
                        f"Using document text for keyword matching."
                    )
                    doc_text = search_result.get('document', '')
                    if doc_text and query_keywords_set:
                        doc_keywords = set(self._extract_keywords(doc_text))
                        intersection = doc_keywords.intersection(query_keywords_set)
                        union = doc_keywords.union(query_keywords_set)
                        keyword_sim = len(intersection) / len(union) if union else 0.0
                    else:
                        keyword_sim = 0.0

            keyword_similarities.append(keyword_sim)

        return keyword_similarities

    def _build_result_data(
        self, search_result: Dict[str, Any], combined_score: float
    ) -> Dict[str, Any]:
        """単一の検索結果からresult_dataを構築

        Args:
            search_result: 検索結果
            combined_score: 統合スコア

        Returns:
            Dict: 結果データ
        """
        metadata = search_result['metadata']
        combined_text = search_result['document']
        parsed_text = self.parse_enhanced_combined_text(combined_text)

        # 階層構造 + 質問を表示
        if metadata.get('source') == 'scenario':
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
            search_result_query = parsed_text['query']
            search_result_answer = parsed_text['answer']

        # シナリオIDを生成（シート名_行番号）
        sheet_name = metadata.get('sheet_name', '')
        row_index = metadata.get('row_index', '')
        scenario_id = f"{sheet_name}_{row_index}" if sheet_name and row_index != '' else ''

        return {
            'Input_Number': '',
            'Original_Query': '',
            'Original_Answer': '',
            'Search_Query': '',
            'Search_Result_Q': search_result_query,
            'Search_Result_A': search_result_answer,
            'Similarity': combined_score,
            'Scenario_ID': scenario_id,
            'Sheet_Name': sheet_name,
            'Row_Index': row_index,
            'Vector_Weight': self.config.vector_weight,
            'Top_K': self.config.top_k
        }

    def _calculate_and_merge_scores(
        self, search_results: List[Dict[str, Any]], keywords: List[str]
    ) -> List[Dict[str, Any]]:
        """スコアを計算してマージ

        Args:
            search_results: 検索結果のリスト
            keywords: 抽出されたキーワード

        Returns:
            List[Dict]: スコア付きの結果リスト
        """
        keyword_similarities = self._calculate_keyword_similarities(search_results, keywords)

        results = []
        max_similarity = 0.0

        logger.info(f"  === 検索結果処理開始 ===")
        logger.info(f"  検索結果数: {len(search_results)}")

        for i, search_result in enumerate(search_results):
            logger.debug(f"  処理中: ループカウンタ i={i}, 総検索結果数={len(search_results)}")

            keyword_sim = keyword_similarities[i]
            vector_sim = search_result['similarity']
            combined_score = (
                self.config.vector_weight * vector_sim +
                self.config.keyword_weight * keyword_sim
            )
            combined_score = max(0.0, min(1.0, combined_score))  # 0〜1にクリップ
            max_similarity = max(max_similarity, combined_score)

            result_data = self._build_result_data(search_result, combined_score)

            # 詳細ログ出力（デバッグレベル）
            logger.debug(f"  【結果{i+1}】Similarity: {combined_score:.4f}")

            results.append(result_data)

        logger.info(f"  === 検索結果処理完了 ===")
        logger.info(f"  作成された結果数: {len(results)}")
        logger.info(f"  【第1段階】最大類似度: {max_similarity:.4f}")

        return results

    def _format_final_results(
        self,
        results: List[Dict[str, Any]],
        input_number: str,
        query_text: str,
        original_answer: str,
        search_query: str
    ) -> List[Dict[str, Any]]:
        """最終結果をフォーマット

        Args:
            results: スコア付き結果リスト
            input_number: 入力番号
            query_text: 元のクエリ
            original_answer: 元の回答
            search_query: 使用した検索クエリ

        Returns:
            List[Dict]: フォーマット済み最終結果
        """
        # スコアでソートして上位を返す
        results.sort(key=lambda x: x['Similarity'], reverse=True)

        # top_k件に制限
        logger.info(f"  制限前の結果数: {len(results)}")
        results = results[:self.config.top_k]
        logger.info(f"  制限後の結果数: {len(results)}")

        # 1位のみに質問情報を設定
        if results:
            results[0]['Input_Number'] = input_number
            results[0]['Original_Query'] = query_text
            results[0]['Original_Answer'] = original_answer
            results[0]['Search_Query'] = search_query
            logger.info(f"  1位の結果に質問情報を設定: Input_Number={input_number}")
            logger.debug(f"  Search_Query set to: {search_query[:50]}...")

        logger.info(f"  Final results: {len(results)} items (limited to top_k={self.config.top_k})")

        # 各結果のInput_Numberを確認（デバッグレベル）
        for j, result in enumerate(results):
            logger.debug(f"    最終結果{j+1}: Input_Number='{result.get('Input_Number', 'MISSING')}'")

        return results

    def _ensure_db_updated_for_business(self, business_area: str) -> None:
        """特定の業務分野のDBを必要に応じて更新

        Args:
            business_area: 更新対象の業務分野名
        """
        if not self.db_manager:
            logger.warning("db_managerが未設定のため、DB更新をスキップ")
            return

        try:
            # 参照ファイルを業務分野ごとに分析
            all_business_areas = self.db_manager.analyze_reference_files()

            if business_area not in all_business_areas:
                logger.warning(f"業務分野 '{business_area}' の参照ファイルが見つかりません")
                return

            # 指定された業務分野のみ更新
            files = all_business_areas[business_area]
            logger.info(f"業務分野 '{business_area}' のDB更新チェック中...")
            self.db_manager.update_business_db(business_area, files)

        except DynamicDBError as e:
            logger.error(f"動的DB更新エラー: {e}")
            raise

    def _select_db_for_business(self, business_area: str):
        """業務分野に対応するDBを選択（必要に応じて更新）"""
        if not self.db_manager:
            return

        try:
            # この業務分野のDB更新が必要かチェックして更新
            self._ensure_db_updated_for_business(business_area)

            db_path = self.db_manager.get_db_path_for_business(business_area)

            # プロバイダー別のコレクション名を取得
            collection_name = self.db_manager._get_collection_name(business_area)

            # ChromaDBクライアントの切り替え（階層構造対応: db_pathを直接指定）
            from src.utils.vector_db import MetadataVectorDB
            self.vector_db = MetadataVectorDB(
                collection_name=collection_name,
                db_path=db_path
            )

            # コレクションの有効性チェック（ドキュメント数の確認）
            info = self.vector_db.get_collection_info()
            if info['document_count'] == 0:
                logger.error(f"コレクション '{collection_name}' にドキュメントがありません。DB更新を実行してください。")
                raise DynamicDBError(f"コレクション '{collection_name}' が空です。参照データのベクトル化が必要です。")

            self.current_db_path = db_path
            self.current_business_area = business_area

            logger.info(f"DB切り替え完了: {collection_name} (業務分野: {business_area}, プロバイダー: {self.config.embedding_provider}, ドキュメント数: {info['document_count']})")

        except DynamicDBError as e:
            logger.error(f"DB選択エラー: {e}")
            raise
