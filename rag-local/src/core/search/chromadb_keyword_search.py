"""ChromaDB全件取得 + キーワードマッチング

評価モード（rev*コレクション）と影響調査モード（naibujimu, smile）の
両方で使用する共通キーワード検索モジュール。

ChromaDB の collection.get() で全件取得し、Python側でキーワードマッチングを行う。
"""
import os
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings
from chromadb.errors import NotFoundError as ChromaNotFoundError

from src.core.search.keyword_search_engine import KeywordSearchEngine
from src.core.search.text_combiner import get_text_combiner

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """キーワードマッチ結果"""
    question: str
    answer: str
    hierarchy: str
    similarity: float
    match_count: int
    scenario_id: str
    sheet_name: str
    row_index: int
    collection_name: str
    source: str  # "scenario" or "history_data"

    @property
    def area(self) -> str:
        """コレクション名からarea名を返す"""
        return ChromaDBKeywordSearcher.extract_area(self.collection_name)


class ChromaDBKeywordSearcher:
    """ChromaDB全件取得 + キーワードマッチング

    Args:
        base_db_path: vector_db ディレクトリのパス（例: data/vector_db）
        keyword_engine: KeywordSearchEngine インスタンス
        area_to_bot: area名 → bot名マッピング
        area_to_category: area名 → 日本語カテゴリ名マッピング
    """

    def __init__(
        self,
        base_db_path: str,
        keyword_engine: KeywordSearchEngine,
        area_to_bot: Dict[str, str],
        area_to_category: Dict[str, str],
    ):
        self.base_db_path = base_db_path
        self.keyword_engine = keyword_engine
        self.area_to_bot = area_to_bot
        self.area_to_category = area_to_category
        self._text_combiner = get_text_combiner()
        self._collection_cache: Dict[Tuple[str, str], Tuple[list, list]] = {}

    def search(
        self,
        collection_names: List[str],
        query: str,
        provider: str = "azure_openai",
        max_results: int = 50,
        source_filter: Optional[str] = None,
    ) -> List[MatchResult]:
        """ChromaDBからキーワード検索を実行

        Args:
            collection_names: 検索対象コレクション名リスト
                評価モード: ["rev02_souzoku"] 等
                影響調査モード: ["naibujimu", "smile"] 等
            query: 検索クエリ
            provider: 埋め込みプロバイダー（DBディレクトリ構造用）
            max_results: 最大結果件数
            source_filter: データソースフィルタ（"scenario" | "history_data" | None=全て）

        Returns:
            MatchResult のリスト（マッチ数降順）
        """
        keywords = self.keyword_engine.extract_keywords(query)
        if not keywords:
            logger.info("キーワード抽出結果が空です")
            return []

        logger.info(f"抽出キーワード: {keywords}")
        all_results: List[MatchResult] = []

        for col_name in collection_names:
            results = self._search_collection(col_name, keywords, provider, source_filter)
            all_results.extend(results)

        all_results.sort(key=lambda r: r.match_count, reverse=True)
        return all_results[:max_results]

    def _get_collection_data(
        self, collection_name: str, provider: str
    ) -> Tuple[list, list]:
        """コレクションデータを取得（キャッシュ付き）"""
        cache_key = (collection_name, provider)
        if cache_key in self._collection_cache:
            return self._collection_cache[cache_key]

        db_path = os.path.join(self.base_db_path, collection_name, provider)
        if not os.path.exists(db_path):
            logger.warning(f"DBパスが存在しません: {db_path}")
            return [], []

        try:
            client = chromadb.PersistentClient(
                path=db_path,
                settings=Settings(anonymized_telemetry=False),
            )
            collection = client.get_collection("default")
        except ChromaNotFoundError:
            logger.warning(f"コレクションが見つかりません: {collection_name}/{provider}")
            return [], []
        except (ValueError, FileNotFoundError) as e:
            logger.warning(f"DB読み込みエラー ({collection_name}/{provider}): {e}")
            return [], []

        result = collection.get(include=["documents", "metadatas"])
        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])

        self._collection_cache[cache_key] = (documents, metadatas)
        return documents, metadatas

    def _search_collection(
        self,
        collection_name: str,
        keywords: List[str],
        provider: str,
        source_filter: Optional[str] = None,
    ) -> List[MatchResult]:
        """単一コレクションからキーワード検索"""
        documents, metadatas = self._get_collection_data(collection_name, provider)

        if not documents:
            logger.info(f"{collection_name}: ドキュメントなし")
            return []

        area = self.extract_area(collection_name)
        bot_name = self._resolve_bot_name(area)
        total_keywords = len(keywords)

        matched: List[MatchResult] = []
        for doc, meta in zip(documents, metadatas):
            if source_filter and meta.get("source") != source_filter:
                continue

            doc_lower = doc.lower()
            match_count = sum(1 for kw in keywords if kw.lower() in doc_lower)
            if match_count == 0:
                continue

            parsed = self._text_combiner.parse(doc)
            row_index = meta.get("row_index", 0)
            excel_row = int(row_index) + 2
            scenario_id = f"{bot_name}_{excel_row}"

            matched.append(MatchResult(
                question=parsed.query,
                answer=parsed.answer,
                hierarchy=parsed.hierarchy,
                similarity=round(match_count / total_keywords, 4),
                match_count=match_count,
                scenario_id=scenario_id,
                sheet_name=meta.get("sheet_name", ""),
                row_index=int(row_index),
                collection_name=collection_name,
                source=meta.get("source", "unknown"),
            ))

        logger.info(f"{collection_name}: {len(matched)}件ヒット（キーワード検索）")
        return matched

    @staticmethod
    def extract_area(collection_name: str) -> str:
        """コレクション名からarea名を抽出（rev02_souzoku → souzoku）"""
        parts = collection_name.split("_", 1)
        if len(parts) == 2 and parts[0].startswith("rev"):
            return parts[1]
        return collection_name

    def _resolve_bot_name(self, area: str) -> str:
        """area名からbot名を解決"""
        area_lower = area.lower()
        for keyword, bot_name in self.area_to_bot.items():
            if keyword in area_lower:
                return bot_name
        return "unknown-bot"
