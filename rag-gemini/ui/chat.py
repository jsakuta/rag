# --- chat.py (旧 ui.py) ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from dotenv import load_dotenv
load_dotenv()  # .envファイルから環境変数を読み込み

from config import SearchConfig, load_settings
from typing import Dict, List, Optional, Tuple
import copy
import pandas as pd

# UI専用設定をYAMLから読み込み
_ui_settings = load_settings("ui")
# 評価設定（改定番号・エリア・ボット名マッピング等）
_eval_settings = load_settings("evaluation")
from src.core.processor import Processor
from src.utils.logger import setup_logger
from src.utils.dynamic_db_manager import DynamicDBManager
import datetime
import os
import html

logger = setup_logger(__name__)

# 評価設定から改定番号関連のマッピングを取得
AREA_TO_BOT = _eval_settings.get("area_to_bot", {})
AREA_TO_CATEGORY = _eval_settings.get("area_to_category", {})
CATEGORY_TO_AREA = {v: k for k, v in AREA_TO_CATEGORY.items()}  # 逆引き辞書

# 改定番号設定を解析
_raw_revision_areas = _eval_settings.get("revision_areas", {})
REVISION_TO_AREAS = {}
REVISION_VECTOR_WEIGHTS = {}
REVISION_SEARCH_TYPES = {}
DEFAULT_VECTOR_WEIGHT = _eval_settings.get("vector_weight", 0.9)

for rev, config in _raw_revision_areas.items():
    if isinstance(config, dict):
        REVISION_TO_AREAS[rev] = config.get("areas", [])
        REVISION_VECTOR_WEIGHTS[rev] = config.get("vector_weight", DEFAULT_VECTOR_WEIGHT)
        REVISION_SEARCH_TYPES[rev] = config.get("search_type", "hybrid")
    else:
        REVISION_TO_AREAS[rev] = config
        REVISION_VECTOR_WEIGHTS[rev] = DEFAULT_VECTOR_WEIGHT
        REVISION_SEARCH_TYPES[rev] = "hybrid"

# パス定数
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_FILE = PROJECT_ROOT / "input" / "multi_stage_input.xlsx"

# デフォルトの業務分野リスト（DBが存在しない場合のフォールバック）
DEFAULT_BUSINESS_AREAS = ["預金", "融資", "外貨", "投信", "住宅ローン", "カード", "保険", "年金", "総則"]


@st.cache_data(ttl=300)
def load_revision_correct_ids() -> Dict[str, Tuple[str, List[str]]]:
    """改定番号 → (改定内容, 正解IDリスト) マッピングを読み込み

    Returns:
        Dict[str, Tuple[str, List[str]]]: {改定番号: (改定内容, [正解ID, ...])}
    """
    if not INPUT_FILE.exists():
        logger.warning(f"正解IDファイルが見つかりません: {INPUT_FILE}")
        return {}

    try:
        df = pd.read_excel(INPUT_FILE)
        result = {}

        # 改定番号でグループ化
        for revision, group in df.groupby("番号", sort=False):
            revision_content = group.iloc[0]["改定内容"]
            correct_ids = group["正解ID"].tolist()
            result[revision] = (revision_content, correct_ids)

        logger.info(f"正解IDデータを読み込み: {len(result)}改定, {sum(len(v[1]) for v in result.values())}件")
        return result
    except Exception as e:
        logger.error(f"正解IDデータの読み込みエラー: {e}")
        return {}


def extract_bot_name_from_category(category: str) -> str:
    """カテゴリ名（Sheet_Name）からボット名を抽出

    Args:
        category: カテゴリ名（例: スマイルタブレット, 内部事務）

    Returns:
        ボット名（例: smile-bot, naibujimu-bot）
    """
    area = CATEGORY_TO_AREA.get(category)
    if area:
        return AREA_TO_BOT.get(area, "unknown-bot")
    return "unknown-bot"


def build_scenario_id(result: Dict) -> str:
    """検索結果からシナリオIDを構築

    Args:
        result: 検索結果辞書（Sheet_Name, Row_Index等を含む）

    Returns:
        シナリオID（例: smile-bot_129）
    """
    sheet_name = result.get("Sheet_Name", "")
    row_index = result.get("Row_Index", "")

    if not sheet_name or row_index == "":
        return ""

    try:
        # Excel行番号 = row_index + 2（ヘッダー行1 + 0-based index）
        excel_row = int(row_index) + 2
        bot_name = extract_bot_name_from_category(sheet_name)
        return f"{bot_name}_{excel_row}"
    except (ValueError, TypeError):
        return ""


def check_if_correct(result: Dict, correct_ids: List[str]) -> Tuple[str, bool]:
    """検索結果が正解IDとマッチするか判定

    Args:
        result: 検索結果辞書
        correct_ids: 正解IDリスト

    Returns:
        (シナリオID, 正解かどうか)
    """
    scenario_id = build_scenario_id(result)
    is_correct = scenario_id in correct_ids if scenario_id else False
    return scenario_id, is_correct


@st.cache_data(ttl=60)  # 60秒キャッシュ
def get_available_business_areas() -> list:
    """利用可能な業務分野を動的に取得"""
    try:
        config = SearchConfig(base_dir=".")
        db_manager = DynamicDBManager(config)
        areas = db_manager.get_all_business_areas()
        if areas:
            return areas
    except Exception as e:
        logger.warning(f"業務分野一覧の取得に失敗: {e}")
    return DEFAULT_BUSINESS_AREAS


def initialize_session_state():
    """セッションステートの初期化"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processing_query" not in st.session_state:
        st.session_state.processing_query = False
    # 正解バッジ・改定番号選択用の新規状態
    if "correct_ids" not in st.session_state:
        st.session_state.correct_ids = []
    if "selected_revision" not in st.session_state:
        st.session_state.selected_revision = None
    # Azure/VertexAI両プロバイダー検索結果
    if "azure_results" not in st.session_state:
        st.session_state.azure_results = []
    if "vertex_results" not in st.session_state:
        st.session_state.vertex_results = []
    # 両プロバイダー検索モード
    if "dual_provider_mode" not in st.session_state:
        st.session_state.dual_provider_mode = False
    if "config" not in st.session_state:
        # 必須環境変数のチェック（LLM設定のみ）
        required_env_vars = [
            "DEFAULT_LLM_PROVIDER",
            "DEFAULT_LLM_MODEL",
        ]
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"必須環境変数が設定されていません: {', '.join(missing_vars)}")

        # UI専用設定をYAMLから取得
        ui_top_k = _ui_settings.get("top_k", 3)
        ui_vector_weight = _ui_settings.get("vector_weight", 0.9)
        ui_search_mode = _ui_settings.get("search_mode", "original")
        ui_search_type = _ui_settings.get("search_type", "hybrid")

        st.session_state.config = SearchConfig(
            search_type=ui_search_type,  # UIで切り替え可能（類似検索/キーワード必須）
            search_mode=ui_search_mode,  # UIで切り替え可能
            top_k=ui_top_k,
            llm_provider=os.getenv("DEFAULT_LLM_PROVIDER"),
            llm_model=os.getenv("DEFAULT_LLM_MODEL"),
            vector_weight=ui_vector_weight,
            base_dir="."
            # embedding_provider, embedding_model は config.py で環境変数から読み込み
        )
    if "business_area" not in st.session_state:
        st.session_state.business_area = "預金"

def format_message(message, is_user=False):
    escaped_message = html.escape(str(message))
    style = f"""
        <div style="display: flex; justify-content: {'flex-end' if is_user else 'flex-start'}; margin: 5px 0;">
            <div style="background-color: {'#e6f3ff' if is_user else '#f5f5f5'}; padding: 10px 15px;
                border-radius: 15px; max-width: 80%; {'margin-left: auto;' if is_user else ''}">
                {escaped_message}
            </div>
        </div>
    """
    return style

def format_response_card(number, similarity, query, answer, category=None,
                         relevance_judgment=None, judgment_reason=None, modification_suggestion=None,
                         scenario_id=None, is_correct=False):
    # XSS対策: ユーザー入力をエスケープ
    query = html.escape(str(query))
    answer = html.escape(str(answer))

    # カテゴリバッジの色設定
    category_colors = {
        'Both': '#E2EFDA',
        'Original_Only': '#FFF2CC',
        'LLM_Enhanced_Only': '#DEEBF7'
    }
    category_labels = {
        'Both': '両方',
        'Original_Only': '原文のみ',
        'LLM_Enhanced_Only': 'LLMのみ'
    }
    category_badge = ""
    if category:
        bg_color = category_colors.get(category, '#f0f0f0')
        label = category_labels.get(category, category)
        category_badge = f" <span style='background-color: {bg_color}; padding: 2px 8px; border-radius: 4px; font-size: 0.85em; margin-left: 8px;'>{label}</span>"

    # 正解バッジ
    correct_badge = ""
    if is_correct:
        correct_badge = " <span style='background-color: #c8e6c9; color: #2e7d32; padding: 2px 8px; border-radius: 4px; font-size: 0.85em; font-weight: bold; margin-left: 8px;'>✓正解</span>"

    # シナリオIDバッジ
    scenario_id_badge = ""
    if scenario_id:
        escaped_id = html.escape(str(scenario_id))
        scenario_id_badge = f" <span style='background-color: #e3f2fd; color: #1565c0; padding: 2px 8px; border-radius: 4px; font-size: 0.85em; margin-left: 8px;'>{escaped_id}</span>"

    # 関連性判定バッジの色設定
    relevance_colors = {
        '関連あり': '#c8e6c9',
        '要確認': '#fff9c4',
        '関連なし': '#ffcdd2',
        '判断支援無効': '#e0e0e0',
        'エラー': '#ffcdd2'
    }
    relevance_badge = ""
    if relevance_judgment:
        # 関連性判定の最初の単語を取得（「関連あり」「要確認」「関連なし」など）
        judgment_key = relevance_judgment.split()[0] if relevance_judgment else ""
        bg_color = relevance_colors.get(judgment_key, '#e0e0e0')
        escaped_judgment = html.escape(str(relevance_judgment))
        relevance_badge = f" <span style='background-color: {bg_color}; padding: 2px 8px; border-radius: 4px; font-size: 0.85em; margin-left: 8px;'>{escaped_judgment}</span>"

    # LLM分析セクション（関連性判定がある場合のみ表示）
    llm_analysis_section = ""
    show_llm_analysis = relevance_judgment and relevance_judgment not in ['判断支援無効', '']
    if show_llm_analysis:
        reason_text = html.escape(str(judgment_reason)) if judgment_reason else "-"
        suggestion_raw = html.escape(str(modification_suggestion)) if modification_suggestion else ""
        suggestion_text = "なし" if not suggestion_raw or suggestion_raw.strip() in ['-', 'なし'] else suggestion_raw

        llm_analysis_section = f"""<div style="background-color: #f0f7ff; padding: 12px; border-radius: 8px; margin: 8px 0;"><div style="font-weight: 600; margin-bottom: 5px;">LLM分析</div><div>関連性: {html.escape(str(relevance_judgment))}</div><div>根拠: {reason_text}</div><div>修正案: {suggestion_text}</div></div>"""

    return f"""
        <div class="response-card" style="border: 1px solid #e0e0e0; border-radius: 10px; padding: 15px;
            margin: 10px 0; background-color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="color: #666; margin-bottom: 10px; font-size: 0.95em; padding-bottom: 8px;
                border-bottom: 1px solid #eee;">候補 {number} (類似度: {similarity:.4f}){correct_badge}{scenario_id_badge}{category_badge}{relevance_badge}
            </div>
            <div style="background-color: #f8f9fa; padding: 12px; border-radius: 8px; margin: 8px 0;">
                <div style="font-weight: 600; margin-bottom: 5px;">類似質問内容:</div>
                <div style="white-space: pre-wrap;">{query}</div>
            </div>
            <div style="background-color: #f8f9fa; padding: 12px; border-radius: 8px; margin: 8px 0;">
                <div style="font-weight: 600; margin-bottom: 5px;">回答:</div>
                <div style="white-space: pre-wrap;">{answer}</div>
            </div>
            {llm_analysis_section}
        </div>
    """

def _needs_processor_reinit() -> bool:
    """Processorの再初期化が必要かどうかを判定"""
    if "processor" not in st.session_state:
        return True

    config = st.session_state.config
    return (
        st.session_state.get("last_business_area") != st.session_state.business_area
        or st.session_state.get("last_search_type") != config.search_type
        or st.session_state.get("last_search_mode") != config.search_mode
        or st.session_state.get("last_judgment_support") != config.multi_stage_enable_judgment_support
        or st.session_state.get("last_search_source") != config.search_source
    )


def _load_reference_data_for_business(config, business_area: str) -> dict:
    """業務分野に応じた参照データを読み込む"""
    # rev系の業務分野かチェック
    if business_area.startswith("rev"):
        # rev系の場合は対応するシナリオファイルから読み込み
        try:
            with DynamicDBManager(config) as db_manager:
                business_areas = db_manager.analyze_reference_files()

                if business_area in business_areas:
                    area_data = business_areas[business_area]
                    # 構造: {'faq': [], 'scenario': [('filename.xlsx', timestamp), ...]}
                    scenario_list = area_data.get("scenario", [])

                    if scenario_list:
                        # 最新のシナリオファイルを使用（タプルの最初の要素がファイル名）
                        scenario_file = scenario_list[0][0]
                        # db_managerからシナリオパスを取得
                        scenario_path = os.path.join(db_manager.reference_scenario_path, scenario_file)

                        from src.handlers.input_handler import HierarchicalExcelInputHandler
                        handler = HierarchicalExcelInputHandler(config, scenario_path)
                        reference_data = handler.load_reference_data()
                        logger.info(f"rev系業務分野 '{business_area}' の参照データ読み込み完了: {len(reference_data['combined_texts'])}件")
                        return reference_data
                    else:
                        logger.warning(f"業務分野 '{business_area}' のシナリオファイルが見つかりません")
                else:
                    logger.warning(f"業務分野 '{business_area}' が参照ファイル分析結果に存在しません")
        except Exception as e:
            import traceback
            logger.error(f"rev系参照データの読み込みエラー: {e}")
            logger.error(traceback.format_exc())
            # rev系でエラーの場合は空のデータを返さず例外を再送出
            raise

    # 従来の方法で読み込み（非rev系の業務分野のみ）
    processor = Processor(config)
    return processor.reference_handler.load_reference_data()


def execute_dual_provider_search(query: str, revision: str) -> Tuple[List[Dict], List[Dict], str]:
    """Azure/VertexAI両方で検索を実行

    Args:
        query: 検索クエリ
        revision: 改定番号（例: ①, ②）

    Returns:
        (azure_results, vertex_results, llm_query)
    """
    from src.core.search.multi_stage_orchestrator import MultiStageOrchestrator
    from src.core.search.vector_search_engine import VectorSearchEngine
    from src.core.search.keyword_search_engine import KeywordSearchEngine
    from src.core.search.query_enhancer import QueryEnhancer
    from src.core.search.text_combiner import get_text_combiner
    from src.utils.auth import create_embedding_model, create_llm
    from src.utils.vector_db import MetadataVectorDB

    config = st.session_state.config
    areas = REVISION_TO_AREAS.get(revision, [])
    search_type = REVISION_SEARCH_TYPES.get(revision, "hybrid")
    vector_weight = REVISION_VECTOR_WEIGHTS.get(revision, DEFAULT_VECTOR_WEIGHT)

    if not areas:
        logger.warning(f"改定 {revision} に対応するエリアがありません")
        return [], [], ""

    # キーワード必須検索の場合はVertexAIをスキップ
    if search_type == "keyword_filter":
        # Azureのみで検索（VertexAIは空の結果）
        azure_results = _search_with_provider(query, revision, "azure_openai", areas, vector_weight)
        return azure_results, [], ""

    # 両プロバイダーで検索
    azure_results = _search_with_provider(query, revision, "azure_openai", areas, vector_weight)
    vertex_results = _search_with_provider(query, revision, "vertex_ai", areas, vector_weight)

    # LLMクエリを取得（Azure結果から）
    llm_query = azure_results[0].get("Search_Query", query) if azure_results else ""

    return azure_results, vertex_results, llm_query


def _search_with_provider(query: str, revision: str, provider: str, areas: List[str], vector_weight: float) -> List[Dict]:
    """特定のプロバイダーで検索を実行"""
    from src.core.search.multi_stage_orchestrator import MultiStageOrchestrator
    from src.core.search.vector_search_engine import VectorSearchEngine
    from src.core.search.keyword_search_engine import KeywordSearchEngine
    from src.core.search.query_enhancer import QueryEnhancer
    from src.core.search.text_combiner import get_text_combiner
    from src.utils.auth import create_embedding_model, create_llm
    from src.utils.vector_db import MetadataVectorDB

    config = st.session_state.config
    VECTOR_DB_BASE = PROJECT_ROOT / "reference" / "vector_db"
    all_results = []

    for area in areas:
        db_path = VECTOR_DB_BASE / area / provider
        if not db_path.exists():
            logger.warning(f"DBが存在しません: {db_path}")
            continue

        try:
            # Embedding モデル作成
            provider_config = copy.copy(config)
            provider_config.embedding_provider = provider
            if provider == "azure_openai":
                provider_config.embedding_model = os.getenv(
                    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"
                )
            else:
                provider_config.embedding_model = os.getenv(
                    "VERTEX_AI_EMBEDDING_MODEL", "gemini-embedding-001"
                )
            embedding_model = create_embedding_model(provider_config)

            # ベクトルDB
            vector_db = MetadataVectorDB(db_path=str(db_path), collection_name="default")

            # 参照クエリ取得
            text_combiner = get_text_combiner()
            result = vector_db.collection.get(include=["documents"])
            documents = result.get("documents", [])
            reference_queries = []
            for doc in documents:
                if doc:
                    parsed = text_combiner.parse(doc)
                    reference_queries.append(parsed.query if parsed.query else doc[:100])

            if not reference_queries:
                continue

            # エンジン作成
            vector_engine = VectorSearchEngine(
                embedding_model=embedding_model, vector_db=vector_db
            )
            keyword_engine = KeywordSearchEngine(
                stop_words=config.STOP_WORDS,
                position_weight=config.POSITION_WEIGHT,
            )
            keyword_engine.build_cache(reference_queries)

            llm = create_llm(config)
            query_enhancer = QueryEnhancer(llm=llm, base_dir=str(PROJECT_ROOT))

            # オーケストレーター
            # UI設定のtop_kを使用（両プロバイダーモードでも候補数を反映）
            ui_top_k = config.top_k if hasattr(config, 'top_k') else _eval_settings.get("top_k", 50)
            orchestrator = MultiStageOrchestrator(
                vector_engine=vector_engine,
                keyword_engine=keyword_engine,
                query_enhancer=query_enhancer,
                text_combiner=text_combiner,
                vector_weight=vector_weight,
                threshold=_eval_settings["thresholds"].get(provider, 0.5),
                max_results=_eval_settings.get("max_results", 100),
                filter_mode="top_k",  # 常にtop_kモード
                top_k=ui_top_k,
            )

            # 検索実行
            results = orchestrator.execute(
                input_number=revision,
                query_text=query,
                original_answer="",
                filter_metadata=None,
            )

            # 結果を変換
            for r in results:
                all_results.append({
                    "Similarity": r.get("Similarity", 0),
                    "Search_Result_Q": r.get("Search_Result_Q", ""),
                    "Search_Result_A": r.get("Search_Result_A", ""),
                    "Search_Category": r.get("Search_Category", ""),
                    "Sheet_Name": r.get("Sheet_Name", ""),
                    "Row_Index": r.get("Row_Index", ""),
                    "Search_Query": r.get("Search_Query", ""),
                })

        except Exception as e:
            logger.error(f"検索エラー ({area}/{provider}): {e}")
            import traceback
            traceback.print_exc()

    # 類似度順でソート
    all_results.sort(key=lambda x: x.get("Similarity", 0), reverse=True)
    return all_results


def _initialize_processor():
    """Processorを初期化してセッションステートを更新"""
    st.session_state.processor = Processor(st.session_state.config)

    # 業務分野に応じた参照データを読み込み
    business_area = st.session_state.business_area
    reference_data = _load_reference_data_for_business(st.session_state.config, business_area)

    st.session_state.processor.searcher.prepare_search(reference_data)
    st.session_state.processor.searcher._select_db_for_business(business_area)
    st.session_state.last_business_area = business_area
    st.session_state.last_search_type = st.session_state.config.search_type
    st.session_state.last_search_mode = st.session_state.config.search_mode
    st.session_state.last_judgment_support = st.session_state.config.multi_stage_enable_judgment_support
    st.session_state.last_search_source = st.session_state.config.search_source


def process_query(query: str):
    st.session_state.processing_query = True
    try:
        query_number = len(st.session_state.chat_history) // 2 + 1
        query_display = f"{query[:80]}..." if len(query) > 80 else query
        logger.info(f"=== 質問 {query_number} の処理開始 ===")
        logger.info(f"クエリ: {query_display}")

        # 両プロバイダーモード（改定番号選択時）
        if st.session_state.dual_provider_mode and st.session_state.selected_revision:
            revision = st.session_state.selected_revision
            search_type = REVISION_SEARCH_TYPES.get(revision, "hybrid")
            logger.info(f"両プロバイダー検索モード: 改定番号={revision}, 検索タイプ={search_type}")

            # Azure/VertexAI両方で検索
            azure_results, vertex_results, llm_query = execute_dual_provider_search(query, revision)

            st.session_state.azure_results = azure_results
            st.session_state.vertex_results = vertex_results
            st.session_state.last_query = query
            st.session_state.last_llm_query = llm_query

            logger.info(f"Azure検索結果数: {len(azure_results)}件")
            logger.info(f"VertexAI検索結果数: {len(vertex_results)}件")

            # チャット履歴には両プロバイダー結果をまとめて保存
            st.session_state.chat_history.append({
                "type": "bot",
                "text": {
                    "mode": "dual_provider",
                    "azure": azure_results,
                    "vertex": vertex_results,
                    "llm_query": llm_query,
                }
            })
        else:
            # 通常の単一プロバイダー検索
            if _needs_processor_reinit():
                _initialize_processor()

            processor = st.session_state.processor

            # ログ: 処理開始
            judgment_enabled = st.session_state.config.multi_stage_enable_judgment_support
            search_type_labels = {"hybrid": "類似検索", "keyword_filter": "キーワード必須"}
            search_type_label = search_type_labels.get(st.session_state.config.search_type, st.session_state.config.search_type)
            logger.info(f"検索タイプ: {search_type_label}")
            source_labels = {"all": "シナリオ+FAQ", "scenario": "シナリオのみ", "history_data": "FAQのみ"}
            search_source_label = source_labels.get(st.session_state.config.search_source, st.session_state.config.search_source)
            if st.session_state.config.search_type == "hybrid":
                logger.info(f"検索モード: {st.session_state.config.search_mode} (LLM判断支援: {'有効' if judgment_enabled else '無効'})")
                logger.info(f"検索対象: {search_source_label}")
                logger.info(f"検索バランス: ベクトル重み={st.session_state.config.vector_weight:.1f}")

            results = processor.searcher.search(str(query_number), query, "")

            if results:
                logger.info(f"検索結果数: {len(results)}件")
                # 検索結果のログ出力
                for i, result in enumerate(results, 1):
                    category = result.get('Search_Category', 'N/A')
                    similarity = result.get('Similarity', 0)
                    logger.info(f"  結果{i}: 類似度={similarity:.4f}, カテゴリ={category}")

                # 最新の検索結果を保存（LLM分析用）
                st.session_state.last_query = query
                st.session_state.last_results = results

                st.session_state.chat_history.append({"type": "bot", "text": results})
            else:
                logger.info("検索結果: 該当なし")
                st.session_state.last_query = None
                st.session_state.last_results = None
                st.session_state.chat_history.append({"type": "bot", "text": "該当する結果が見つかりませんでした。"})

        logger.info(f"=== 質問 {query_number} の検索完了 ===")

    except Exception as e:
        # セキュリティ: XSS対策としてエラーメッセージをエスケープ
        escaped_error = html.escape(str(e))
        error_message = f"エラーが発生しました: {escaped_error}"
        st.error(error_message)
        logger.error(f"Error processing query: {str(e)}", exc_info=True)  # ログにはオリジナルを出力
        st.session_state.chat_history.append({"type": "bot", "text": error_message})
    finally:
        st.session_state.processing_query = False


def run_llm_analysis():
    """最新の検索結果に対してLLM分析を実行"""
    if "last_results" not in st.session_state or not st.session_state.last_results:
        st.warning("分析対象の検索結果がありません。")
        return

    if "processor" not in st.session_state:
        st.error("Processorが初期化されていません。")
        return

    processor = st.session_state.processor
    if not hasattr(processor, 'judgment_support') or processor.judgment_support is None:
        st.error("JudgmentSupportが初期化されていません。")
        return

    results = st.session_state.last_results
    query = st.session_state.last_query
    judgment_support = processor.judgment_support

    logger.info(f"=== LLM分析開始 ({len(results)}件) ===")

    for i, result in enumerate(results, 1):
        # 安定性: 各結果の分析を個別にtry-except
        try:
            evaluation = judgment_support.evaluate(
                query,
                result.get('Search_Result_Q', ''),
                result.get('Search_Result_A', '')
            )
            result['Relevance_Judgment'] = evaluation['relevance_judgment']
            result['Judgment_Reason'] = evaluation['judgment_reason']
            result['Modification_Suggestion'] = evaluation['modification_suggestion']
            logger.info(f"  結果{i}: → LLM判定: {evaluation['relevance_judgment']}")
        except Exception as e:
            logger.error(f"  結果{i}: LLM分析エラー: {e}")
            result['Relevance_Judgment'] = "エラー"
            result['Judgment_Reason'] = f"分析エラー: {str(e)[:50]}"
            result['Modification_Suggestion'] = ""

    # チャット履歴の最新の結果を更新
    if st.session_state.chat_history:
        for msg in reversed(st.session_state.chat_history):
            if msg["type"] == "bot" and isinstance(msg["text"], list):
                msg["text"] = results
                break

    logger.info("=== LLM分析完了 ===")

def save_chat_history():
    """チャット履歴を保存"""
    try:
        chat_data = []
        for i in range(0, len(st.session_state.chat_history), 2):
            if i + 1 < len(st.session_state.chat_history):
                user_query = st.session_state.chat_history[i]["text"]
                bot_response = st.session_state.chat_history[i + 1]["text"]

                if isinstance(bot_response, list):
                    for response in bot_response:
                        chat_data.append({
                            'Input_Number': response.get('Input_Number', ''),
                            'Original_Query': user_query,
                            'Search_Query': response.get('Search_Query', ''),
                            'Search_Result_Q': response.get('Search_Result_Q', ''),
                            'Search_Result_A': response.get('Search_Result_A', ''),
                            'Similarity': response.get('Similarity', ''),
                            'Vector_Weight': response.get('Vector_Weight', ''),
                            'Top_K': response.get('Top_K', '')
                        })

        if chat_data:
            processor = Processor(st.session_state.config)
            # OutputHandlerを使用してチャット履歴を保存
            processor.output_handler.save_data(chat_data, mode="chat")  # modeを"chat"に設定
            st.sidebar.success("チャット履歴を保存しました。")
        else:
            st.sidebar.warning("保存するチャット履歴がありません。")

    except Exception as e:
        logger.error(f"Error saving chat history: {str(e)}", exc_info=True)
        st.sidebar.error("チャット履歴の保存中にエラーが発生しました。")

def run_streamlit_ui():
    st.set_page_config(page_title="類似回答検索ボット【預金】", layout="wide", initial_sidebar_state="expanded")
    st.markdown("""
        <style>
        div.stButton > button:hover { background-color: #007bff; color: white; }
        div.stButton > button:focus { background-color: #007bff; color: white; }
        [data-baseweb="button"]:has(#save_chat_history_button) { background-color: #28a745 !important; color: white !important; }
        [data-baseweb="button"]:has(#save_chat_history_button):hover { background-color: #1e7e34 !important; }
        </style>
    """, unsafe_allow_html=True)

    initialize_session_state()

    with st.sidebar:
        st.title("設定")
        with st.expander("パラメータ調整", expanded=True):
            # 検索タイプ選択（類似検索 / キーワード必須）
            search_type_labels = {
                "hybrid": "類似検索（意味で探す）",
                "keyword_filter": "キーワード必須（ワードを含むもの）"
            }
            current_search_type = st.session_state.config.search_type if hasattr(st.session_state.config, 'search_type') else "hybrid"
            selected_search_type = st.radio(
                "検索タイプ",
                options=["hybrid", "keyword_filter"],
                format_func=lambda x: search_type_labels[x],
                index=0 if current_search_type == "hybrid" else 1,
                key="search_type_radio"
            )
            st.session_state.config.search_type = selected_search_type

            # 類似検索（hybrid）選択時のみ検索バランススライダーを表示
            if selected_search_type == "hybrid":
                # ワード重視 ↔ 意味重視のスライダー
                st.session_state.config.vector_weight = st.slider(
                    "検索バランス",
                    0.0, 1.0,
                    st.session_state.config.vector_weight, 0.1,
                    help="左：ワード重視 / 右：意味重視"
                )
                # keyword_weight を自動計算
                st.session_state.config.keyword_weight = 1.0 - st.session_state.config.vector_weight

                # スライダーの両端ラベルを表示
                col1, col2 = st.columns(2)
                with col1:
                    st.caption("← ワード重視")
                with col2:
                    st.caption("意味重視 →")

            # 検索モード選択（類似検索時のみ有効）
            if selected_search_type == "hybrid":
                search_modes = ["original", "llm_enhanced", "multi_stage"]
                mode_labels = {"original": "原文検索", "llm_enhanced": "LLMクエリ検索", "multi_stage": "多段階OR検索"}
                current_mode_index = search_modes.index(st.session_state.config.search_mode) if st.session_state.config.search_mode in search_modes else 0
                selected_mode = st.selectbox(
                    "検索モード",
                    search_modes,
                    format_func=lambda x: mode_labels[x],
                    index=current_mode_index
                )
                st.session_state.config.search_mode = selected_mode

                # 検索対象選択
                search_sources = ["all", "scenario", "history_data"]
                source_labels = {"all": "シナリオ+FAQ", "scenario": "シナリオのみ", "history_data": "FAQのみ"}
                current_source_index = search_sources.index(st.session_state.config.search_source) if st.session_state.config.search_source in search_sources else 0
                selected_source = st.selectbox(
                    "検索対象",
                    search_sources,
                    format_func=lambda x: source_labels[x],
                    index=current_source_index
                )
                st.session_state.config.search_source = selected_source

                # 多段階検索パラメータ（multi_stage時のみ表示）
                if selected_mode == "multi_stage":
                    # 候補数入力（top_k）- 多段階検索は閾値ではなくtop_kで制御
                    # デフォルト値が10未満の場合は100に設定
                    current_top_k = max(10, st.session_state.config.top_k) if st.session_state.config.top_k < 10 else st.session_state.config.top_k
                    st.session_state.config.top_k = st.number_input(
                        "候補数", min_value=10, max_value=200,
                        value=current_top_k, step=10
                    )
                    st.session_state.config.multi_stage_enable_judgment_support = st.checkbox(
                        "LLM判断支援",
                        value=st.session_state.config.multi_stage_enable_judgment_support
                    )
                else:
                    # 通常検索モード（original, llm_enhanced）の候補数
                    st.session_state.config.top_k = st.number_input(
                        "表示する候補数", min_value=1, max_value=10,
                        value=st.session_state.config.top_k, step=1
                    )
            else:
                # キーワード必須検索の場合は固定値
                selected_mode = "original"
                st.session_state.config.search_mode = "original"

            # 業務分野選択（動的に取得）
            business_areas = get_available_business_areas()
            # 現在選択中の業務分野がリストにない場合は先頭を選択
            current_area = st.session_state.business_area
            if current_area not in business_areas:
                current_area = business_areas[0] if business_areas else "預金"
                st.session_state.business_area = current_area
            st.session_state.business_area = st.selectbox(
                "業務分野",
                business_areas,
                index=business_areas.index(current_area)
            )

        # 改定番号選択（事務改定評価モード）
        st.markdown("---")
        st.subheader("事務改定評価")

        # 正解IDデータを読み込み
        revision_data = load_revision_correct_ids()
        revision_options = ["なし"] + list(revision_data.keys())

        current_revision_idx = 0
        if st.session_state.selected_revision in revision_options:
            current_revision_idx = revision_options.index(st.session_state.selected_revision)

        selected_revision = st.selectbox(
            "改定番号",
            revision_options,
            index=current_revision_idx,
            key="revision_select",
            help="改定番号を選択すると、Azure/VertexAI両方で検索し、正解IDとマッチした結果にバッジを表示します"
        )

        # 改定番号選択時の処理
        if selected_revision != "なし":
            st.session_state.selected_revision = selected_revision
            st.session_state.dual_provider_mode = True

            # 正解IDを設定
            if selected_revision in revision_data:
                content, correct_ids = revision_data[selected_revision]
                st.session_state.correct_ids = correct_ids

                # 改定番号に応じた検索タイプを自動設定
                revision_search_type = REVISION_SEARCH_TYPES.get(selected_revision, "hybrid")
                st.session_state.config.search_type = revision_search_type

                # 情報表示
                st.info(f"正解ID: {len(correct_ids)}件設定")
                search_type_label = "キーワード必須" if revision_search_type == "keyword_filter" else "類似検索"
                st.caption(f"検索タイプ: {search_type_label}")
        else:
            st.session_state.selected_revision = None
            st.session_state.dual_provider_mode = False
            st.session_state.correct_ids = []

        if st.button("チャット履歴を保存", use_container_width=True, key="save_chat_history_button"):
            save_chat_history()

    # タイトル（改定番号選択時は改定番号も表示）
    if st.session_state.dual_provider_mode and st.session_state.selected_revision:
        st.title(f"類似回答検索ボット【改定{st.session_state.selected_revision}】")
    else:
        st.title(f"類似回答検索ボット【{st.session_state.business_area}】")

    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["type"] == "user":
                st.markdown(format_message(msg["text"], True), unsafe_allow_html=True)
            else:
                # 両プロバイダーモードの結果表示
                if isinstance(msg["text"], dict) and msg["text"].get("mode") == "dual_provider":
                    azure_results = msg["text"].get("azure", [])
                    vertex_results = msg["text"].get("vertex", [])
                    llm_query = msg["text"].get("llm_query", "")
                    correct_ids = st.session_state.correct_ids

                    # LLMクエリを表示（あれば）
                    if llm_query:
                        st.markdown(f"<div style='background-color: #f0f7ff; padding: 8px 12px; border-radius: 6px; margin-bottom: 10px; font-size: 0.9em;'><strong>LLM強化クエリ:</strong> {html.escape(llm_query)}</div>", unsafe_allow_html=True)

                    # タブ切り替え表示
                    tab_azure, tab_vertex = st.tabs(["Azure", "VertexAI"])

                    with tab_azure:
                        if azure_results:
                            for idx, response in enumerate(azure_results[:20], 1):  # 最大20件表示
                                scenario_id, is_correct = check_if_correct(response, correct_ids)
                                category = response.get("Search_Category")
                                card_html = format_response_card(
                                    idx, response["Similarity"],
                                    response["Search_Result_Q"], response["Search_Result_A"],
                                    category=category, scenario_id=scenario_id, is_correct=is_correct
                                )
                                st.markdown(card_html, unsafe_allow_html=True)
                            if len(azure_results) > 20:
                                st.info(f"他 {len(azure_results) - 20}件の結果があります")
                        else:
                            st.info("該当する結果がありません")

                    with tab_vertex:
                        if vertex_results:
                            for idx, response in enumerate(vertex_results[:20], 1):  # 最大20件表示
                                scenario_id, is_correct = check_if_correct(response, correct_ids)
                                category = response.get("Search_Category")
                                card_html = format_response_card(
                                    idx, response["Similarity"],
                                    response["Search_Result_Q"], response["Search_Result_A"],
                                    category=category, scenario_id=scenario_id, is_correct=is_correct
                                )
                                st.markdown(card_html, unsafe_allow_html=True)
                            if len(vertex_results) > 20:
                                st.info(f"他 {len(vertex_results) - 20}件の結果があります")
                        else:
                            # キーワード必須検索の場合
                            search_type = REVISION_SEARCH_TYPES.get(st.session_state.selected_revision, "hybrid")
                            if search_type == "keyword_filter":
                                st.info("キーワード必須検索のためスキップ（Azureタブの結果をご確認ください）")
                            else:
                                st.info("該当する結果がありません")

                elif isinstance(msg["text"], list):
                    # 通常の単一プロバイダー検索結果
                    correct_ids = st.session_state.correct_ids
                    for idx, response in enumerate(msg["text"], 1):
                        category = response.get("Search_Category")  # 多段階検索時のみ存在
                        # LLM分析結果（多段階検索+LLM判断支援有効時のみ存在）
                        relevance_judgment = response.get("Relevance_Judgment")
                        judgment_reason = response.get("Judgment_Reason")
                        modification_suggestion = response.get("Modification_Suggestion")
                        # 正解バッジ表示（改定番号選択時のみ）
                        scenario_id, is_correct = check_if_correct(response, correct_ids)
                        card_html = format_response_card(
                            idx, response["Similarity"],
                            response["Search_Result_Q"], response["Search_Result_A"],
                            category, relevance_judgment, judgment_reason, modification_suggestion,
                            scenario_id=scenario_id, is_correct=is_correct
                        )
                        st.markdown(card_html, unsafe_allow_html=True)
                else:
                    st.markdown(format_message(msg["text"], False), unsafe_allow_html=True)

        # LLM分析ボタン（多段階検索+LLM判断支援有効+検索結果あり時のみ表示）
        show_llm_button = (
            st.session_state.config.search_mode == "multi_stage"
            and st.session_state.config.multi_stage_enable_judgment_support
            and st.session_state.get("last_results")
            and not any(r.get("Relevance_Judgment") for r in st.session_state.get("last_results", []))
        )
        if show_llm_button:
            if st.button("LLM分析を実行", use_container_width=True, type="primary"):
                with st.spinner("LLM分析を実行中..."):
                    run_llm_analysis()
                st.rerun()

        st.markdown('<div style="height: 50px;"></div>', unsafe_allow_html=True)

    with st.form(key="chat_form", clear_on_submit=True):
        query = st.text_input("質問を入力してください", key="query", label_visibility="collapsed", placeholder="質問を入力...")
        submit_button = st.form_submit_button("送信", use_container_width=True, disabled=st.session_state.processing_query)

    if submit_button and query.strip():
        st.session_state.chat_history.append({"type": "user", "text": query})
        process_query(query.strip())
        st.rerun()

    if st.session_state.processing_query:
        st.markdown('<div class="processing-indicator">処理中...</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    run_streamlit_ui()