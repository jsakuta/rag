import os
import re
import shutil
import logging
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import chromadb
from chromadb.config import Settings
from chromadb.errors import NotFoundError as ChromaNotFoundError
from config import SearchConfig
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class DynamicDBError(Exception):
    """動的DB管理のエラー"""
    pass

class DynamicDBManager:
    """動的DB管理システム"""
    
    def __init__(self, config: SearchConfig):
        self.config = config
        self.base_db_path = os.path.join(config.base_dir, "reference", "vector_db")
        self.reference_faq_path = os.path.join(config.base_dir, "reference", "faq_data")
        self.reference_scenario_path = os.path.join(config.base_dir, "reference", "scenario")
        
        # ディレクトリの作成
        os.makedirs(self.base_db_path, exist_ok=True)
        os.makedirs(self.reference_faq_path, exist_ok=True)
        os.makedirs(self.reference_scenario_path, exist_ok=True)
        
        # 更新日時記録ファイルのパス
        self.update_timestamp_file = os.path.join(self.base_db_path, "update_timestamps.json")
        
        # 更新日時の読み込み
        self._load_update_timestamps()
        
        # 既存DBの移行（初回のみ）
        self._migrate_existing_db()
    
    def _load_update_timestamps(self):
        """更新日時の記録を読み込み"""
        try:
            if os.path.exists(self.update_timestamp_file):
                with open(self.update_timestamp_file, 'r', encoding='utf-8') as f:
                    timestamps = json.load(f)
                    self._last_faq_mtime = timestamps.get('faq', {})
                    self._last_scenario_mtime = timestamps.get('scenario', {})
                    logger.info(f"更新日時記録を読み込み: FAQ={len(self._last_faq_mtime)}件, シナリオ={len(self._last_scenario_mtime)}件")
            else:
                self._last_faq_mtime = {}
                self._last_scenario_mtime = {}
                logger.info("更新日時記録ファイルが存在しないため、新規作成します")
        except Exception as e:
            logger.warning(f"更新日時記録の読み込みエラー: {e}")
            self._last_faq_mtime = {}
            self._last_scenario_mtime = {}
    
    def _save_update_timestamps(self):
        """更新日時の記録を保存"""
        try:
            timestamps = {
                'faq': self._last_faq_mtime,
                'scenario': self._last_scenario_mtime
            }
            with open(self.update_timestamp_file, 'w', encoding='utf-8') as f:
                json.dump(timestamps, f, ensure_ascii=False, indent=2)
            logger.info(f"更新日時記録を保存: FAQ={len(self._last_faq_mtime)}件, シナリオ={len(self._last_scenario_mtime)}件")
        except Exception as e:
            logger.warning(f"更新日時記録の保存エラー: {e}")
    
    def _migrate_existing_db(self):
        """既存のDB移行処理（現在は不要）"""
        # rag_collectionは完全に廃止、移行処理は不要
        logger.info("既存DB移行処理は不要（動的DB管理システムが唯一の方式）")
        pass
    
    def analyze_reference_files(self) -> Dict[str, Dict[str, List[Tuple[str, str]]]]:
        """参照ファイルを業務分野ごとに分類"""
        logger.info("参照ファイルの分析を開始...")
        
        business_areas = {}
        
        # 履歴データの分析
        faq_files = self._get_files_in_directory(self.reference_faq_path)
        for file in faq_files:
            match = re.match(self.config.REFERENCE_FILE_PATTERN, file)
            if match:
                business, data_type, date = match.groups()
                if business not in business_areas:
                    business_areas[business] = {"faq": [], "scenario": []}
                business_areas[business]["faq"].append((file, date))
                logger.info(f"履歴データ検出: {business} - {file}")
            else:
                logger.warning(f"不正な履歴データファイル名: {file}")
        
        # シナリオデータの分析
        scenario_files = self._get_files_in_directory(self.reference_scenario_path)
        for file in scenario_files:
            match = re.match(self.config.REFERENCE_FILE_PATTERN, file)
            if match:
                business, data_type, date = match.groups()
                if business not in business_areas:
                    business_areas[business] = {"faq": [], "scenario": []}
                business_areas[business]["scenario"].append((file, date))
                logger.info(f"シナリオデータ検出: {business} - {file}")
            else:
                logger.warning(f"不正なシナリオデータファイル名: {file}")
        
        logger.info(f"業務分野検出: {list(business_areas.keys())}")
        return business_areas
    
    def _get_files_in_directory(self, directory: str) -> List[str]:
        """ディレクトリ内のExcelファイルを取得"""
        if not os.path.exists(directory):
            return []
        
        files = []
        for file in os.listdir(directory):
            if file.endswith('.xlsx'):
                files.append(file)
        return files
    
    def get_latest_file(self, files: List[Tuple[str, str]]) -> Optional[str]:
        """最新日付のファイルを取得"""
        if not files:
            return None
        
        # 日付でソートして最新のファイルを返す
        sorted_files = sorted(files, key=lambda x: x[1], reverse=True)
        latest_file = sorted_files[0][0]
        logger.info(f"最新ファイル選択: {latest_file}")
        return latest_file
    
    def needs_update(self, db_path: str, latest_faq: Optional[str], latest_scenario: Optional[str], business_area: str) -> bool:
        """DB更新の必要性をチェック"""
        if not os.path.exists(db_path):
            logger.info(f"DBが存在しないため新規作成: {db_path}")
            return True
        
        # ChromaDBコレクションの存在確認
        try:
            import chromadb
            from chromadb.config import Settings
            
            client = chromadb.PersistentClient(
                path=self.base_db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # コレクション名を取得（パスから抽出）
            collection_name = os.path.basename(db_path)
            
            try:
                collection = client.get_collection(name=collection_name)
                logger.info(f"コレクション存在確認: {collection_name}")
                
                # ファイル更新日時の比較
                needs_update = False
                
                # DBの最終更新日時を取得（コレクションの作成日時を使用）
                try:
                    # コレクション内のドキュメント数をチェック
                    doc_count = collection.count()
                    logger.info(f"コレクション内のドキュメント数: {doc_count}")
                    
                    # ドキュメントが存在する場合のみ最新と判定
                    db_is_current = doc_count > 0
                    if not db_is_current:
                        logger.info("コレクションは存在するが、ドキュメントが存在しないため更新が必要")
                except Exception as e:
                    logger.warning(f"コレクション情報取得エラー: {e}")
                    db_is_current = False
                
                # ファイルの更新日時をチェック
                if latest_faq:
                    faq_path = os.path.join(self.reference_faq_path, latest_faq)
                    if os.path.exists(faq_path):
                        faq_mtime = os.path.getmtime(faq_path)
                        logger.info(f"履歴データファイル更新確認: {latest_faq} (更新日時: {faq_mtime})")
                        
                        # 更新日時をチェックしてDB更新の必要性を判定
                        last_mtime = self._last_faq_mtime.get(business_area, 0)
                        if faq_mtime > last_mtime:
                            needs_update = True
                            logger.info(f"履歴データファイルの更新日時が変更されたため、DB更新が必要 (前回: {last_mtime}, 現在: {faq_mtime})")
                            self._last_faq_mtime[business_area] = faq_mtime
                            # 更新日時を永続化
                            self._save_update_timestamps()
                        elif not db_is_current:
                            needs_update = True
                            logger.info("履歴データファイルが存在するが、DBが最新でないため更新が必要")
                
                if latest_scenario:
                    scenario_path = os.path.join(self.reference_scenario_path, latest_scenario)
                    if os.path.exists(scenario_path):
                        scenario_mtime = os.path.getmtime(scenario_path)
                        logger.info(f"シナリオデータファイル更新確認: {latest_scenario} (更新日時: {scenario_mtime})")
                        
                        # 更新日時をチェックしてDB更新の必要性を判定
                        last_mtime = self._last_scenario_mtime.get(business_area, 0)
                        if scenario_mtime > last_mtime:
                            needs_update = True
                            logger.info(f"シナリオデータファイルの更新日時が変更されたため、DB更新が必要 (前回: {last_mtime}, 現在: {scenario_mtime})")
                            self._last_scenario_mtime[business_area] = scenario_mtime
                            # 更新日時を永続化
                            self._save_update_timestamps()
                        elif not db_is_current:
                            needs_update = True
                            logger.info("シナリオデータファイルが存在するが、DBが最新でないため更新が必要")
                
                # 強制更新フラグのチェック
                if self.config.force_db_update:
                    logger.info(f"強制更新フラグが有効のため、DB更新を実行: {db_path}")
                    return True
                
                if needs_update:
                    logger.info(f"DB更新が必要: {db_path}")
                else:
                    logger.info(f"DBは最新: {db_path}")
                
                return needs_update
                
            except Exception as e:
                logger.info(f"コレクションが存在しません: {collection_name}")
                return True
                
        except Exception as e:
            logger.warning(f"DB更新チェックエラー: {e}")
            return True  # エラーの場合は安全のため更新
    
    def update_business_db(self, business_area: str, files: Dict[str, List[Tuple[str, str]]]):
        """特定業務分野のDBを更新"""
        # 日本語の業務分野名を英語に変換
        english_name = self._translate_business_area(business_area)
        db_name = f"{english_name}_DB"
        db_path = os.path.join(self.base_db_path, db_name)
        
        logger.info(f"業務分野 '{business_area}' のDB更新開始")
        
        # 最新ファイルの選択
        latest_faq = self.get_latest_file(files["faq"])
        latest_scenario = self.get_latest_file(files["scenario"])
        
        # DB更新の必要性チェック
        if self.needs_update(db_path, latest_faq, latest_scenario, business_area):
            try:
                # DBリセットと再ベクトル化
                self._reset_and_revectorize(db_path, business_area, latest_faq, latest_scenario)
                logger.info(f"業務分野 '{business_area}' のDB更新完了")
            except Exception as e:
                logger.error(f"DB更新エラー: {e}")
                raise DynamicDBError(f"DB更新に失敗しました: {e}")
        else:
            logger.info(f"業務分野 '{business_area}' のDBは最新です")
    
    def _reset_and_revectorize(self, db_path: str, business_area: str, 
                              latest_faq: Optional[str], latest_scenario: Optional[str]):
        """DBリセットと再ベクトル化"""
        logger.info(f"DBリセットと再ベクトル化開始: {db_path}")
        
        # ChromaDBの実際の動作に合わせて修正
        # コレクション名のフォルダは空になるが、実際のデータはUUIDフォルダに格納される
        
        # 既存DBの削除（ChromaDBのメタデータから削除）
        self._delete_chromadb_collection(business_area)
        
        # 新しいDBの作成（フォルダは空だが、ChromaDBが自動的にUUIDフォルダを作成）
        os.makedirs(db_path, exist_ok=True)
        logger.info(f"新規DB作成: {db_path}")
        
        # ベクトル化処理（ここでは簡易実装）
        # 実際の実装では、input_handler.pyと連携してベクトル化を実行
        self._vectorize_data(db_path, business_area, latest_faq, latest_scenario)
    
    def _delete_chromadb_collection(self, business_area: str):
        """ChromaDBのコレクションを削除"""
        try:
            import chromadb
            from chromadb.config import Settings
            
            # ChromaDBクライアントの初期化
            client = chromadb.PersistentClient(
                path=self.base_db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # 英語名に変換
            english_name = self._translate_business_area(business_area)
            collection_name = f"{english_name}_DB"
            
            # コレクションが存在する場合は削除
            try:
                collection = client.get_collection(name=collection_name)
                client.delete_collection(name=collection_name)
                logger.info(f"ChromaDBコレクション削除: {collection_name}")
            except ChromaNotFoundError:
                logger.info(f"ChromaDBコレクションは存在しません: {collection_name}")
                
        except Exception as e:
            logger.warning(f"ChromaDBコレクション削除エラー: {e}")
            # エラーが発生しても処理を続行
    
    def _vectorize_data(self, db_path: str, business_area: str, 
                        latest_faq: Optional[str], latest_scenario: Optional[str]):
        """データのベクトル化（既存のMetadataVectorDBクラスを使用）"""
        logger.info(f"ベクトル化処理開始: {business_area}")
        
        try:
            # 英語名に変換
            english_name = self._translate_business_area(business_area)
            collection_name = f"{english_name}_DB"
            
            # 参照データの準備
            reference_data = self._prepare_reference_data_for_vectorization()
            
            # ベクトル化モデルの初期化
            from utils.gemini_embedding import GeminiEmbeddingModel
            embedding_model = GeminiEmbeddingModel(self.config)
            
            # テキストのベクトル化
            texts = reference_data['combined_texts']
            logger.info(f"ベクトル化開始: {len(texts)}件のテキスト")
            
            # バッチサイズで分割してベクトル化
            batch_size = self.config.VECTOR_DB_BATCH_SIZE
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                end_idx = min(i + batch_size, len(texts))
                batch_texts = texts[i:end_idx]
                logger.info(f"バッチ処理中: {i+1}-{end_idx}/{len(texts)}")
                
                batch_embeddings = embedding_model.encode(batch_texts, normalize_embeddings=True)
                all_embeddings.append(batch_embeddings)
            
            # 全バッチの結果を結合
            embeddings = np.concatenate(all_embeddings, axis=0)
            logger.info(f"ベクトル化完了: {len(embeddings)}件のベクトル")
            
            # 既存のMetadataVectorDBクラスを使用してベクトルDBに追加
            from utils.vector_db import MetadataVectorDB
            vector_db = MetadataVectorDB(self.config.base_dir, collection_name)
            
            # メタデータの準備
            metadatas = reference_data.get('metadatas', [])
            
            # 既存のadd_documentsメソッドを使用（メタデータ正規化処理が組み込まれている）
            vector_db.add_documents(
                texts=texts,
                embeddings=embeddings.tolist(),
                metadatas=metadatas
            )
            
            logger.info(f"ベクトル化処理完了: {business_area} - {len(texts)}件のデータを{collection_name}に追加")
            
        except Exception as e:
            logger.error(f"ベクトル化処理エラー: {e}")
            raise DynamicDBError(f"ベクトル化処理に失敗しました: {e}")
    
    def extract_business_area_from_input(self, input_file: str) -> str:
        """入力ファイルから業務分野を抽出"""
        match = re.match(self.config.INPUT_FILE_PATTERN, input_file)
        if match:
            business_area, date = match.groups()
            logger.info(f"入力ファイルから業務分野抽出: {business_area}")
            return business_area
        else:
            raise DynamicDBError(f"不正な入力ファイル名: {input_file}")
    
    def _translate_business_area(self, business_area: str) -> str:
        """業務分野名を英語に変換（ChromaDB制限対応）"""
        translation_map = {
            "総則": "general",
            "預金": "deposit", 
            "融資": "loan",
            "外貨": "foreign_currency",
            "投信": "investment_trust",
            "住宅ローン": "housing_loan",
            "投資信託": "investment_fund",
            "カード": "card",
            "保険": "insurance",
            "年金": "pension"
        }
        
        # 完全一致を優先
        if business_area in translation_map:
            return translation_map[business_area]
        
        # 部分一致で検索
        for japanese, english in translation_map.items():
            if japanese in business_area:
                return english
        
        # デフォルト: 英数字のみに変換
        import re
        sanitized = re.sub(r'[^a-zA-Z0-9._-]', '_', business_area)
        sanitized = re.sub(r'_+', '_', sanitized)  # 連続するアンダースコアを単一に
        sanitized = sanitized.strip('_')  # 先頭・末尾のアンダースコアを除去
        
        if not sanitized:
            sanitized = "default"
        
        return sanitized
    
    def get_db_path_for_business(self, business_area: str) -> str:
        """業務分野に対応するDBパスを取得"""
        # 日本語の業務分野名を英語に変換
        english_name = self._translate_business_area(business_area)
        db_name = f"{english_name}_DB"
        db_path = os.path.join(self.base_db_path, db_name)
        
        # ChromaDBの実際の動作では、コレクション名のフォルダは空になる
        # 実際のデータはUUIDフォルダに格納されるため、フォルダの存在チェックは不要
        # 代わりにChromaDBのメタデータでコレクションの存在を確認
        
        return db_path
    
    def validate_file_name(self, filename: str, pattern: str, file_type: str):
        """ファイル名の妥当性チェック"""
        if not re.match(pattern, filename):
            raise DynamicDBError(
                f"不正な{file_type}ファイル名: {filename}\n"
                f"期待される形式: {pattern}"
            )
    
    def check_db_creation_permission(self, db_path: str):
        """DB作成権限のチェック"""
        try:
            os.makedirs(db_path, exist_ok=True)
        except PermissionError:
            raise DynamicDBError(f"DB作成権限がありません: {db_path}")
    
    def get_all_business_areas(self) -> List[str]:
        """全業務分野の一覧を取得"""
        business_areas = set()
        
        # 既存DBから業務分野を抽出
        if os.path.exists(self.base_db_path):
            for item in os.listdir(self.base_db_path):
                if item.endswith('_DB') and os.path.isdir(os.path.join(self.base_db_path, item)):
                    business_area = item[:-3]  # "_DB"を除去
                    business_areas.add(business_area)
        
        return list(business_areas)
    
    def _prepare_reference_data_for_vectorization(self) -> dict:
        """動的DB管理システム用の参照データ準備（既存実装を活用）"""
        logger.info("動的DB管理システム用の参照データ準備開始（既存実装を活用）")
        
        # 既存のMultiFolderInputHandlerを使用してデータを読み込み
        from input_handler import MultiFolderInputHandler
        input_handler = MultiFolderInputHandler(self.config)
        
        # 参照データの読み込み
        reference_data = input_handler.load_reference_data()
        
        logger.info(f"参照データ準備完了: 総件数{len(reference_data['combined_texts'])}件")
        
        return reference_data
