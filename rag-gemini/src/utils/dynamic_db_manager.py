import os
import re
import shutil
import logging
import json
import pandas as pd
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

        # ChromaDBクライアントを一度だけ初期化（パフォーマンス向上）
        self._chroma_client = chromadb.PersistentClient(
            path=self.base_db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # 埋め込みプロバイダー（DB分離用）
        self.embedding_provider = config.embedding_provider

        # 更新日時記録ファイルのパス（プロバイダー別）
        self.update_timestamp_file = os.path.join(
            self.base_db_path, f"update_timestamps_{self.embedding_provider}.json"
        )

        # 更新日時の読み込み
        self._load_update_timestamps()

        # 既存DBの移行（初回のみ：旧形式→プロバイダー別形式）
        self._migrate_existing_db()

        # リソース管理フラグ
        self._closed = False

    def __enter__(self):
        """コンテキストマネージャ: 開始"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャ: 終了時にリソースをクリーンアップ"""
        self.close()
        return False  # 例外は再スロー

    def close(self):
        """リソースのクリーンアップ"""
        if self._closed:
            return

        try:
            # タイムスタンプを永続化
            self._save_update_timestamps()
            logger.info("DynamicDBManager: リソースをクリーンアップしました")
        except Exception as e:
            logger.warning(f"DynamicDBManager close時のエラー: {e}")
        finally:
            self._closed = True

    def _load_update_timestamps(self):
        """更新日時の記録を読み込み（型検証付き）"""
        try:
            if os.path.exists(self.update_timestamp_file):
                with open(self.update_timestamp_file, 'r', encoding='utf-8') as f:
                    timestamps = json.load(f)

                # 品質: JSONデータの型検証
                if not isinstance(timestamps, dict):
                    logger.warning(f"タイムスタンプファイルの形式が不正です（dict期待）: {type(timestamps)}")
                    self._last_faq_mtime = {}
                    self._last_scenario_mtime = {}
                    return

                faq_data = timestamps.get('faq', {})
                scenario_data = timestamps.get('scenario', {})

                # 各フィールドの型検証
                if not isinstance(faq_data, dict):
                    logger.warning(f"faqフィールドの型が不正です: {type(faq_data)}")
                    faq_data = {}
                if not isinstance(scenario_data, dict):
                    logger.warning(f"scenarioフィールドの型が不正です: {type(scenario_data)}")
                    scenario_data = {}

                # 値の型検証（タイムスタンプはfloat/int）
                self._last_faq_mtime = {
                    k: v for k, v in faq_data.items()
                    if isinstance(k, str) and isinstance(v, (int, float))
                }
                self._last_scenario_mtime = {
                    k: v for k, v in scenario_data.items()
                    if isinstance(k, str) and isinstance(v, (int, float))
                }

                # 無効なエントリがあった場合は警告
                if len(self._last_faq_mtime) != len(faq_data):
                    logger.warning(f"FAQタイムスタンプに無効なエントリがありました（スキップ）")
                if len(self._last_scenario_mtime) != len(scenario_data):
                    logger.warning(f"シナリオタイムスタンプに無効なエントリがありました（スキップ）")

                logger.info(f"更新日時記録を読み込み: FAQ={len(self._last_faq_mtime)}件, シナリオ={len(self._last_scenario_mtime)}件")
            else:
                self._last_faq_mtime = {}
                self._last_scenario_mtime = {}
                logger.info("更新日時記録ファイルが存在しないため、新規作成します")
        except json.JSONDecodeError as e:
            logger.warning(f"タイムスタンプファイルのJSON解析エラー: {e}")
            self._last_faq_mtime = {}
            self._last_scenario_mtime = {}
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

    def _update_timestamps_after_success(
        self, business_area: str, latest_faq: Optional[str], latest_scenario: Optional[str]
    ) -> None:
        """DB更新成功後にタイムスタンプを記録・永続化

        Args:
            business_area: 業務分野名
            latest_faq: 最新の履歴データファイル名
            latest_scenario: 最新のシナリオデータファイル名
        """
        # FAQファイルのタイムスタンプ更新
        if latest_faq:
            faq_path = os.path.join(self.reference_faq_path, latest_faq)
            if os.path.exists(faq_path):
                self._last_faq_mtime[business_area] = os.path.getmtime(faq_path)

        # シナリオファイルのタイムスタンプ更新
        if latest_scenario:
            scenario_path = os.path.join(self.reference_scenario_path, latest_scenario)
            if os.path.exists(scenario_path):
                self._last_scenario_mtime[business_area] = os.path.getmtime(scenario_path)

        # タイムスタンプを永続化
        self._save_update_timestamps()
        logger.info(f"業務分野 '{business_area}' のタイムスタンプを更新しました")

    def _get_collection_name(self, business_area: str) -> str:
        """プロバイダー別のコレクション名を生成

        Args:
            business_area: 業務分野名（日本語）

        Returns:
            str: コレクション名（例: deposit_vertex_ai_DB）
        """
        english_name = self._translate_business_area(business_area)
        return f"{english_name}_{self.embedding_provider}_DB"

    def _migrate_existing_db(self):
        """既存DB移行処理

        注意: プロバイダー別のコレクション名が導入されたため、
        旧形式のデータは移行せず、新規ベクトル化を強制する。
        タイムスタンプのコピーは行わない（空のDBを「最新」と誤判定しないため）。
        """
        # 旧形式のタイムスタンプファイルが存在する場合はバックアップ
        old_timestamp_file = os.path.join(self.base_db_path, "update_timestamps.json")
        if os.path.exists(old_timestamp_file):
            backup_file = old_timestamp_file + ".backup"
            if not os.path.exists(backup_file):
                try:
                    os.rename(old_timestamp_file, backup_file)
                    logger.info(f"旧タイムスタンプファイルをバックアップ: {backup_file}")
                except Exception as e:
                    logger.warning(f"旧タイムスタンプファイルのバックアップに失敗: {e}")

        logger.info(f"プロバイダー '{self.embedding_provider}' 用のDB初期化（新規ベクトル化が必要）")
    
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
        """ディレクトリ内のExcelファイルを取得（一時ファイルを除外）"""
        if not os.path.exists(directory):
            return []

        files = []
        for file in os.listdir(directory):
            # Excelの一時ファイル（~$で始まる）を除外
            if file.endswith('.xlsx') and not file.startswith('~$'):
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

        # 強制更新フラグのチェック（早期リターン）
        if self.config.force_db_update:
            logger.info(f"強制更新フラグが有効のため、DB更新を実行: {db_path}")
            return True

        collection_name = os.path.basename(db_path)

        try:
            collection = self._chroma_client.get_collection(name=collection_name)
            logger.info(f"コレクション存在確認: {collection_name}")
        except Exception:
            logger.info(f"コレクションが存在しません: {collection_name}")
            return True

        # コレクション内のドキュメント数をチェック
        db_is_current = self._check_collection_has_documents(collection)

        # ファイルの更新日時をチェック
        faq_needs_update = self._check_file_needs_update(
            latest_faq, self.reference_faq_path,
            self._last_faq_mtime.get(business_area, 0),
            db_is_current, "履歴データ"
        )
        scenario_needs_update = self._check_file_needs_update(
            latest_scenario, self.reference_scenario_path,
            self._last_scenario_mtime.get(business_area, 0),
            db_is_current, "シナリオデータ"
        )

        needs_update = faq_needs_update or scenario_needs_update
        logger.info(f"DB{'更新が必要' if needs_update else 'は最新'}: {db_path}")
        return needs_update

    def _check_collection_has_documents(self, collection) -> bool:
        """コレクションにドキュメントが存在するかチェック"""
        try:
            doc_count = collection.count()
            logger.info(f"コレクション内のドキュメント数: {doc_count}")
            if doc_count == 0:
                logger.info("コレクションは存在するが、ドキュメントが存在しないため更新が必要")
            return doc_count > 0
        except Exception as e:
            logger.warning(f"コレクション情報取得エラー: {e}")
            return False

    def _check_file_needs_update(
        self, filename: Optional[str], base_path: str,
        last_mtime: float, db_is_current: bool, file_type: str
    ) -> bool:
        """ファイルの更新日時をチェックしてDB更新の必要性を判定"""
        if not filename:
            return False

        file_path = os.path.join(base_path, filename)
        if not os.path.exists(file_path):
            return False

        current_mtime = os.path.getmtime(file_path)
        logger.info(f"{file_type}ファイル更新確認: {filename} (更新日時: {current_mtime})")

        if current_mtime > last_mtime:
            logger.info(f"{file_type}ファイルの更新日時が変更されたため、DB更新が必要 (前回: {last_mtime}, 現在: {current_mtime})")
            return True

        if not db_is_current:
            logger.info(f"{file_type}ファイルが存在するが、DBが最新でないため更新が必要")
            return True

        return False
    
    def update_business_db(self, business_area: str, files: Dict[str, List[Tuple[str, str]]]):
        """特定業務分野のDBを更新"""
        # プロバイダー別のコレクション名を生成
        db_name = self._get_collection_name(business_area)
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

                # 更新成功後にタイムスタンプを記録・永続化
                self._update_timestamps_after_success(business_area, latest_faq, latest_scenario)

                logger.info(f"業務分野 '{business_area}' のDB更新完了")
            except Exception as e:
                logger.error(f"DB更新エラー: {e}")
                raise DynamicDBError(f"DB更新に失敗しました: {e}")
        else:
            logger.info(f"業務分野 '{business_area}' のDBは最新です")

    def preflight_business_db(
        self,
        business_area: str,
        files: Dict[str, List[Tuple[str, str]]],
        sample_size: int = 5,
    ) -> Dict[str, object]:
        """DB更新の事前チェック（プレフライト）

        本番コレクション（例: deposit_DB）には触らず、
        参照データ読込 → 埋め込み生成 → ChromaDBへ少量書込/検索 → クリーンアップ
        までを通して、更新が通りそうかを検証します。

        Args:
            business_area: 業務分野名（日本語）
            files: analyze_reference_files() の戻り値（業務分野別のファイル一覧）
            sample_size: 事前検証に使うサンプル件数

        Returns:
            実施結果のサマリ辞書
        """
        if sample_size <= 0:
            raise DynamicDBError("sample_size must be > 0")

        # 最新ファイルの選択（存在確認も兼ねる）
        latest_faq = self.get_latest_file(files.get("faq", []))
        latest_scenario = self.get_latest_file(files.get("scenario", []))

        if latest_faq:
            faq_path = os.path.join(self.reference_faq_path, latest_faq)
            if not os.path.exists(faq_path):
                raise DynamicDBError(f"FAQファイルが見つかりません: {faq_path}")
        if latest_scenario:
            scenario_path = os.path.join(self.reference_scenario_path, latest_scenario)
            if not os.path.exists(scenario_path):
                raise DynamicDBError(f"シナリオファイルが見つかりません: {scenario_path}")

        # 書き込み権限の軽い検証（ディレクトリに一時ファイルを作れるか）
        try:
            os.makedirs(self.base_db_path, exist_ok=True)
            probe_path = os.path.join(self.base_db_path, ".write_probe")
            with open(probe_path, "w", encoding="utf-8") as f:
                f.write("ok")
            os.remove(probe_path)
        except Exception as e:
            raise DynamicDBError(f"DB保存先への書き込み権限/ロックを確認してください: {e}")

        # 参照データの読み込み（実際の更新と同じ経路）
        reference_data = self._prepare_reference_data_for_vectorization()
        texts: List[str] = reference_data.get("combined_texts", [])
        metadatas: List[dict] = reference_data.get("metadatas", [])

        if not texts:
            raise DynamicDBError("参照データが空です（combined_texts が 0 件）")

        if metadatas and len(metadatas) != len(texts):
            raise DynamicDBError(
                f"参照データ不整合: combined_texts={len(texts)}件, metadatas={len(metadatas)}件"
            )

        effective_sample_size = min(sample_size, len(texts))
        sample_texts = texts[:effective_sample_size]
        if metadatas:
            sample_metadatas = metadatas[:effective_sample_size]
        else:
            sample_metadatas = [{"source": "preflight"} for _ in range(effective_sample_size)]

        # 埋め込み生成（API到達性 + 形状検証）
        try:
            from src.utils.auth import create_embedding_model

            embedding_model = create_embedding_model(self.config)
            sample_embeddings = embedding_model.encode(sample_texts, normalize_embeddings=True)
        except Exception as e:
            raise DynamicDBError(f"埋め込み生成の事前検証に失敗しました: {e}")

        if getattr(sample_embeddings, "ndim", 0) != 2 or sample_embeddings.shape[0] != effective_sample_size:
            raise DynamicDBError(f"埋め込みの形状が不正です: shape={getattr(sample_embeddings, 'shape', None)}")

        # ChromaDBへの少量書込/検索/削除（本番コレクションは触らない）
        english_name = self._translate_business_area(business_area)
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        # ChromaDB のコレクション名制約: 3-512 chars in [a-zA-Z0-9._-] かつ先頭末尾が英数字
        # プロバイダー情報も含める
        temp_collection_name = f"preflight_{english_name}_{self.embedding_provider}_DB_{timestamp}"

        try:
            from src.utils.vector_db import MetadataVectorDB

            vector_db = MetadataVectorDB(self.config.base_dir, temp_collection_name)
            vector_db.add_documents(
                texts=sample_texts,
                embeddings=sample_embeddings.tolist(),
                metadatas=sample_metadatas,
                ids=[f"preflight_{i}" for i in range(effective_sample_size)],
            )

            # 1件だけ検索して、クエリが動作するか確認
            query_embedding = sample_embeddings[0].tolist()
            results = vector_db.search(query_embedding=query_embedding, n_results=min(3, effective_sample_size))
            if not results:
                raise DynamicDBError("ChromaDBへの書込は成功したが、検索結果が0件でした")

        except Exception as e:
            raise DynamicDBError(f"ChromaDB書込/検索の事前検証に失敗しました: {e}")
        finally:
            # 一時コレクションを必ず削除
            try:
                self._chroma_client.delete_collection(name=temp_collection_name)
            except Exception:
                pass

        return {
            "business_area": business_area,
            "latest_faq": latest_faq,
            "latest_scenario": latest_scenario,
            "reference_texts": len(texts),
            "sample_size": effective_sample_size,
            "embedding_dim": int(sample_embeddings.shape[1]),
            "temp_collection": temp_collection_name,
            "status": "ok",
        }
    
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
        """ChromaDBのコレクションを削除（キャッシュされたクライアントを使用）"""
        try:
            # プロバイダー別のコレクション名を生成
            collection_name = self._get_collection_name(business_area)

            # コレクションが存在する場合は削除
            try:
                self._chroma_client.get_collection(name=collection_name)
                self._chroma_client.delete_collection(name=collection_name)
                logger.info(f"ChromaDBコレクション削除: {collection_name}")
            except ChromaNotFoundError:
                logger.info(f"ChromaDBコレクションは存在しません: {collection_name}")

        except Exception as e:
            logger.warning(f"ChromaDBコレクション削除エラー: {e}")
            # エラーが発生しても処理を続行
    
    def _vectorize_data(self, db_path: str, business_area: str,
                        latest_faq: Optional[str], latest_scenario: Optional[str]):
        """データのベクトル化（ストリーミング書き込みでメモリ効率化）"""
        logger.info(f"ベクトル化処理開始: {business_area} (プロバイダー: {self.embedding_provider})")

        try:
            # プロバイダー別のコレクション名を生成
            collection_name = self._get_collection_name(business_area)

            # 参照データの準備
            reference_data = self._prepare_reference_data_for_vectorization()

            # ベクトル化モデルの初期化（プロバイダー設定に応じて自動切替）
            from src.utils.auth import create_embedding_model
            embedding_model = create_embedding_model(self.config)

            # テキストとメタデータの準備
            texts = reference_data['combined_texts']
            metadatas = reference_data.get('metadatas', [])
            logger.info(f"ベクトル化開始: {len(texts)}件のテキスト")

            # パフォーマンス: VectorDBを先に初期化
            from src.utils.vector_db import MetadataVectorDB
            vector_db = MetadataVectorDB(self.config.base_dir, collection_name)

            # パフォーマンス: バッチごとにベクトル化してストリーミング書き込み（メモリ効率化）
            batch_size = self.config.VECTOR_DB_BATCH_SIZE
            total_added = 0

            for i in range(0, len(texts), batch_size):
                end_idx = min(i + batch_size, len(texts))
                batch_texts = texts[i:end_idx]
                batch_metadatas = metadatas[i:end_idx] if metadatas else []
                logger.info(f"バッチ処理中: {i+1}-{end_idx}/{len(texts)}")

                # バッチの埋め込みを生成
                batch_embeddings = embedding_model.encode(batch_texts, normalize_embeddings=True)

                # バッチIDを生成（オフセット付き）
                batch_ids = [f"doc_{j}" for j in range(i, end_idx)]

                # その場でDBに書き込み（numpy配列を直接渡す - add_documents内で変換）
                vector_db.add_documents(
                    texts=batch_texts,
                    embeddings=batch_embeddings,  # numpy配列を直接渡す
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                total_added += len(batch_texts)

                # メモリ解放のヒント（GCに任せる）
                del batch_embeddings

            logger.info(f"ベクトル化処理完了: {business_area} - {total_added}件のデータを{collection_name}に追加")

        except Exception as e:
            logger.error(f"ベクトル化処理エラー: {e}")
            raise DynamicDBError(f"ベクトル化処理に失敗しました: {e}")
    
    def extract_business_area_from_input(self, input_file: str) -> str:
        """入力ファイルから業務分野を抽出（セキュリティ強化版）"""
        import unicodedata

        # セキュリティ: ファイル名のみを抽出（パストラバーサル防止）
        filename = os.path.basename(input_file)

        # セキュリティ: Unicode正規化（ホモグラフ攻撃防止）
        filename = unicodedata.normalize('NFKC', filename)

        match = re.match(self.config.INPUT_FILE_PATTERN, filename)
        if not match:
            raise DynamicDBError(
                f"不正な入力ファイル名: {filename}\n"
                f"期待される形式: {self.config.INPUT_FILE_PATTERN}"
            )

        business_area, date = match.groups()

        # サニタイズ前に空チェック
        if not business_area or len(business_area.strip()) == 0:
            raise DynamicDBError(f"業務分野が空です: {filename}")

        # セキュリティ: パストラバーサル文字を除去
        business_area = business_area.replace('..', '').replace('/', '').replace('\\', '')

        # 空白、制御文字、ゼロ幅文字を除去
        ZERO_WIDTH_CHARS = '\u200B\u200C\u200D\uFEFF\u00AD'
        business_area = ''.join(
            c for c in business_area
            if c.isprintable() and c not in ' \t\n\r\x00' and c not in ZERO_WIDTH_CHARS
        )

        # サニタイズ後に空チェック
        if not business_area:
            raise DynamicDBError(f"サニタイズ後の業務分野が空です: {filename}")

        logger.info(f"入力ファイルから業務分野抽出: {business_area}")
        return business_area
    
    def _translate_business_area(self, business_area: str) -> str:
        """業務分野名を英語に変換（ChromaDB制限対応・セキュリティ強化版）"""
        # ChromaDB の制約: コレクション名は 3-512 文字
        MAX_COLLECTION_NAME_LENGTH = 512
        MIN_COLLECTION_NAME_LENGTH = 3

        if len(business_area) > MAX_COLLECTION_NAME_LENGTH:
            logger.warning(f"Business area name too long: {len(business_area)} chars, truncating")
            business_area = business_area[:MAX_COLLECTION_NAME_LENGTH]

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

        # デフォルト: 英数字のみに変換（re はトップレベルでインポート済み）
        sanitized = re.sub(r'[^a-zA-Z0-9._-]', '_', business_area)
        sanitized = re.sub(r'_+', '_', sanitized).strip('_')

        # ChromaDB の制約: 先頭末尾が英数字であること
        if sanitized and not sanitized[0].isalnum():
            sanitized = 'c' + sanitized
        if sanitized and not sanitized[-1].isalnum():
            sanitized = sanitized + 'c'

        # 最小長チェック
        if len(sanitized) < MIN_COLLECTION_NAME_LENGTH:
            sanitized = 'default_collection'

        return sanitized if sanitized else "default"
    
    def get_db_path_for_business(self, business_area: str) -> str:
        """業務分野に対応するDBパスを取得（プロバイダー別）"""
        # プロバイダー別のコレクション名を生成
        db_name = self._get_collection_name(business_area)
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
        from src.handlers.input_handler import MultiFolderInputHandler
        input_handler = MultiFolderInputHandler(self.config)
        
        # 参照データの読み込み
        reference_data = input_handler.load_reference_data()
        
        logger.info(f"参照データ準備完了: 総件数{len(reference_data['combined_texts'])}件")
        
        return reference_data
