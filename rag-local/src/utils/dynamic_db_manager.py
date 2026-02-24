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
from src.utils.business_area_translator import BusinessAreaTranslator
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class DynamicDBError(Exception):
    """動的DB管理のエラー"""
    pass

class DynamicDBManager:
    """動的DB管理システム"""
    
    def __init__(self, config: SearchConfig):
        self.config = config
        self.base_db_path = os.path.join(config.base_dir, "data", "vector_db")
        self.reference_faq_path = os.path.join(config.base_dir, "data", "source", "faq", "latest")
        self.reference_scenario_path = os.path.join(config.base_dir, "data", "source", "scenarios", "revisions")
        self._translator = BusinessAreaTranslator()

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

        # 更新日時記録ファイルのパス（共通1ファイル）
        self.update_timestamp_file = os.path.join(
            self.base_db_path, "update_timestamps.json"
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
        """リソースのクリーンアップ（ChromaDBクライアント含む）"""
        if self._closed:
            return

        try:
            # タイムスタンプを永続化
            self._save_update_timestamps()

            # Resource Leak防止: ChromaDBクライアントのクリーンアップ
            if hasattr(self, '_chroma_client') and self._chroma_client is not None:
                try:
                    # ChromaDB PersistentClient はexplicitなcloseメソッドを持たないが、
                    # 内部のサーバー参照をクリアすることでリソースを解放
                    if hasattr(self._chroma_client, '_server'):
                        self._chroma_client._server = None
                    # Windowsでのファイルロック問題を軽減するため参照をクリア
                    self._chroma_client = None
                    logger.info("DynamicDBManager: ChromaDBクライアントをクリーンアップしました")
                except Exception as e:
                    logger.warning(f"ChromaDBクライアントのクリーンアップに失敗: {e}")

            logger.info("DynamicDBManager: リソースをクリーンアップしました")
        except Exception as e:
            logger.warning(f"DynamicDBManager close時のエラー: {e}")
        finally:
            self._closed = True

    def _load_update_timestamps(self):
        """更新日時の記録を読み込み（フラット構造: "{area}_{provider}_{type}" → timestamp）"""
        try:
            if os.path.exists(self.update_timestamp_file):
                with open(self.update_timestamp_file, 'r', encoding='utf-8') as f:
                    timestamps = json.load(f)

                if not isinstance(timestamps, dict):
                    logger.warning(f"タイムスタンプファイルの形式が不正です（dict期待）: {type(timestamps)}")
                    self._last_faq_mtime = {}
                    self._last_scenario_mtime = {}
                    return

                self._last_faq_mtime = {}
                self._last_scenario_mtime = {}

                # フラット構造を検出: キーに "_faq" or "_scenario" が含まれるか
                is_flat = any("_faq" in k or "_scenario" in k for k in timestamps.keys())

                if is_flat:
                    # フラット構造: "{area}_{provider}_{type}" → timestamp
                    suffix_faq = f"_{self.embedding_provider}_faq"
                    suffix_scenario = f"_{self.embedding_provider}_scenario"
                    for key, value in timestamps.items():
                        if not isinstance(value, (int, float)):
                            continue
                        if key.endswith(suffix_faq):
                            area = key[:-len(suffix_faq)]
                            self._last_faq_mtime[area] = value
                        elif key.endswith(suffix_scenario):
                            area = key[:-len(suffix_scenario)]
                            self._last_scenario_mtime[area] = value
                else:
                    # 旧3階層構造からの移行読み込み
                    for business_area, providers in timestamps.items():
                        if not isinstance(providers, dict):
                            continue
                        provider_data = providers.get(self.embedding_provider, {})
                        if isinstance(provider_data, dict):
                            if 'faq' in provider_data and isinstance(provider_data['faq'], (int, float)):
                                self._last_faq_mtime[business_area] = provider_data['faq']
                            if 'scenario' in provider_data and isinstance(provider_data['scenario'], (int, float)):
                                self._last_scenario_mtime[business_area] = provider_data['scenario']
                    # 旧形式を検出した場合、次回保存時にフラット形式に自動移行
                    if self._last_faq_mtime or self._last_scenario_mtime:
                        logger.info("旧3階層形式を検出。次回保存時にフラット形式に移行します")

                logger.info(f"更新日時記録を読み込み: FAQ={len(self._last_faq_mtime)}件, シナリオ={len(self._last_scenario_mtime)}件 (プロバイダー: {self.embedding_provider})")

                # 旧キー名を現行のDB名に移行
                self._migrate_timestamp_keys()
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
    
    def _migrate_timestamp_keys(self):
        """旧キー名を現行のDB名に移行（日本語→英語、rev系アンダースコア統一）

        例:
          "スマイル" → "smile"
          "総則"    → "general"
          "rev03smile" → "rev03_smile"
        """
        migrated = False

        for mtime_dict in [self._last_faq_mtime, self._last_scenario_mtime]:
            new_dict = {}
            for area, timestamp in mtime_dict.items():
                new_area = self._normalize_timestamp_key(area)
                if new_area != area:
                    migrated = True
                    logger.info(f"タイムスタンプキー移行: '{area}' → '{new_area}'")
                # 同一キーが複数の旧名から移行される場合は新しい方を優先
                if new_area in new_dict:
                    new_dict[new_area] = max(new_dict[new_area], timestamp)
                else:
                    new_dict[new_area] = timestamp
            mtime_dict.clear()
            mtime_dict.update(new_dict)

        if migrated:
            # JSONファイルからも旧キーを除去して書き直す
            self._cleanup_stale_timestamp_file()
            logger.info("タイムスタンプキーの移行が完了しました")

    def _normalize_timestamp_key(self, area: str) -> str:
        """タイムスタンプキーを現行のDB名に正規化"""
        # 1. translatorで変換（日本語→英語、既知のrev系はパススルー）
        translated = self._translator.translate(area)
        if translated != area:
            return translated

        # 2. rev系のアンダースコア補完: "rev03smile" → "rev03_smile"
        if area.startswith("rev") and area not in self._translator.revision_mappings:
            area_stripped = area.replace("_", "")
            for rev_key in self._translator.revision_mappings:
                if rev_key.replace("_", "") == area_stripped:
                    return rev_key

        return area

    def _cleanup_stale_timestamp_file(self):
        """JSONファイル内の旧キーを正規化して書き直す"""
        try:
            if not os.path.exists(self.update_timestamp_file):
                return

            with open(self.update_timestamp_file, 'r', encoding='utf-8') as f:
                raw_timestamps = json.load(f)

            if not isinstance(raw_timestamps, dict):
                return

            # 各プロバイダー・タイプの組み合わせでサフィックスを検出し、エリア部分を正規化
            normalized = {}
            for key, value in raw_timestamps.items():
                if not isinstance(value, (int, float)):
                    normalized[key] = value
                    continue

                # "{area}_{provider}_{type}" 形式からエリア部分を抽出
                new_key = key
                for suffix in ["_faq", "_scenario"]:
                    idx = key.rfind(suffix)
                    if idx > 0:
                        # suffix の前にプロバイダー名がある: "{area}_{provider}_faq"
                        area_and_provider = key[:idx]
                        type_part = key[idx:]
                        # プロバイダー名を逆引き（末尾から探す）
                        for provider in ["azure_openai", "vertex_ai", "gemini"]:
                            prov_suffix = f"_{provider}"
                            if area_and_provider.endswith(prov_suffix):
                                area = area_and_provider[:-len(prov_suffix)]
                                new_area = self._normalize_timestamp_key(area)
                                new_key = f"{new_area}{prov_suffix}{type_part}"
                                break
                        break

                # 重複キーは新しい方を優先
                if new_key in normalized:
                    normalized[new_key] = max(normalized[new_key], value)
                else:
                    normalized[new_key] = value

            with open(self.update_timestamp_file, 'w', encoding='utf-8') as f:
                json.dump(normalized, f, ensure_ascii=False, indent=2)

            logger.info(f"タイムスタンプファイルをクリーンアップ: {len(raw_timestamps)}件 → {len(normalized)}件")

        except Exception as e:
            logger.warning(f"タイムスタンプファイルのクリーンアップエラー: {e}")

    def _save_update_timestamps(self):
        """更新日時の記録を保存（フラット構造: "{area}_{provider}_{type}" → timestamp）"""
        try:
            # 既存のタイムスタンプファイルを読み込み（他のプロバイダーのデータを保持）
            existing_timestamps = {}
            if os.path.exists(self.update_timestamp_file):
                try:
                    with open(self.update_timestamp_file, 'r', encoding='utf-8') as f:
                        existing_timestamps = json.load(f)
                    if not isinstance(existing_timestamps, dict):
                        existing_timestamps = {}
                    # 旧3階層形式が残っていたらクリア（フラット形式に完全移行）
                    if any(isinstance(v, dict) for v in existing_timestamps.values()):
                        logger.info("旧3階層形式をフラット形式に移行します")
                        existing_timestamps = {}
                except Exception as e:
                    logger.warning(f"既存タイムスタンプの読み込みエラー: {e}")
                    existing_timestamps = {}

            # フラット構造で保存: "{area}_{provider}_{type}" → timestamp
            for business_area in set(list(self._last_faq_mtime.keys()) + list(self._last_scenario_mtime.keys())):
                if business_area in self._last_faq_mtime:
                    key = f"{business_area}_{self.embedding_provider}_faq"
                    existing_timestamps[key] = self._last_faq_mtime[business_area]
                if business_area in self._last_scenario_mtime:
                    key = f"{business_area}_{self.embedding_provider}_scenario"
                    existing_timestamps[key] = self._last_scenario_mtime[business_area]

            with open(self.update_timestamp_file, 'w', encoding='utf-8') as f:
                json.dump(existing_timestamps, f, ensure_ascii=False, indent=2)
            logger.info(f"更新日時記録を保存: FAQ={len(self._last_faq_mtime)}件, シナリオ={len(self._last_scenario_mtime)}件 (プロバイダー: {self.embedding_provider})")
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
        """固定のコレクション名を返す（新構造では常に'default'）

        Args:
            business_area: 業務分野名（互換性のため保持、使用されない）

        Returns:
            str: 固定のコレクション名 'default'
        """
        # 新構造では業務分野とプロバイダーはディレクトリで分離されるため、
        # コレクション名は常に'default'
        return "default"

    def _create_chromadb_client(self, db_path: str) -> chromadb.PersistentClient:
        """ChromaDB PersistentClient を標準設定で作成（DRY: 設定を一元管理）

        Args:
            db_path: ChromaDBデータベースのパス

        Returns:
            chromadb.PersistentClient インスタンス
        """
        return chromadb.PersistentClient(
            path=db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

    def _cleanup_chromadb_client(self, client) -> None:
        """ChromaDB クライアントのリソースをクリーンアップ（Resource Leak防止）

        Args:
            client: クリーンアップするChromaDBクライアント
        """
        if client is not None and hasattr(client, '_server') and client._server is not None:
            try:
                client._server = None
            except Exception as e:
                logger.warning(f"ChromaDB client cleanup warning: {e}")

    def _migrate_existing_db(self):
        """既存DB移行処理（階層構造への移行）

        旧形式（単一chroma.sqlite3 + プロバイダー別タイムスタンプファイル）から
        新形式（業務分野/プロバイダー階層構造）への移行を行う。
        旧データは移行せず、新規ベクトル化を強制する。
        """
        # 旧形式のプロバイダー別タイムスタンプファイルをバックアップ
        old_provider_timestamp = os.path.join(
            self.base_db_path, f"update_timestamps_{self.embedding_provider}.json"
        )
        if os.path.exists(old_provider_timestamp):
            backup_file = old_provider_timestamp + ".backup"
            if not os.path.exists(backup_file):
                try:
                    shutil.copy2(old_provider_timestamp, backup_file)
                    os.remove(old_provider_timestamp)
                    logger.info(f"旧プロバイダー別タイムスタンプファイルをバックアップ: {backup_file}")
                except Exception as e:
                    logger.warning(f"旧タイムスタンプファイルのバックアップに失敗: {e}")

        logger.info(f"プロバイダー '{self.embedding_provider}' 用のDB初期化（階層構造）")
    
    def _normalize_business_name(self, raw_name: str) -> str:
        """ファイル名から抽出した業務分野名をDB互換名に正規化

        日本語名（スマイル等）はDBディレクトリ名（smile等）に変換。
        rev系（rev01_smile等）はそのまま返す。
        """
        translated = self._translator.translate(raw_name)
        return translated

    def analyze_reference_files(self) -> Dict[str, Dict[str, List[Tuple[str, str]]]]:
        """参照ファイルを業務分野ごとに分類（DB互換名をキーとして返す）"""
        logger.info("参照ファイルの分析を開始...")

        business_areas = {}

        # 履歴データの分析
        faq_files = self._get_files_in_directory(self.reference_faq_path)
        for file in faq_files:
            match = re.match(self.config.REFERENCE_FILE_PATTERN, file)
            if match:
                raw_business, data_type, date = match.groups()
                business = self._normalize_business_name(raw_business)
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
                raw_business, data_type, date = match.groups()
                business = self._normalize_business_name(raw_business)
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
        """DB更新の必要性をチェック（階層構造対応）"""
        # 新構造: db_path/chroma.sqlite3 の存在をチェック
        sqlite_path = os.path.join(db_path, "chroma.sqlite3")
        if not os.path.exists(sqlite_path):
            logger.info(f"DBファイルが存在しないため新規作成: {sqlite_path}")
            return True

        # 強制更新フラグのチェック（早期リターン）
        if self.config.force_db_update:
            logger.info(f"強制更新フラグが有効のため、DB更新を実行: {db_path}")
            return True

        # 新構造: コレクション名は固定で "default"
        collection_name = "default"

        # 新構造: 専用のChromaDBクライアントでコレクションを確認（Resource Leak防止）
        temp_client = None
        try:
            temp_client = self._create_chromadb_client(db_path)
            collection = temp_client.get_collection(name=collection_name)
            logger.info(f"コレクション存在確認: {collection_name} in {db_path}")

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

        except (ValueError, ChromaNotFoundError):
            logger.info(f"コレクションが存在しません: {collection_name} in {db_path}")
            return True
        except Exception as e:
            logger.warning(f"コレクション確認エラー: {e}")
            return True
        finally:
            self._cleanup_chromadb_client(temp_client)

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
        logger.info(f"業務分野 '{business_area}' のDB更新開始")

        # 階層構造対応: 業務分野とプロバイダーに対応するDBパスを取得
        db_path = self.get_db_path_for_business(business_area)
        
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

        # 階層構造対応: ビジネス領域のdb_pathを取得
        preflight_db_path = self.get_db_path_for_business(business_area)

        try:
            from src.utils.vector_db import MetadataVectorDB

            vector_db = MetadataVectorDB(collection_name=temp_collection_name, db_path=preflight_db_path)
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
            # 一時コレクションを必ず削除（階層構造対応: Resource Leak防止）
            cleanup_client = None
            try:
                cleanup_client = self._create_chromadb_client(preflight_db_path)
                cleanup_client.delete_collection(name=temp_collection_name)
            except Exception:
                pass
            finally:
                self._cleanup_chromadb_client(cleanup_client)

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
        """ChromaDBのコレクションを削除（階層構造対応: 専用クライアントを使用）"""
        temp_client = None
        try:
            # 階層構造のdb_pathを取得
            db_path = self.get_db_path_for_business(business_area)
            collection_name = self._get_collection_name(business_area)

            # 専用クライアントでコレクションを削除（Resource Leak防止）
            temp_client = self._create_chromadb_client(db_path)
            temp_client.get_collection(name=collection_name)
            temp_client.delete_collection(name=collection_name)
            logger.info(f"ChromaDBコレクション削除: {collection_name} in {db_path}")

        except ChromaNotFoundError:
            logger.info(f"ChromaDBコレクションは存在しません: {collection_name} in {db_path}")
        except Exception as e:
            logger.warning(f"ChromaDBコレクション削除エラー: {e}")
            # エラーが発生しても処理を続行
        finally:
            self._cleanup_chromadb_client(temp_client)
    
    def _vectorize_data(self, db_path: str, business_area: str,
                        latest_faq: Optional[str], latest_scenario: Optional[str]):
        """データのベクトル化（ストリーミング書き込みでメモリ効率化）"""
        logger.info(f"ベクトル化処理開始: {business_area} (プロバイダー: {self.embedding_provider})")

        try:
            # プロバイダー別のコレクション名を生成
            collection_name = self._get_collection_name(business_area)

            # 参照データの準備（業務分野に対応するファイルのみ読み込み）
            reference_data = self._prepare_reference_data_for_vectorization(latest_scenario, latest_faq)

            # ベクトル化モデルの初期化（プロバイダー設定に応じて自動切替）
            from src.utils.auth import create_embedding_model
            embedding_model = create_embedding_model(self.config)

            # テキストとメタデータの準備
            texts = reference_data['combined_texts']
            metadatas = reference_data.get('metadatas', [])
            logger.info(f"ベクトル化開始: {len(texts)}件のテキスト")

            # パフォーマンス: VectorDBを先に初期化（階層構造対応: db_pathを直接指定）
            from src.utils.vector_db import MetadataVectorDB
            vector_db = MetadataVectorDB(collection_name=collection_name, db_path=db_path)

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
        """業務分野名を英語に変換（BusinessAreaTranslatorに委譲）"""
        return self._translator.translate(business_area)
    
    def get_db_path_for_business(self, business_area: str) -> str:
        """業務分野とプロバイダーに対応するDBパスを取得（階層構造対応）

        Args:
            business_area: 業務分野名（日本語）

        Returns:
            str: DBディレクトリパス（例: reference/vector_db/general/vertex_ai/）
        """
        # 業務分野名を英語に変換
        english_name = self._translate_business_area(business_area)

        # 階層的パス生成: {base}/{business}/{provider}/
        db_path = os.path.join(
            self.base_db_path,
            english_name,
            self.embedding_provider
        )

        # ディレクトリが存在しない場合は作成
        os.makedirs(db_path, exist_ok=True)

        return db_path
    
    def get_all_business_areas(self, include_revisions: bool = True) -> List[str]:
        """全業務分野の一覧を取得（新旧両構造対応）

        Args:
            include_revisions: Falseの場合、改定別コレクション（rev*）を除外

        新構造: {business}/{provider}/chroma.sqlite3
        旧構造: {business}_DB/chroma.sqlite3
        """
        business_areas = set()

        if not os.path.exists(self.base_db_path):
            return list(business_areas)

        for item in os.listdir(self.base_db_path):
            item_path = os.path.join(self.base_db_path, item)
            if not os.path.isdir(item_path):
                continue

            # 旧構造: {business}_DB/
            if item.endswith('_DB'):
                business_area = item[:-3]  # "_DB"を除去
                if include_revisions or not business_area.startswith("rev"):
                    business_areas.add(business_area)
                continue

            # 新構造: {business}/{provider}/chroma.sqlite3
            if not include_revisions and item.startswith("rev"):
                continue

            # プロバイダーサブディレクトリをチェック
            for provider in self.config.VALID_EMBEDDING_PROVIDERS:
                provider_path = os.path.join(item_path, provider)
                db_file = os.path.join(provider_path, "chroma.sqlite3")
                if os.path.exists(db_file):
                    business_areas.add(item)
                    break

        return sorted(list(business_areas))
    
    def _prepare_reference_data_for_vectorization(
        self,
        latest_scenario: Optional[str] = None,
        latest_faq: Optional[str] = None
    ) -> dict:
        """動的DB管理システム用の参照データ準備（業務分野フィルタリング対応）

        Args:
            latest_scenario: 読み込むシナリオファイル名
            latest_faq: 読み込むFAQファイル名

        Returns:
            dict: 参照データ（combined_texts, metadatas）
        """
        logger.info(f"参照データ準備開始 (シナリオ: {latest_scenario}, FAQ: {latest_faq})")

        # どちらも指定されていない場合は従来の動作
        if not latest_scenario and not latest_faq:
            from src.handlers.input_handler import MultiFolderInputHandler
            input_handler = MultiFolderInputHandler(self.config)
            reference_data = input_handler.load_reference_data()
            logger.info(f"参照データ準備完了（全データ）: {len(reference_data['combined_texts'])}件")
            return reference_data

        all_queries = []
        all_answers = []
        all_metadatas = []

        # シナリオデータの読み込み
        if latest_scenario:
            from src.handlers.input_handler import HierarchicalExcelInputHandler
            scenario_path = os.path.join(self.reference_scenario_path, latest_scenario)
            if os.path.exists(scenario_path):
                handler = HierarchicalExcelInputHandler(self.config, scenario_path)
                scenario_data = handler.load_reference_data()
                all_queries.extend(scenario_data['queries'])
                all_answers.extend(scenario_data['answers'])
                all_metadatas.extend(scenario_data['metadatas'])
                logger.info(f"シナリオデータ読み込み完了: {len(scenario_data['queries'])}件")
            else:
                logger.warning(f"シナリオファイルが見つかりません: {scenario_path}")

        # FAQデータの読み込み
        if latest_faq:
            faq_path = os.path.join(self.reference_faq_path, latest_faq)
            if os.path.exists(faq_path):
                faq_data = self._load_faq_file(faq_path)
                all_queries.extend(faq_data['queries'])
                all_answers.extend(faq_data['answers'])
                all_metadatas.extend(faq_data['metadatas'])
                logger.info(f"FAQデータ読み込み完了: {len(faq_data['queries'])}件")
            else:
                logger.warning(f"FAQファイルが見つかりません: {faq_path}")

        # combined_textsを生成
        all_combined_texts = []
        for query, answer, metadata in zip(all_queries, all_answers, all_metadatas):
            hierarchy = metadata.get('hierarchy', '') if metadata else ''
            text_parts = []
            if hierarchy.strip():
                text_parts.append(f"分類: {hierarchy}")
            if query.strip():
                text_parts.append(f"質問: {query}")
            if answer.strip():
                text_parts.append(f"回答: {answer}")
            combined_text = " | ".join(text_parts) if text_parts else ""
            all_combined_texts.append(combined_text)

        logger.info(f"参照データ準備完了: {len(all_combined_texts)}件")

        return {
            'queries': all_queries,
            'answers': all_answers,
            'combined_texts': all_combined_texts,
            'metadatas': all_metadatas
        }

    def _load_faq_file(self, faq_path: str) -> dict:
        """特定のFAQファイルを読み込み

        Args:
            faq_path: FAQファイルのフルパス

        Returns:
            dict: FAQデータ（queries, answers, combined_texts, metadatas）
        """
        logger.info(f"FAQファイル読み込み: {faq_path}")
        reference_df = pd.read_excel(faq_path)

        # 列名の検索ロジック
        possible_query_cols = ['分割後質問', '問合せ内容', '質問内容', '問い合わせ', '質問', 'query', 'Query']
        possible_answer_cols = ['分割後回答', '回答', '既存回答', 'answer', 'Answer']
        possible_supplement_cols = ['補足回答', '補足', 'supplement', 'Supplement']

        query_col = next((c for c in possible_query_cols if c in reference_df.columns), None)
        answer_col = next((c for c in possible_answer_cols if c in reference_df.columns), None)
        supplement_col = next((c for c in possible_supplement_cols if c in reference_df.columns), None)

        if query_col is None or answer_col is None:
            raise DynamicDBError(f"FAQファイルに必須列が見つかりません: {list(reference_df.columns)}")

        logger.info(f"FAQ列検出: Query='{query_col}', Answer='{answer_col}', Supplement='{supplement_col}'")

        queries = []
        answers = []
        combined_texts = []
        metadatas = []

        for idx, row in reference_df.iterrows():
            query_text = str(row[query_col]) if pd.notna(row[query_col]) else ""
            answer_text = str(row[answer_col]) if pd.notna(row[answer_col]) else ""
            supplement_text = str(row[supplement_col]) if supplement_col and pd.notna(row[supplement_col]) else ""

            # 補足回答を回答にマージ
            if supplement_text.strip():
                if answer_text.strip():
                    answer_text = f"{answer_text}\n{supplement_text}"
                else:
                    answer_text = supplement_text

            # combined_text生成
            text_parts = []
            if query_text.strip():
                text_parts.append(f"質問: {query_text}")
            if answer_text.strip():
                text_parts.append(f"回答: {answer_text}")
            combined_texts.append(" | ".join(text_parts) if text_parts else "")

            queries.append(query_text)
            answers.append(answer_text)

            metadatas.append({
                'source': 'history_data',
                'row_index': idx
            })

        logger.info(f"FAQファイル読み込み完了: {len(queries)}件")

        return {
            'queries': queries,
            'answers': answers,
            'combined_texts': combined_texts,
            'metadatas': metadatas
        }
