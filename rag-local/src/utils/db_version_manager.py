# --- src/utils/db_version_manager.py ---
"""DBバージョン管理

タイムスタンプ管理を簡素化し、一元的なバージョン管理を提供。
"""

import os
import json
from dataclasses import dataclass, field
from typing import Dict, Optional
from datetime import datetime

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class DBVersionInfo:
    """単一DBのバージョン情報

    Attributes:
        faq_mtime: FAQファイルの最終更新時刻（Unix timestamp）
        scenario_mtime: シナリオファイルの最終更新時刻（Unix timestamp）
        last_updated: DBの最終更新日時（ISO形式）
    """
    faq_mtime: Optional[float] = None
    scenario_mtime: Optional[float] = None
    last_updated: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        """辞書形式に変換"""
        result = {}
        if self.faq_mtime is not None:
            result['faq'] = self.faq_mtime
        if self.scenario_mtime is not None:
            result['scenario'] = self.scenario_mtime
        if self.last_updated is not None:
            result['last_updated'] = self.last_updated
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> 'DBVersionInfo':
        """辞書から生成"""
        return cls(
            faq_mtime=data.get('faq'),
            scenario_mtime=data.get('scenario'),
            last_updated=data.get('last_updated'),
        )


class DBVersionManager:
    """DBバージョン管理クラス

    タイムスタンプの3階層構造を抽象化し、シンプルなAPIを提供。

    構造:
        {
            "business_area": {
                "provider": {
                    "faq": timestamp,
                    "scenario": timestamp,
                    "last_updated": "2024-01-01T00:00:00"
                }
            }
        }

    Attributes:
        timestamp_file: タイムスタンプファイルのパス
        provider: 現在の埋め込みプロバイダー
    """

    def __init__(self, timestamp_file: str, provider: str):
        """DBVersionManagerを初期化

        Args:
            timestamp_file: タイムスタンプファイルのパス
            provider: 埋め込みプロバイダー名（vertex_ai / azure_openai）
        """
        self.timestamp_file = timestamp_file
        self.provider = provider
        self._cache: Dict[str, DBVersionInfo] = {}
        self._dirty = False

        self._load()

    def _load(self) -> None:
        """タイムスタンプファイルを読み込み"""
        if not os.path.exists(self.timestamp_file):
            logger.info("タイムスタンプファイルが存在しないため、新規作成します")
            return

        try:
            with open(self.timestamp_file, 'r', encoding='utf-8') as f:
                all_data = json.load(f)

            if not isinstance(all_data, dict):
                logger.warning(f"タイムスタンプファイルの形式が不正です: {type(all_data)}")
                return

            # 現在のプロバイダーのデータのみ抽出
            for business_area, providers in all_data.items():
                if not isinstance(providers, dict):
                    continue
                provider_data = providers.get(self.provider, {})
                if isinstance(provider_data, dict):
                    self._cache[business_area] = DBVersionInfo.from_dict(provider_data)

            logger.info(f"タイムスタンプ読み込み完了: {len(self._cache)}件 (プロバイダー: {self.provider})")

        except json.JSONDecodeError as e:
            logger.warning(f"タイムスタンプファイルのJSON解析エラー: {e}")
        except Exception as e:
            logger.warning(f"タイムスタンプ読み込みエラー: {e}")

    def save(self) -> None:
        """タイムスタンプファイルに保存"""
        if not self._dirty:
            return

        try:
            # 既存データを読み込み（他プロバイダーのデータを保持）
            existing_data = {}
            if os.path.exists(self.timestamp_file):
                try:
                    with open(self.timestamp_file, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                    if not isinstance(existing_data, dict):
                        existing_data = {}
                except Exception as e:
                    logger.warning(f"既存タイムスタンプの読み込みエラー: {e}")

            # 現在のプロバイダーのデータを更新
            for business_area, version_info in self._cache.items():
                if business_area not in existing_data:
                    existing_data[business_area] = {}
                existing_data[business_area][self.provider] = version_info.to_dict()

            # 書き込み
            with open(self.timestamp_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)

            self._dirty = False
            logger.info(f"タイムスタンプ保存完了: {len(self._cache)}件 (プロバイダー: {self.provider})")

        except Exception as e:
            logger.warning(f"タイムスタンプ保存エラー: {e}")

    def get_version(self, business_area: str) -> DBVersionInfo:
        """指定業務分野のバージョン情報を取得

        Args:
            business_area: 業務分野名

        Returns:
            DBVersionInfo: バージョン情報（存在しない場合は空のDBVersionInfo）
        """
        return self._cache.get(business_area, DBVersionInfo())

    def update_faq_mtime(self, business_area: str, mtime: float) -> None:
        """FAQファイルの最終更新時刻を更新

        Args:
            business_area: 業務分野名
            mtime: Unix timestamp
        """
        if business_area not in self._cache:
            self._cache[business_area] = DBVersionInfo()

        self._cache[business_area] = DBVersionInfo(
            faq_mtime=mtime,
            scenario_mtime=self._cache[business_area].scenario_mtime,
            last_updated=datetime.utcnow().isoformat(),
        )
        self._dirty = True

    def update_scenario_mtime(self, business_area: str, mtime: float) -> None:
        """シナリオファイルの最終更新時刻を更新

        Args:
            business_area: 業務分野名
            mtime: Unix timestamp
        """
        if business_area not in self._cache:
            self._cache[business_area] = DBVersionInfo()

        self._cache[business_area] = DBVersionInfo(
            faq_mtime=self._cache[business_area].faq_mtime,
            scenario_mtime=mtime,
            last_updated=datetime.utcnow().isoformat(),
        )
        self._dirty = True

    def needs_update(
        self,
        business_area: str,
        faq_path: Optional[str] = None,
        scenario_path: Optional[str] = None
    ) -> bool:
        """更新が必要かどうかを判定

        Args:
            business_area: 業務分野名
            faq_path: FAQファイルのパス
            scenario_path: シナリオファイルのパス

        Returns:
            bool: 更新が必要な場合True
        """
        version = self.get_version(business_area)

        # FAQファイルのチェック
        if faq_path and os.path.exists(faq_path):
            current_mtime = os.path.getmtime(faq_path)
            last_mtime = version.faq_mtime or 0.0
            if current_mtime > last_mtime:
                logger.info(f"FAQファイル更新検出: {business_area} (前回: {last_mtime}, 現在: {current_mtime})")
                return True

        # シナリオファイルのチェック
        if scenario_path and os.path.exists(scenario_path):
            current_mtime = os.path.getmtime(scenario_path)
            last_mtime = version.scenario_mtime or 0.0
            if current_mtime > last_mtime:
                logger.info(f"シナリオファイル更新検出: {business_area} (前回: {last_mtime}, 現在: {current_mtime})")
                return True

        return False

    def clear(self) -> None:
        """キャッシュをクリア"""
        self._cache.clear()
        self._dirty = True
        logger.info("タイムスタンプキャッシュをクリアしました")
