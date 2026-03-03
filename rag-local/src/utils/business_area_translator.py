# --- src/utils/business_area_translator.py ---
"""業務分野名変換ユーティリティ

YAMLファイルから設定を読み込み、業務分野名を変換。
"""

import os
import re
from typing import Dict, Optional

import yaml

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class BusinessAreaTranslator:
    """業務分野名変換クラス

    日本語の業務分野名をChromaDB互換の英語名に変換する。

    Attributes:
        mappings: 基本マッピング辞書
        revision_mappings: 事務改定用マッピング辞書
        constraints: ChromaDBコレクション名の制約
        defaults: デフォルト値
    """

    def __init__(self, config_path: Optional[str] = None):
        """BusinessAreaTranslatorを初期化

        Args:
            config_path: 設定ファイルのパス（省略時はデフォルトパス）
        """
        self.mappings: Dict[str, str] = {}
        self.revision_mappings: Dict[str, str] = {}
        self.constraints: Dict[str, object] = {
            'min_length': 3,
            'max_length': 512,
        }
        self.defaults: Dict[str, str] = {
            'collection_name': 'default',
            'fallback_prefix': 'c',
            'fallback_suffix': 'c',
        }

        if config_path is None:
            # デフォルトパスを使用
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            config_path = os.path.join(base_dir, 'config', 'business_areas.yaml')

        self._load_config(config_path)

    def _load_config(self, config_path: str) -> None:
        """設定ファイルを読み込み

        Args:
            config_path: 設定ファイルのパス
        """
        if not os.path.exists(config_path):
            logger.warning(f"業務分野設定ファイルが見つかりません: {config_path}")
            logger.info("デフォルトのハードコードされたマッピングを使用します")
            self._use_default_mappings()
            return

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            if not isinstance(config, dict):
                logger.warning("設定ファイルの形式が不正です")
                self._use_default_mappings()
                return

            # マッピングの読み込み
            self.mappings = config.get('mappings', {})
            self.revision_mappings = config.get('revision_mappings', {})
            self.constraints = config.get('collection_constraints', self.constraints)
            self.defaults = config.get('defaults', self.defaults)

            logger.debug(f"業務分野設定を読み込みました: {len(self.mappings)}件 + 事務改定{len(self.revision_mappings)}件")

        except yaml.YAMLError as e:
            logger.warning(f"YAML解析エラー: {e}")
            self._use_default_mappings()
        except Exception as e:
            logger.warning(f"設定ファイル読み込みエラー: {e}")
            self._use_default_mappings()

    def _use_default_mappings(self) -> None:
        """デフォルトのハードコードされたマッピングを使用（YAML読み込み失敗時のフォールバック）"""
        self.mappings = {
            "スマイル": "smile",
            "スマイルタブレット": "smile_tablet",
            "内部事務": "naibujimu",
            "相続": "souzoku",
            "取引時確認": "torikaku",
        }
        logger.info("デフォルトマッピングを使用します")

    def translate(self, business_area: str) -> str:
        """業務分野名を英語に変換

        Args:
            business_area: 日本語の業務分野名

        Returns:
            str: ChromaDB互換の英語名
        """
        max_length = self.constraints.get('max_length', 512)
        min_length = self.constraints.get('min_length', 3)

        # 長さ制限
        if len(business_area) > max_length:
            logger.warning(f"業務分野名が長すぎます: {len(business_area)}文字、切り詰めます")
            business_area = business_area[:max_length]

        # 事務改定マッピングを優先チェック（revXXで始まる場合）
        if business_area.startswith('rev') and business_area in self.revision_mappings:
            return self.revision_mappings[business_area]

        # 完全一致を優先
        if business_area in self.mappings:
            return self.mappings[business_area]

        # 部分一致で検索
        for japanese, english in self.mappings.items():
            if japanese in business_area:
                return english

        # マッチしない場合: 英数字のみに変換
        sanitized = re.sub(r'[^a-zA-Z0-9._-]', '_', business_area)
        sanitized = re.sub(r'_+', '_', sanitized).strip('_')

        # 先頭・末尾が英数字であることを保証
        prefix = self.defaults.get('fallback_prefix', 'c')
        suffix = self.defaults.get('fallback_suffix', 'c')

        if sanitized and not sanitized[0].isalnum():
            sanitized = prefix + sanitized
        if sanitized and not sanitized[-1].isalnum():
            sanitized = sanitized + suffix

        # 最小長チェック
        if len(sanitized) < min_length:
            sanitized = self.defaults.get('collection_name', 'default_collection')

        return sanitized if sanitized else self.defaults.get('collection_name', 'default')



# エリア名の日本語表示用マッピング
_AREA_DISPLAY_NAMES = {
    "naibujimu": "内部事務",
    "smile": "スマイル",
    "souzoku": "相続",
    "torikaku": "取引時確認",
}


def get_display_name(area: str) -> str:
    """内部エリア名を日本語表示名に変換する。

    例: "rev03_naibujimu" -> "内部事務", "naibujimu" -> "内部事務"
    マッチしない場合はそのまま返す。
    """
    for key, name in _AREA_DISPLAY_NAMES.items():
        if key in area:
            return name
    return area


def resolve_bot_name(area: str, area_to_bot: dict) -> str:
    """area名からbot名をsubstring-matchで解決

    Args:
        area: エリア名（例: "rev02_souzoku", "smile"）
        area_to_bot: エリア→ボット名マッピング dict

    Returns:
        str: ボット名（マッチなしの場合 "unknown-bot"）
    """
    area_lower = area.lower()
    for keyword, bot_name in area_to_bot.items():
        if keyword in area_lower:
            return bot_name
    return "unknown-bot"
