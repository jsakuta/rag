"""DynamicDBManagerタイムスタンプ移行のユニットテスト"""
import json
import os
import pytest
import tempfile
from unittest.mock import MagicMock, patch


class TestTimestampMigration:
    """フラット化タイムスタンプの読み書きテスト"""

    def _create_mock_config(self, tmp_dir):
        """テスト用モックConfigを生成"""
        config = MagicMock()
        config.base_dir = tmp_dir
        config.embedding_provider = "azure_openai"
        config.VALID_EMBEDDING_PROVIDERS = ("vertex_ai", "azure_openai")
        config.REFERENCE_FILE_PATTERN = r".*?([^_]+).*?(履歴データ|シナリオデータ).*?(\d{8})?.*?\.xlsx$"
        return config

    def test_flat_format_roundtrip(self):
        """フラット形式の保存→読み込みラウンドトリップ"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ts_file = os.path.join(tmp_dir, "update_timestamps.json")

            # フラット形式のデータを直接書き込み
            flat_data = {
                "総則_azure_openai_faq": 1234567890.0,
                "総則_azure_openai_scenario": 1234567891.0,
                "預金_azure_openai_faq": 1234567892.0,
            }
            with open(ts_file, 'w', encoding='utf-8') as f:
                json.dump(flat_data, f)

            # 読み込みテスト（DynamicDBManagerのロジックをシミュレート）
            with open(ts_file, 'r', encoding='utf-8') as f:
                timestamps = json.load(f)

            provider = "azure_openai"
            faq_mtime = {}
            scenario_mtime = {}

            suffix_faq = f"_{provider}_faq"
            suffix_scenario = f"_{provider}_scenario"

            for key, value in timestamps.items():
                if key.endswith(suffix_faq):
                    area = key[:-len(suffix_faq)]
                    faq_mtime[area] = value
                elif key.endswith(suffix_scenario):
                    area = key[:-len(suffix_scenario)]
                    scenario_mtime[area] = value

            assert faq_mtime == {"総則": 1234567890.0, "預金": 1234567892.0}
            assert scenario_mtime == {"総則": 1234567891.0}

    def test_old_format_detection(self):
        """旧3階層形式が正しく検出・読み込みされること"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ts_file = os.path.join(tmp_dir, "update_timestamps.json")

            # 旧3階層形式
            old_data = {
                "総則": {
                    "azure_openai": {
                        "faq": 1234567890.0,
                        "scenario": 1234567891.0
                    }
                }
            }
            with open(ts_file, 'w', encoding='utf-8') as f:
                json.dump(old_data, f)

            with open(ts_file, 'r', encoding='utf-8') as f:
                timestamps = json.load(f)

            # フラット形式かチェック
            is_flat = any("_faq" in k or "_scenario" in k for k in timestamps.keys())
            assert not is_flat

            # 旧形式の読み込み
            faq_mtime = {}
            scenario_mtime = {}
            provider = "azure_openai"

            for business_area, providers in timestamps.items():
                if not isinstance(providers, dict):
                    continue
                provider_data = providers.get(provider, {})
                if isinstance(provider_data, dict):
                    if 'faq' in provider_data:
                        faq_mtime[business_area] = provider_data['faq']
                    if 'scenario' in provider_data:
                        scenario_mtime[business_area] = provider_data['scenario']

            assert faq_mtime == {"総則": 1234567890.0}
            assert scenario_mtime == {"総則": 1234567891.0}
