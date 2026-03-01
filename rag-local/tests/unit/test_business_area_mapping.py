# --- tests/unit/test_business_area_mapping.py ---
"""業務分野マッピングの統一テスト

YAMLマッピングと_translate_business_areaの一致を検証する。
"""

import pytest
import os

from src.utils.business_area_translator import BusinessAreaTranslator


class TestYAMLMappingCompleteness:
    """YAMLマッピングの完全性テスト"""

    @pytest.fixture
    def translator(self):
        return BusinessAreaTranslator()

    def test_smile_mapped(self, translator):
        assert translator.translate("スマイル") == "smile"

    def test_naibujimu_mapped(self, translator):
        assert translator.translate("内部事務") == "naibujimu"

    def test_souzoku_mapped(self, translator):
        assert translator.translate("相続") == "souzoku"

    def test_torikaku_mapped(self, translator):
        assert translator.translate("取引時確認") == "torikaku"

    def test_smile_tablet_mapped(self, translator):
        assert translator.translate("スマイルタブレット") == "smile_tablet"

    def test_existing_mappings_unchanged(self, translator):
        """既存マッピングが壊れていないことを確認"""
        assert translator.translate("総則") == "general"
        assert translator.translate("預金") == "deposit"
        assert translator.translate("融資") == "loan"
        assert translator.translate("外貨") == "foreign_currency"
        assert translator.translate("投信") == "investment_trust"

    def test_revision_mappings_unchanged(self, translator):
        """rev系マッピングが壊れていないことを確認"""
        assert translator.translate("rev01_smile") == "rev01_smile"
        assert translator.translate("rev02_souzoku") == "rev02_souzoku"
        assert translator.translate("rev03_naibujimu") == "rev03_naibujimu"


class TestTranslateDelegatesToTranslator:
    """_translate_business_area が BusinessAreaTranslator と同じ結果を返すことを検証"""

    @pytest.fixture
    def translator(self):
        return BusinessAreaTranslator()

    @pytest.fixture
    def db_manager(self, mock_config):
        from src.utils.dynamic_db_manager import DynamicDBManager
        return DynamicDBManager(mock_config)

    @pytest.mark.parametrize("input_name", [
        "総則", "預金", "融資",
        "スマイル", "スマイルタブレット",
        "内部事務", "相続", "取引時確認",
        "外貨", "投信",
        "rev01_smile", "rev02_souzoku",
    ])
    def test_results_match(self, db_manager, translator, input_name):
        """全入力パターンで _translate_business_area と translator.translate が一致"""
        assert db_manager._translate_business_area(input_name) == translator.translate(input_name)


class TestTimestampKeyMigration:
    """タイムスタンプキーの移行テスト"""

    @pytest.fixture
    def db_manager(self, mock_config):
        from src.utils.dynamic_db_manager import DynamicDBManager
        return DynamicDBManager(mock_config)

    def test_normalize_japanese_to_english(self, db_manager):
        """日本語キーが英語に正規化されること"""
        assert db_manager._normalize_timestamp_key("スマイル") == "smile"
        assert db_manager._normalize_timestamp_key("総則") == "general"
        assert db_manager._normalize_timestamp_key("預金") == "deposit"
        assert db_manager._normalize_timestamp_key("内部事務") == "naibujimu"
        assert db_manager._normalize_timestamp_key("相続") == "souzoku"
        assert db_manager._normalize_timestamp_key("取引時確認") == "torikaku"

    def test_normalize_rev_underscore(self, db_manager):
        """rev系のアンダースコアなしキーが補完されること"""
        assert db_manager._normalize_timestamp_key("rev01smile") == "rev01_smile"
        assert db_manager._normalize_timestamp_key("rev02souzoku") == "rev02_souzoku"
        assert db_manager._normalize_timestamp_key("rev03naibujimu") == "rev03_naibujimu"
        assert db_manager._normalize_timestamp_key("rev03smile") == "rev03_smile"
        assert db_manager._normalize_timestamp_key("rev05smile") == "rev05_smile"
        assert db_manager._normalize_timestamp_key("rev06smile") == "rev06_smile"

    def test_normalize_already_correct_keys(self, db_manager):
        """既に正しいキーは変更されないこと"""
        assert db_manager._normalize_timestamp_key("smile") == "smile"
        assert db_manager._normalize_timestamp_key("general") == "general"
        assert db_manager._normalize_timestamp_key("deposit") == "deposit"
        assert db_manager._normalize_timestamp_key("rev01_smile") == "rev01_smile"
        assert db_manager._normalize_timestamp_key("rev03_torikaku") == "rev03_torikaku"

    def test_migrate_merges_duplicates(self, db_manager):
        """移行時に重複キーは新しいタイムスタンプを優先すること"""
        db_manager._last_faq_mtime = {
            "スマイル": 100.0,  # 古い日本語キー
            "smile": 200.0,    # 新しい英語キー
        }
        db_manager._migrate_timestamp_keys()
        assert db_manager._last_faq_mtime == {"smile": 200.0}

    def test_migrate_scenario_rev_keys(self, db_manager):
        """rev系シナリオキーが正しく移行されること"""
        db_manager._last_scenario_mtime = {
            "rev03smile": 100.0,
            "rev03_smile": 200.0,  # 既に新キーが存在
        }
        db_manager._migrate_timestamp_keys()
        assert db_manager._last_scenario_mtime == {"rev03_smile": 200.0}


class TestAnalyzeReferenceFilesKeyConsistency:
    """analyze_reference_files のキーが業務分野セレクタと一致することを検証"""

    def test_smile_faq_keyed_by_smile(self, mock_config):
        """スマイル履歴データが 'smile' キーで返されること"""
        from src.utils.dynamic_db_manager import DynamicDBManager
        with DynamicDBManager(mock_config) as db_manager:
            business_areas = db_manager.analyze_reference_files()
            # スマイル_履歴データ_XXXXXXXX.xlsx が存在すれば 'smile' キーで分類される
            if os.path.exists(db_manager.reference_faq_path) and any(
                "スマイル" in f for f in os.listdir(db_manager.reference_faq_path)
            ):
                assert "smile" in business_areas, (
                    f"'smile' キーが見つかりません。検出キー: {list(business_areas.keys())}"
                )
                assert len(business_areas["smile"]["faq"]) > 0

    def test_no_default_key_for_known_areas(self, mock_config):
        """既知業務分野が 'default' キーにフォールバックしないこと"""
        from src.utils.dynamic_db_manager import DynamicDBManager
        with DynamicDBManager(mock_config) as db_manager:
            business_areas = db_manager.analyze_reference_files()
            assert "default" not in business_areas, (
                f"'default' キーが存在します。マッピング漏れの可能性: {business_areas.get('default')}"
            )


class TestGetAllBusinessAreasRevisionFilter:
    """get_all_business_areas の改定別コレクション除外テスト"""

    @pytest.fixture
    def db_manager_with_dirs(self, mock_config, tmp_path):
        """通常業務 + 改定別のDBディレクトリを持つDynamicDBManager"""
        from src.utils.dynamic_db_manager import DynamicDBManager
        db_manager = DynamicDBManager(mock_config)
        db_manager.base_db_path = str(tmp_path)

        # 通常業務: smile/azure_openai/chroma.sqlite3
        for area in ["smile", "deposit", "general"]:
            db_dir = tmp_path / area / "azure_openai"
            db_dir.mkdir(parents=True)
            (db_dir / "chroma.sqlite3").touch()

        # 改定別: rev01_smile/azure_openai/chroma.sqlite3
        for area in ["rev01_smile", "rev02_souzoku", "rev03_naibujimu"]:
            db_dir = tmp_path / area / "azure_openai"
            db_dir.mkdir(parents=True)
            (db_dir / "chroma.sqlite3").touch()

        return db_manager

    def test_default_includes_revisions(self, db_manager_with_dirs):
        """デフォルトでは改定別コレクションを含む"""
        areas = db_manager_with_dirs.get_all_business_areas()
        assert "rev01_smile" in areas
        assert "smile" in areas

    def test_exclude_revisions(self, db_manager_with_dirs):
        """include_revisions=False で改定別コレクションを除外"""
        areas = db_manager_with_dirs.get_all_business_areas(include_revisions=False)
        assert "smile" in areas
        assert "deposit" in areas
        assert "general" in areas
        assert not any(a.startswith("rev") for a in areas)

    def test_include_revisions_explicit(self, db_manager_with_dirs):
        """include_revisions=True で改定別コレクションを含む"""
        areas = db_manager_with_dirs.get_all_business_areas(include_revisions=True)
        assert "rev01_smile" in areas
        assert "rev02_souzoku" in areas
        assert "smile" in areas


class TestAnalyzeReferenceFilesRevisionFilter:
    """analyze_reference_files の改定別エリア除外テスト"""

    @pytest.fixture
    def db_manager_with_files(self, mock_config, tmp_path):
        """通常業務 + 改定別の参照ファイルを持つDynamicDBManager"""
        from src.utils.dynamic_db_manager import DynamicDBManager
        db_manager = DynamicDBManager(mock_config)

        # シナリオ latest ディレクトリ（通常業務ファイル）
        scenario_latest_dir = tmp_path / "scenarios" / "latest"
        scenario_latest_dir.mkdir(parents=True)
        db_manager.reference_scenario_path = str(scenario_latest_dir)
        (scenario_latest_dir / "スマイル_シナリオデータ_20260224.xlsx").touch()

        # シナリオ revisions ディレクトリ（改定別ファイル）
        scenario_rev_dir = tmp_path / "scenarios" / "revisions"
        scenario_rev_dir.mkdir(parents=True)
        db_manager.reference_revision_scenario_path = str(scenario_rev_dir)
        (scenario_rev_dir / "rev01_smile_シナリオデータ_20260203.xlsx").touch()
        (scenario_rev_dir / "rev02_souzoku_シナリオデータ_20260203.xlsx").touch()

        # FAQディレクトリ（通常業務ファイル）
        faq_dir = tmp_path / "faq"
        faq_dir.mkdir()
        db_manager.reference_faq_path = str(faq_dir)
        (faq_dir / "預金_履歴データ_20250830.xlsx").touch()
        (faq_dir / "スマイル_履歴データ_20250205.xlsx").touch()

        return db_manager

    def test_default_includes_revisions(self, db_manager_with_files):
        """デフォルトでは改定別エリアを含む"""
        areas = db_manager_with_files.analyze_reference_files()
        assert "rev01_smile" in areas
        assert "rev02_souzoku" in areas
        assert "deposit" in areas
        assert "smile" in areas

    def test_exclude_revisions(self, db_manager_with_files):
        """include_revisions=False で改定別エリアを除外"""
        areas = db_manager_with_files.analyze_reference_files(include_revisions=False)
        assert "deposit" in areas
        assert "smile" in areas
        assert not any(k.startswith("rev") for k in areas)

    def test_include_revisions_explicit(self, db_manager_with_files):
        """include_revisions=True で改定別エリアを含む"""
        areas = db_manager_with_files.analyze_reference_files(include_revisions=True)
        assert "rev01_smile" in areas
        assert "rev02_souzoku" in areas
        assert "deposit" in areas
        assert "smile" in areas


class TestGetScenarioBasePath:
    """_get_scenario_base_path が業務分野に応じた正しいパスを返すことを検証"""

    @pytest.fixture
    def db_manager(self, mock_config):
        from src.utils.dynamic_db_manager import DynamicDBManager
        return DynamicDBManager(mock_config)

    def test_regular_area_returns_latest(self, db_manager):
        """通常業務は scenarios/latest/ パスを返す"""
        path = db_manager._get_scenario_base_path("smile")
        assert path == db_manager.reference_scenario_path
        assert "latest" in path

    def test_revision_area_returns_revisions(self, db_manager):
        """改定別は scenarios/revisions/ パスを返す"""
        path = db_manager._get_scenario_base_path("rev01_smile")
        assert path == db_manager.reference_revision_scenario_path
        assert "revisions" in path

    def test_naibujimu_returns_latest(self, db_manager):
        path = db_manager._get_scenario_base_path("naibujimu")
        assert "latest" in path

    def test_rev03_torikaku_returns_revisions(self, db_manager):
        path = db_manager._get_scenario_base_path("rev03_torikaku")
        assert "revisions" in path


class TestGetDisplayName:
    def test_rev_prefix(self):
        from src.utils.business_area_translator import get_display_name
        assert get_display_name("rev03_naibujimu") == "内部事務"

    def test_plain_name(self):
        from src.utils.business_area_translator import get_display_name
        assert get_display_name("smile") == "スマイル"

    def test_unknown_returns_as_is(self):
        from src.utils.business_area_translator import get_display_name
        assert get_display_name("unknown_area") == "unknown_area"

    def test_souzoku(self):
        from src.utils.business_area_translator import get_display_name
        assert get_display_name("rev02_souzoku") == "相続"
