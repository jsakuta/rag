"""run_eval.py のキャッシュ機構テスト"""
import importlib
import sys
from pathlib import Path

import pytest
from unittest.mock import MagicMock, patch

# apps/revision-ops はハイフン付きディレクトリのため importlib で読み込む
PROJECT_ROOT = Path(__file__).parent.parent.parent
_mod_path = PROJECT_ROOT / "apps" / "revision-ops" / "run_eval.py"
_spec = importlib.util.spec_from_file_location("run_eval", _mod_path)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["run_eval"] = _mod
_spec.loader.exec_module(_mod)

RevisionEvaluator = _mod.RevisionEvaluator


class TestReferenceQueriesCache:
    """_get_reference_queries のメモ化テスト"""

    def _make_evaluator(self):
        """テスト用の RevisionEvaluator を最小構成で作成"""
        with patch.object(_mod, "create_llm") as mock_llm, \
             patch.object(_mod, "JudgmentSupport"):
            mock_llm.return_value = MagicMock()
            config = MagicMock()
            config.STOP_WORDS = ("の", "は", "が")
            config.POSITION_WEIGHT = 1.2
            evaluator = RevisionEvaluator(config, enable_llm_analysis=False)
            return evaluator

    def test_cache_hit_returns_same_result(self):
        """同じ (area, provider) の2回目呼び出しはキャッシュを返す"""
        evaluator = self._make_evaluator()
        evaluator._reference_queries_cache[("rev02_souzoku", "azure_openai")] = ["q1", "q2"]
        result = evaluator._get_reference_queries("rev02_souzoku", "azure_openai")
        assert result == ["q1", "q2"]

    def test_cache_miss_calls_chromadb(self):
        """キャッシュにない場合はChromaDBを呼ぶ"""
        evaluator = self._make_evaluator()
        assert ("rev02_souzoku", "azure_openai") not in evaluator._reference_queries_cache


class TestOrchestratorCache:
    """_create_orchestrator のメモ化テスト"""

    def _make_evaluator(self):
        """テスト用の RevisionEvaluator を最小構成で作成"""
        with patch.object(_mod, "create_llm") as mock_llm, \
             patch.object(_mod, "JudgmentSupport"):
            mock_llm.return_value = MagicMock()
            config = MagicMock()
            config.STOP_WORDS = ("の", "は", "が")
            config.POSITION_WEIGHT = 1.2
            evaluator = RevisionEvaluator(config, enable_llm_analysis=False)
            return evaluator

    def test_orchestrator_cache_hit(self):
        """同じ (area, provider, weight) の2回目はキャッシュを返す"""
        evaluator = self._make_evaluator()
        mock_orch = MagicMock()
        evaluator._orchestrator_cache[("rev02_souzoku", "azure_openai", 0.9)] = mock_orch
        result = evaluator._create_orchestrator("azure_openai", "rev02_souzoku", ["q1"], 0.9)
        assert result is mock_orch


class TestLlmQueryCache:
    """_llm_query_cache のメモ化テスト"""

    def _make_evaluator(self):
        """テスト用の RevisionEvaluator を最小構成で作成"""
        with patch.object(_mod, "create_llm") as mock_llm, \
             patch.object(_mod, "JudgmentSupport"):
            mock_llm.return_value = MagicMock()
            config = MagicMock()
            config.STOP_WORDS = ("の", "は", "が")
            config.POSITION_WEIGHT = 1.2
            evaluator = RevisionEvaluator(config, enable_llm_analysis=False)
            return evaluator

    def test_llm_query_cache_hit(self):
        """同じ revision_content の2回目はキャッシュを返す"""
        evaluator = self._make_evaluator()
        evaluator._llm_query_cache["テスト改定内容"] = "拡張クエリ"
        assert evaluator._llm_query_cache["テスト改定内容"] == "拡張クエリ"
