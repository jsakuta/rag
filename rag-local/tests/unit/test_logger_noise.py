import logging

import pytest

from src.utils.logger import suppress_noise

# suppress_noise() が変更するロガー名のリスト
_NOISY_LOGGERS = [
    "chromadb", "httpx", "urllib3", "streamlit",
    "google.auth", "google.api_core",
    "azure.core", "azure.identity",
    "streamlit.runtime.scriptrunner_utils",
]


@pytest.fixture(autouse=True)
def _reset_logger_levels():
    """テスト後にロガーレベルをリセットし、グローバル状態の汚染を防ぐ"""
    original_levels = {name: logging.getLogger(name).level for name in _NOISY_LOGGERS}
    yield
    for name, level in original_levels.items():
        logging.getLogger(name).setLevel(level)


def test_suppress_noise_sets_third_party_to_warning():
    suppress_noise()
    assert logging.getLogger("chromadb").level == logging.WARNING
    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("urllib3").level == logging.WARNING
    assert logging.getLogger("streamlit").level == logging.WARNING


def test_suppress_noise_sets_google_azure_to_warning():
    suppress_noise()
    assert logging.getLogger("google.auth").level == logging.WARNING
    assert logging.getLogger("google.api_core").level == logging.WARNING
    assert logging.getLogger("azure.core").level == logging.WARNING
    assert logging.getLogger("azure.identity").level == logging.WARNING


def test_suppress_noise_filters_scriptruncontext():
    """ScriptRunContext 警告がフィルタされること"""
    suppress_noise()
    streamlit_logger = logging.getLogger("streamlit.runtime.scriptrunner_utils")
    assert streamlit_logger.level >= logging.ERROR
