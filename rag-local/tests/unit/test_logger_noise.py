import logging
from src.utils.logger import suppress_noise


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
