"""print_startup_summary / print_query_panel のユニットテスト

Rich Console は force_terminal=True で capsys を迂回するため、
テストではモジュールの RICH_AVAILABLE を一時的に False に差し替え、
プレーンテキストフォールバックパスを検証する。
"""

import src.utils.logger as logger_mod
from src.utils.logger import print_startup_summary, print_query_panel


def _force_plain(monkeypatch):
    """Rich を無効化してプレーンテキスト出力にする"""
    monkeypatch.setattr(logger_mod, "RICH_AVAILABLE", False)
    monkeypatch.setattr(logger_mod, "_console", None)


def test_print_startup_summary_all_ok(monkeypatch, capsys):
    """全チェックOKの場合、アプリ名とチェック項目が出ること"""
    _force_plain(monkeypatch)
    checks = [
        ("DB接続", True, "naibujimu: 11,439件"),
        ("LLM接続", True, "Gemini 2.5 Flash Lite"),
    ]
    print_startup_summary("テストAI v1.0", checks)
    captured = capsys.readouterr()
    assert "テストAI" in captured.out
    assert "DB接続" in captured.out


def test_print_startup_summary_with_failure(monkeypatch, capsys):
    """チェック失敗がある場合、NGアイコンが表示されること"""
    _force_plain(monkeypatch)
    checks = [
        ("DB接続", True, "OK"),
        ("LLM接続", False, "接続失敗"),
    ]
    print_startup_summary("テストAI v1.0", checks)
    captured = capsys.readouterr()
    assert "接続失敗" in captured.out


def test_print_startup_summary_with_skip(monkeypatch, capsys):
    """ok=Noneの場合、スキップ表示(--アイコン)になること"""
    _force_plain(monkeypatch)
    checks = [
        ("DB接続", True, "OK"),
        ("LLM判定", None, "無効 (スキップ) (gemini-2.5-flash-lite)"),
    ]
    print_startup_summary("テストAI v1.0", checks)
    captured = capsys.readouterr()
    assert "[--]" in captured.out
    assert "無効 (スキップ)" in captured.out


def test_print_query_panel_basic(monkeypatch, capsys):
    """検索パネルにクエリとメタデータが表示されること"""
    _force_plain(monkeypatch)
    print_query_panel(
        query_number=1,
        query_text="保険証が廃止された場合の手続き",
        metadata={"改定": "③", "タイプ": "hybrid"},
        results={"Azure": 45, "VertexAI": 38},
        elapsed=1.2,
    )
    captured = capsys.readouterr()
    assert "保険証" in captured.out
    assert "45" in captured.out
    assert "1.2" in captured.out


def test_print_query_panel_single_provider(monkeypatch, capsys):
    """プロバイダーが1つの場合でも表示されること"""
    _force_plain(monkeypatch)
    print_query_panel(
        query_number=1,
        query_text="テストクエリ",
        metadata={"業務": "内部事務"},
        results={"結果": 12},
        elapsed=0.8,
    )
    captured = capsys.readouterr()
    assert "テストクエリ" in captured.out
    assert "12" in captured.out
