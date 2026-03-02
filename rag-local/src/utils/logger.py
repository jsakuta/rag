# --- utils/logger.py ---
"""
ターミナルログ出力モジュール

デザインコンセプト: インダストリアル・ダッシュボード
- 情報密度と視認性のバランス
- 改定ごとの明確な視覚的区切り
- プロバイダー別の色分け（Azure=青、VertexAI=緑）
"""
import logging
import os
import sys
from typing import Optional, List

# richがインストールされているか確認
try:
    from rich.logging import RichHandler
    from rich.console import Console, Group
    from rich.theme import Theme
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.table import Table
    from rich.text import Text
    from rich.box import ROUNDED
    from rich import box
    RICH_AVAILABLE = True

    # カスタムテーマ（インダストリアル・ダッシュボード）
    CUSTOM_THEME = Theme({
        "info": "#64B5F6",           # ライトブルー
        "warning": "#FFB74D",        # オレンジ
        "error": "#EF5350 bold",     # レッド
        "critical": "#EF5350 bold reverse",
        "debug": "#9E9E9E",          # グレー
        "success": "#81C784",        # グリーン
        "highlight": "#BA68C8 bold", # パープル
        "azure": "#2196F3",          # Azure Blue
        "vertex": "#4CAF50",         # Google Green
        "revision": "#FF9800 bold",  # オレンジ（改定番号）
        "muted": "#757575",          # ミュート
        "accent": "#00BCD4",         # シアン
    })
except ImportError:
    RICH_AVAILABLE = False
    CUSTOM_THEME = None

# グローバルコンソール（rich用）
_console: Optional["Console"] = None


def get_console() -> Optional["Console"]:
    """共有Consoleインスタンスを取得"""
    global _console
    if _console is None and RICH_AVAILABLE:
        _console = Console(theme=CUSTOM_THEME, force_terminal=True)
    return _console


def _shorten_module_name(name: str) -> str:
    """モジュール名を短縮 (src.utils.vector_db -> vector_db)"""
    parts = name.split(".")
    if len(parts) > 1:
        return parts[-1]
    return name


class ShortModuleFormatter(logging.Formatter):
    """モジュール名を短縮するカスタムフォーマッター"""

    def format(self, record):
        record.short_name = _shorten_module_name(record.name)
        return super().format(record)


def setup_logger(name: str) -> logging.Logger:
    """ロガーの設定

    rich がインストールされている場合は色付きログを出力。
    ファイルには詳細なフォーマットで保存。
    """
    logger = logging.getLogger(name)

    # セキュリティ: ホワイトリスト方式でログレベルを検証
    VALID_LOG_LEVELS = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()

    invalid_level = None
    if log_level not in VALID_LOG_LEVELS:
        invalid_level = log_level
        log_level = 'INFO'

    level = getattr(logging, log_level)
    logger.setLevel(level)

    # ハンドラの重複追加を防止
    if logger.handlers:
        return logger

    # ログディレクトリ作成
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'app.log')

    # ファイルハンドラ（詳細フォーマット）
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    )
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # コンソールハンドラ（rich使用時は色付き、それ以外は短縮フォーマット）
    if RICH_AVAILABLE:
        # RichHandlerで色付きログ
        console = get_console()
        rich_handler = RichHandler(
            console=console,
            show_time=True,
            show_level=True,
            show_path=False,  # パスは非表示（短縮モジュール名を使うため）
            rich_tracebacks=True,
            tracebacks_show_locals=False,
            markup=True,
            log_time_format="[%H:%M:%S]",
        )
        # モジュール名を短縮して表示
        rich_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(rich_handler)
    else:
        # 標準フォーマット（短縮版）
        stream_formatter = ShortModuleFormatter(
            '%(asctime)s [%(levelname).1s] %(short_name)s: %(message)s',
            datefmt='%H:%M:%S'
        )
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)

    # 無効なログレベルの警告
    if invalid_level:
        logger.warning(
            f"Invalid LOG_LEVEL '{invalid_level}' specified. "
            f"Using default 'INFO'. Valid: {VALID_LOG_LEVELS}"
        )

    return logger


def print_section(title: str, char: str = "=", width: int = 60):
    """セクション区切りを出力（改定番号用の強調版）"""
    if RICH_AVAILABLE:
        console = get_console()
        # 改定番号を検出してスタイル適用
        if title.startswith("改定"):
            console.print()
            console.rule(
                f"[bold #FF9800]▶ {title}[/bold #FF9800]",
                style="#FF9800",
                characters="━"
            )
        else:
            console.rule(f"[bold #00BCD4]{title}[/bold #00BCD4]", style="#546E7A")
    else:
        line = char * width
        print(f"\n{line}")
        print(f" {title}")
        print(line)


def print_table(title: str, data: list, columns: list):
    """テーブル形式でデータを出力"""
    if RICH_AVAILABLE:
        from rich.table import Table
        console = get_console()
        table = Table(
            title=f"[bold #00BCD4]{title}[/bold #00BCD4]",
            show_header=True,
            header_style="bold #90A4AE",
            border_style="#546E7A",
            box=box.ROUNDED,
            padding=(0, 1),
            collapse_padding=True,
        )

        # 列スタイルを設定
        for i, col in enumerate(columns):
            if col == "番号":
                table.add_column(col, style="#FF9800 bold", justify="center", width=6)
            elif col == "正解数":
                table.add_column(col, style="#81C784", justify="right", width=8)
            elif "Azure" in col:
                table.add_column(col, style="#64B5F6", justify="right")
            elif "VertexAI" in col or "Vertex" in col:
                table.add_column(col, style="#81C784", justify="right")
            else:
                table.add_column(col, style="#B0BEC5")

        for row in data:
            table.add_row(*[str(cell) for cell in row])
        console.print(table)
    else:
        print(f"\n{title}")
        print("-" * 40)
        for row in data:
            print("  ".join(str(cell) for cell in row))


def print_status(message: str, status: str = "info"):
    """ステータスメッセージを出力

    status: "info", "success", "warning", "error"
    """
    if RICH_AVAILABLE:
        console = get_console()
        icons = {
            "info": "[#64B5F6]│[/#64B5F6]",
            "success": "[#81C784]✓[/#81C784]",
            "warning": "[#FFB74D]⚠[/#FFB74D]",
            "error": "[#EF5350]✗[/#EF5350]",
        }
        icon = icons.get(status, icons["info"])
        console.print(f"  {icon} {message}")
    else:
        icons = {
            "info": "[i]",
            "success": "[+]",
            "warning": "[!]",
            "error": "[x]",
        }
        icon = icons.get(status, icons["info"])
        print(f"{icon} {message}")


def print_revision_header(
    revision: str,
    content: str,
    correct_count: int,
    current: int,
    total: int
):
    """改定評価のヘッダーを表示（視認性重視）"""
    if RICH_AVAILABLE:
        console = get_console()

        # プログレスバー風の表示
        progress_text = f"[#757575]{current}/{total}[/#757575]"

        # 改定内容パネル
        content_preview = content[:100] + "..." if len(content) > 100 else content

        header_text = Text()
        header_text.append(f"  {revision} ", style="bold #FF9800")
        header_text.append(f"  正解ID: {correct_count}件", style="#81C784")
        header_text.append(f"  {progress_text}", style="#757575")

        console.print()
        console.print(Panel(
            Group(
                header_text,
                Text(f"\n  {content_preview}", style="#B0BEC5"),
            ),
            border_style="#FF9800",
            box=box.ROUNDED,
            padding=(0, 1),
        ))
    else:
        print(f"\n{'='*60}")
        print(f" [{current}/{total}] {revision}")
        print(f" 正解ID: {correct_count}件")
        print(f" {content[:80]}...")
        print("="*60)


def print_search_result(
    provider: str,
    result_count: int,
    areas: List[str],
    found_correct: int = 0,
    total_correct: int = 0
):
    """検索結果をコンパクトに表示"""
    if RICH_AVAILABLE:
        console = get_console()

        if provider.lower() in ("azure", "azure_openai"):
            color = "#2196F3"
            icon = "◆"
            name = "Azure"
        elif provider.lower() in ("keyword", "keyword_filter"):
            color = "#FF9800"
            icon = "◈"
            name = "Keyword"
        else:
            color = "#4CAF50"
            icon = "◇"
            name = "VertexAI"

        # 正解発見率
        rate = f"{found_correct}/{total_correct}" if total_correct > 0 else "-"
        rate_pct = f"({found_correct/total_correct*100:.0f}%)" if total_correct > 0 and found_correct > 0 else ""

        from src.utils.business_area_translator import get_display_name
        areas_display = [get_display_name(a) for a in areas] if areas else ["-"]
        areas_str = ", ".join(areas_display)

        result_text = Text()
        result_text.append(f"  {icon} ", style=f"bold {color}")
        result_text.append(f"{name:10}", style=f"bold {color}")
        result_text.append(f" {result_count:>4}件", style="#B0BEC5")
        result_text.append(f"  正解: {rate} {rate_pct}", style="#81C784" if found_correct > 0 else "#757575")
        result_text.append(f"  [{areas_str}]", style="#757575")

        console.print(result_text)
    else:
        print(f"  {provider}: {result_count}件 ({', '.join(areas)})")


def print_completion(output_file: str, elapsed_time: float = 0):
    """完了メッセージを表示"""
    if RICH_AVAILABLE:
        console = get_console()
        console.print()
        console.print(Panel(
            f"[bold #81C784]✓ 評価完了[/bold #81C784]\n\n"
            f"[#B0BEC5]出力ファイル:[/#B0BEC5] [bold]{output_file}[/bold]"
            + (f"\n[#757575]処理時間: {elapsed_time:.1f}秒[/#757575]" if elapsed_time > 0 else ""),
            border_style="#81C784",
            box=box.ROUNDED,
            padding=(1, 2),
        ))
    else:
        print(f"\n✓ 評価完了")
        print(f"  出力: {output_file}")


def suppress_noise():
    """サードパーティライブラリのノイズログを抑制

    Note: Streamlit の ScriptRunContext 警告は .streamlit/config.toml の
    [logger] level = "error" で抑制済み（アプリコード実行前に効く）。
    ここではアプリコード実行後に初期化されるサードパーティのみ対象。
    """
    noisy_loggers = [
        "chromadb", "chromadb.config", "chromadb.telemetry",
        "httpx", "httpcore", "urllib3",
        "google.auth", "google.api_core", "google.cloud",
        "azure.core", "azure.identity",
        "altair",
    ]
    for name in noisy_loggers:
        logging.getLogger(name).setLevel(logging.WARNING)


def print_startup_summary(app_name: str, checks: list):
    """起動サマリをダッシュボード形式で表示

    Args:
        app_name: アプリ名（例: "回答支援AI v1.0"）
        checks: [(label, ok, detail), ...] のリスト
    """
    console = get_console()
    if RICH_AVAILABLE:
        console.print()
        console.print(Rule(f" {app_name} ", style="bold cyan"))
        console.print()
        for label, ok, detail in checks:
            icon = "[green]\u2714[/green]" if ok else "[red]\u2717[/red]"
            console.print(f"  {icon} {label} {detail}")
        console.print()
        console.print(Rule(style="dim"))
    else:
        print(f"\n{'=' * 40}")
        print(f"  {app_name}")
        print(f"{'=' * 40}")
        for label, ok, detail in checks:
            icon = "OK" if ok else "NG"
            print(f"  [{icon}] {label} {detail}")
        print(f"{'─' * 40}")


def print_query_panel(
    query_number: int,
    query_text: str,
    metadata: dict,
    results: dict,
    elapsed: float,
):
    """検索リクエストを構造化パネルで表示

    Args:
        query_number: クエリ番号
        query_text: 検索テキスト（80文字で切り詰め）
        metadata: キー=値ペアの辞書（改定番号、検索タイプ等）
        results: プロバイダー名=件数の辞書
        elapsed: 所要時間（秒）
    """
    console = get_console()
    truncated = query_text[:80] + "..." if len(query_text) > 80 else query_text
    meta_str = "  ".join(f"{k}: {v}" for k, v in metadata.items())
    result_str = "  ".join(f"{k}: {v}件" for k, v in results.items())

    if RICH_AVAILABLE:
        title = f"検索 #{query_number}"
        if metadata:
            title += f"  {meta_str}"
        lines = [f"  Q: {truncated}"]
        if results:
            lines.append(f"  {result_str}")
        lines.append(f"  所要: {elapsed:.1f}s")
        content = "\n".join(lines)
        console.print(Panel(
            content,
            title=title,
            title_align="left",
            border_style="cyan",
            padding=(0, 1),
        ))
    else:
        print(f"\n--- 検索 #{query_number} {meta_str} ---")
        print(f"  Q: {truncated}")
        if results:
            print(f"  {result_str}")
        print(f"  所要: {elapsed:.1f}s")
        print(f"{'─' * 35}")
