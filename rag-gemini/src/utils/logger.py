# --- utils/logger.py ---
import logging
import os
import sys
from typing import Optional

# richがインストールされているか確認
try:
    from rich.logging import RichHandler
    from rich.console import Console
    from rich.theme import Theme
    RICH_AVAILABLE = True

    # カスタムテーマ（色設定）
    CUSTOM_THEME = Theme({
        "info": "cyan",
        "warning": "yellow",
        "error": "red bold",
        "critical": "red bold reverse",
        "debug": "dim",
        "success": "green",
        "highlight": "bold magenta",
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
    """セクション区切りを出力"""
    if RICH_AVAILABLE:
        console = get_console()
        console.rule(f"[bold]{title}[/bold]", style="cyan")
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
        table = Table(title=title, show_header=True, header_style="bold cyan")
        for col in columns:
            table.add_column(col)
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
            "info": "[cyan]ℹ[/cyan]",
            "success": "[green]✓[/green]",
            "warning": "[yellow]⚠[/yellow]",
            "error": "[red]✗[/red]",
        }
        icon = icons.get(status, icons["info"])
        console.print(f"{icon} {message}")
    else:
        icons = {
            "info": "[i]",
            "success": "[+]",
            "warning": "[!]",
            "error": "[x]",
        }
        icon = icons.get(status, icons["info"])
        print(f"{icon} {message}")
