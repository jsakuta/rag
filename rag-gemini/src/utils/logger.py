# --- utils/logger.py ---
import logging
import os


def setup_logger(name):
    """ロガーの設定"""
    logger = logging.getLogger(name)

    # セキュリティ: ホワイトリスト方式でログレベルを検証
    VALID_LOG_LEVELS = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()

    invalid_level = None
    if log_level not in VALID_LOG_LEVELS:
        # 無効なログレベルが指定された場合は後で警告ログを出力
        invalid_level = log_level
        log_level = 'INFO'

    level = getattr(logging, log_level)
    logger.setLevel(level)

    # ハンドラの重複追加を防止
    if logger.handlers:
        return logger

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'app.log')

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # ハンドラ設定後に無効なログレベルの警告を出力
    if invalid_level:
        logger.warning(
            f"Invalid LOG_LEVEL '{invalid_level}' specified in environment. "
            f"Using default 'INFO'. Valid levels: {VALID_LOG_LEVELS}"
        )

    return logger
