# --- utils/logger.py ---
import logging
import os


def setup_logger(name):
    """ロガーの設定"""
    logger = logging.getLogger(name)

    # 環境変数でログレベルを制御
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    level = getattr(logging, log_level, None)
    if level is None:
        # 無効なログレベルが指定された場合は警告してデフォルト(INFO)を使用
        print(f"[WARNING] Invalid LOG_LEVEL '{log_level}' specified. Using default 'INFO'.")
        level = logging.INFO
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
    return logger
