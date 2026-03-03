# --- main.py ---
import sys
import os
import subprocess
import argparse
from pathlib import Path

# apps/answer-support/ → rag-local/ ルートへのパス解決
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from config import SearchConfig
from src.core.processor import Processor
from src.utils.dynamic_db_manager import DynamicDBManager, DynamicDBError
from src.utils.logger import setup_logger

# 環境変数の読み込み
load_dotenv(PROJECT_ROOT / ".env")
logger = setup_logger(__name__)


def parse_args():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(
        description="RAG検索システム",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python main.py                        # 全業務分野でバッチ処理
  python main.py --business naibujimu   # 内部事務のみバッチ処理
  python main.py interactive            # UIモード（DB更新はオンデマンド）
  python main.py preflight --business smile  # プレフライト（事前検証）
        """
    )

    # サブコマンド
    subparsers = parser.add_subparsers(dest="command", help="実行モード")

    # interactiveサブコマンド
    subparsers.add_parser("interactive", help="StreamlitベースのUIモード")

    # preflightサブコマンド
    preflight_parser = subparsers.add_parser("preflight", help="DB更新プレフライト（本番更新は行いません）")
    preflight_parser.add_argument("--business", dest="business", default=None, help="対象の業務分野（naibujimu / smile）")
    preflight_parser.add_argument("--sample-size", dest="sample_size", type=int, default=5, help="検証に使うサンプル件数")

    # メイン（バッチ）モードのオプション
    parser.add_argument("--business", dest="business", default=None, help="対象の業務分野（naibujimu / smile）。未指定時は全業務分野")
    parser.add_argument("--limit", dest="limit", type=int, default=None, help="処理する入力データの件数上限（例: --limit 5 で先頭5件のみ）")

    return parser.parse_args()


def run_preflight(config, args):
    """プレフライト実行"""
    try:
        logger.info("動的DB管理システムを初期化中（preflight）...")
        db_manager = DynamicDBManager(config)
        reference_files = db_manager.analyze_reference_files(include_revisions=False)

        if args.business:
            targets = {k: v for k, v in reference_files.items() if k == args.business}
            if not targets:
                logger.error(f"指定された業務分野が見つかりません: {args.business}")
                logger.info(f"検出された業務分野: {list(reference_files.keys())}")
                sys.exit(1)
        else:
            targets = reference_files

        for business_area, files in targets.items():
            logger.info(f"業務分野 '{business_area}' のプレフライト開始")
            result = db_manager.preflight_business_db(
                business_area=business_area,
                files=files,
                sample_size=args.sample_size,
            )
            logger.info(
                f"プレフライトOK: {result['business_area']} (sample={result['sample_size']}, dim={result['embedding_dim']})"
            )

        logger.info("プレフライト完了: すべてOK")
        sys.exit(0)
    except DynamicDBError as e:
        logger.error(f"プレフライト失敗: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"予期しないエラー（preflight）: {e}")
        sys.exit(1)


def run_db_update(config, business_filter=None):
    """DB更新を実行（業務分野フィルタ対応）"""
    try:
        logger.info("動的DB管理システムを初期化中...")
        db_manager = DynamicDBManager(config)

        # 参照ファイルの分析
        reference_files = db_manager.analyze_reference_files(include_revisions=False)

        # 業務分野フィルタの適用
        if business_filter:
            if business_filter not in reference_files:
                logger.error(f"指定された業務分野が見つかりません: {business_filter}")
                logger.info(f"検出された業務分野: {list(reference_files.keys())}")
                sys.exit(1)
            targets = {business_filter: reference_files[business_filter]}
            logger.info(f"業務分野 '{business_filter}' のみを処理対象とします")
        else:
            targets = reference_files
            logger.info(f"全業務分野を処理対象とします: {list(reference_files.keys())}")

        # 業務分野ごとのDB更新
        for business_area, files in targets.items():
            logger.info(f"業務分野 '{business_area}' の処理開始")
            db_manager.update_business_db(business_area, files)

        logger.info("動的DB管理システムの初期化完了")
        return db_manager

    except DynamicDBError as e:
        logger.error(f"動的DB管理エラー: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"予期しないエラー: {e}")
        sys.exit(1)


def main():
    args = parse_args()

    # 設定の初期化
    config = SearchConfig(base_dir=str(PROJECT_ROOT))

    # プレフライトモード
    if args.command == "preflight":
        run_preflight(config, args)
        return

    # インタラクティブモード（UIモード）- DB更新は検索時にオンデマンド実行
    if args.command == "interactive":
        logger.info("Starting in interactive mode (DB updates on-demand)")
        config.vector_weight = config.DEFAULT_VECTOR_WEIGHT
        process = None
        try:
            # Subprocess Security: sys.executableの検証
            python_executable = sys.executable
            if not python_executable or not os.path.isfile(python_executable):
                raise RuntimeError(f"Invalid Python executable: {python_executable}")
            if not os.access(python_executable, os.X_OK):
                raise RuntimeError(f"Python executable is not executable: {python_executable}")

            # subprocessを使用してStreamlitを起動（セキュリティ向上）
            import time
            ui_chat_path = str(Path(__file__).parent / "ui" / "chat.py")
            process = subprocess.Popen([python_executable, "-m", "streamlit", "run", ui_chat_path])
            # プロセスが起動したか確認（最大5秒待機）
            startup_timeout = 5
            poll_interval = 0.5
            elapsed = 0
            while elapsed < startup_timeout:
                if process.poll() is not None:
                    logger.error(f"Streamlit process exited with code: {process.returncode}")
                    sys.exit(1)
                time.sleep(poll_interval)
                elapsed += poll_interval
            logger.info("Streamlit app started successfully")

            # パフォーマンス/安定性: プロセス終了まで待機（リソースリーク防止）
            process.wait()
            logger.info("Streamlit process exited normally")
            sys.exit(0)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received, terminating Streamlit...")
            if process:
                process.terminate()
                process.wait(timeout=5)
            sys.exit(0)
        except Exception as e:
            logger.error(f"Failed to start Streamlit: {e}")
            if process:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning("Process did not terminate within 5 seconds, killing...")
                    process.kill()
            sys.exit(1)

    # バッチモード（デフォルト）
    else:
        # DB更新（業務分野フィルタ対応）
        business_filter = getattr(args, 'business', None)
        run_db_update(config, business_filter)

        logger.info("Starting in batch mode")
        processor = Processor(config)
        limit = getattr(args, 'limit', None)
        processor.process_data(mode="batch", limit=limit)

if __name__ == "__main__":
    main()