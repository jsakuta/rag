# --- main.py ---
import sys
import os
import subprocess
from dotenv import load_dotenv
from config import SearchConfig
from src.core.processor import Processor
from src.utils.dynamic_db_manager import DynamicDBManager, DynamicDBError
from src.utils.logger import setup_logger

# 環境変数の読み込み
load_dotenv()
logger = setup_logger(__name__)

def main():
    # 設定の初期化
    config = SearchConfig(base_dir=os.path.dirname(os.path.abspath(__file__)))

    # プレフライト（DB更新の事前チェック）
    if len(sys.argv) > 1 and sys.argv[1] == "preflight":
        import argparse

        parser = argparse.ArgumentParser(description="DB更新プレフライト（本番更新は行いません）")
        parser.add_argument("--business", dest="business", default=None, help="対象の業務分野（例: 預金）")
        parser.add_argument("--sample-size", dest="sample_size", type=int, default=5, help="検証に使うサンプル件数")
        args = parser.parse_args(sys.argv[2:])

        try:
            logger.info("動的DB管理システムを初期化中（preflight）...")
            db_manager = DynamicDBManager(config)
            reference_files = db_manager.analyze_reference_files()

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

    # 動的DB管理システムの初期化
    try:
        logger.info("動的DB管理システムを初期化中...")
        db_manager = DynamicDBManager(config)
        
        # 参照ファイルの分析
        reference_files = db_manager.analyze_reference_files()
        
        # 業務分野ごとのDB更新
        for business_area, files in reference_files.items():
            logger.info(f"業務分野 '{business_area}' の処理開始")
            db_manager.update_business_db(business_area, files)
        
        logger.info("動的DB管理システムの初期化完了")
        
    except DynamicDBError as e:
        logger.error(f"動的DB管理エラー: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"予期しないエラー: {e}")
        sys.exit(1)

    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        logger.info("Starting in interactive mode")
        config.vector_weight = config.DEFAULT_UI_VECTOR_WEIGHT
        try:
            # subprocessを使用してStreamlitを起動（セキュリティ向上）
            import time
            process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "ui/chat.py"])
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
            sys.exit(0)
        except Exception as e:
            logger.error(f"Failed to start Streamlit: {e}")
            sys.exit(1)
    else:
        logger.info("Starting in batch mode")
        processor = Processor(config)
        processor.process_data(mode="batch")

if __name__ == "__main__":
    main()