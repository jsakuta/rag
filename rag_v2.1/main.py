# --- main.py ---
import sys
import os
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
        config.is_interactive = True
        config.vector_weight = config.DEFAULT_UI_VECTOR_WEIGHT
        mode = "chat"
        os.system("streamlit run ui/chat.py")
    else:
        logger.info("Starting in batch mode")
        config.is_interactive = False
        mode = "batch"
        processor = Processor(config)
        processor.process_data(mode=mode)

if __name__ == "__main__":
    main()