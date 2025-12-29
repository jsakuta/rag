import sys
import os
from dotenv import load_dotenv
from config import SearchConfig
from src.core.processor import ExcelVectorProcessor
from src.utils.utils import setup_logger

# 環境変数の読み込み
load_dotenv()
logger = setup_logger(__name__)

def main():
    # base_dirのみ動的に設定し、他はデフォルト値を使用
    config = SearchConfig(
        base_dir=os.path.dirname(os.path.abspath(__file__))
    )
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        logger.info("Starting in interactive mode")
        # UI用の設定値に変更
        config.vector_weight = SearchConfig.DEFAULT_UI_VECTOR_WEIGHT
        os.system("streamlit run ui/ui.py")
    else:
        logger.info("Starting in batch mode")
        processor = ExcelVectorProcessor(config)
        processor.process_files()

if __name__ == "__main__":
    main()