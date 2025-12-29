# --- main.py ---
import sys
import os
from dotenv import load_dotenv
from config import SearchConfig
from processor import Processor
from utils.logger import setup_logger

# 環境変数の読み込み
load_dotenv()
logger = setup_logger(__name__)

def main():
    # 設定の初期化
    config = SearchConfig(base_dir=os.path.dirname(os.path.abspath(__file__)))

    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        logger.info("Starting in interactive mode")
        config.is_interactive = True
        config.vector_weight = config.DEFAULT_UI_VECTOR_WEIGHT
        mode = "chat"
        os.system("streamlit run chat.py")
    else:
        logger.info("Starting in batch mode")
        config.is_interactive = False
        mode = "batch"
        processor = Processor(config)
        processor.process_data(mode=mode)

if __name__ == "__main__":
    main()