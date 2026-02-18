#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FAQデータのDB構築スクリプト

使用方法:
    python scripts/rebuild_faq_db.py

処理内容:
    - reference/faq_data/ のFAQファイルを検出
    - 業務分野別にベクトルDBを構築
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from config import SearchConfig
from src.utils.dynamic_db_manager import DynamicDBManager
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# 使用するプロバイダー（azure_openai または vertex_ai）
PROVIDER = os.getenv("EMBEDDING_PROVIDER", "azure_openai")


def main() -> None:
    print("=" * 60)
    print("FAQ DB 構築スクリプト")
    print(f"プロバイダー: {PROVIDER}")
    print("=" * 60)

    config = SearchConfig(base_dir=str(PROJECT_ROOT))
    config.embedding_provider = PROVIDER
    config.force_db_update = True  # 強制更新

    with DynamicDBManager(config) as db_manager:
        # FAQファイルを検出
        business_areas = db_manager.analyze_reference_files()

        print(f"\n検出された業務分野: {list(business_areas.keys())}")

        for area, files in business_areas.items():
            print(f"\n--- {area} ---")
            print(f"  FAQ: {files.get('faq', [])}")
            print(f"  シナリオ: {files.get('scenario', [])}")

            # FAQファイルがある業務分野のみ構築
            if files.get('faq'):
                print(f"\n[構築中] {area}...")
                try:
                    db_manager.update_business_db(area, files)
                    print(f"[完了] {area}")
                except Exception as e:
                    print(f"[エラー] {area}: {e}")

    print("\n" + "=" * 60)
    print("FAQ DB 構築完了！")
    print("=" * 60)
    print("\nUIを起動するには:")
    print("  streamlit run apps/answer-support/ui/chat.py")


if __name__ == "__main__":
    main()
