#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
変更前シナリオDBの再構築スクリプト

使用方法:
    1. Streamlit UIを停止
    2. python scripts/rebuild_before_scenario_db.py

処理内容:
    - rev* DBディレクトリを削除
    - タイムスタンプファイルをリセット
    - 全9つのDBをAzure OpenAIとVertexAI両方で再構築
"""

import os
import sys
import shutil
import json
from pathlib import Path

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from config import SearchConfig
from src.utils.dynamic_db_manager import DynamicDBManager
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# 再構築対象の業務分野
REVISION_AREAS = [
    "rev01smile",
    "rev02souzoku",
    "rev03smile",
    "rev03naibujimu",
    "rev03souzoku",
    "rev03torikaku",
    "rev04naibujimu",
    "rev05smile",
    "rev06smile",
]

# 対応するプロバイダー
PROVIDERS = ["azure_openai", "vertex_ai"]


def delete_existing_dbs():
    """既存のrev* DBディレクトリを削除"""
    db_base = PROJECT_ROOT / "reference" / "vector_db"

    for area in REVISION_AREAS:
        db_path = db_base / area
        if db_path.exists():
            try:
                shutil.rmtree(db_path)
                logger.info(f"削除完了: {db_path}")
            except Exception as e:
                logger.error(f"削除エラー: {db_path} - {e}")
                return False

    return True


def reset_timestamps():
    """タイムスタンプファイルからrev*エントリを削除"""
    timestamp_file = PROJECT_ROOT / "reference" / "vector_db" / "update_timestamps.json"

    if not timestamp_file.exists():
        logger.info("タイムスタンプファイルが存在しません")
        return True

    try:
        with open(timestamp_file, 'r', encoding='utf-8') as f:
            timestamps = json.load(f)

        # rev*エントリを削除
        keys_to_remove = [k for k in timestamps.keys() if k.startswith("rev")]
        for key in keys_to_remove:
            del timestamps[key]
            logger.info(f"タイムスタンプ削除: {key}")

        with open(timestamp_file, 'w', encoding='utf-8') as f:
            json.dump(timestamps, f, ensure_ascii=False, indent=2)

        logger.info("タイムスタンプファイル更新完了")
        return True
    except Exception as e:
        logger.error(f"タイムスタンプ更新エラー: {e}")
        return False


def rebuild_dbs():
    """全DBを両プロバイダーで再構築"""
    for provider in PROVIDERS:
        logger.info(f"\n{'='*60}")
        logger.info(f"=== {provider} でDB構築開始 ===")
        logger.info(f"{'='*60}")

        # プロバイダー別に設定を作成
        config = SearchConfig(base_dir=str(PROJECT_ROOT))
        config.embedding_provider = provider

        # embedding_modelをプロバイダーに応じて環境変数から取得
        if provider == "azure_openai":
            config.embedding_model = os.getenv(
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"
            )
        else:  # vertex_ai
            config.embedding_model = os.getenv(
                "VERTEX_AI_EMBEDDING_MODEL", "gemini-embedding-001"
            )

        with DynamicDBManager(config) as db_manager:
            # 参照ファイルの分析
            business_areas = db_manager.analyze_reference_files()

            logger.info(f"検出された業務分野: {list(business_areas.keys())}")

            # rev*業務分野のみ再構築
            for area in REVISION_AREAS:
                if area not in business_areas:
                    logger.warning(f"業務分野が見つかりません: {area}")
                    continue

                logger.info(f"=== {area} ({provider}) の再構築開始 ===")
                try:
                    db_manager.update_business_db(area, business_areas[area])
                    logger.info(f"=== {area} ({provider}) の再構築完了 ===")
                except Exception as e:
                    logger.error(f"{area} ({provider}) の再構築エラー: {e}")
                    raise

        logger.info(f"=== {provider} でDB構築完了 ===\n")


def main():
    print("=" * 60)
    print("変更前シナリオDB 再構築スクリプト")
    print("（Azure OpenAI + VertexAI 両プロバイダー対応）")
    print("=" * 60)

    # Step 1: 既存DBの削除
    print("\n[Step 1/3] 既存DBの削除...")
    if not delete_existing_dbs():
        print("エラー: DBディレクトリの削除に失敗しました")
        print("Streamlit UIが実行中の場合は停止してください")
        sys.exit(1)

    # Step 2: タイムスタンプのリセット
    print("\n[Step 2/3] タイムスタンプのリセット...")
    if not reset_timestamps():
        print("エラー: タイムスタンプの更新に失敗しました")
        sys.exit(1)

    # Step 3: DB再構築（両プロバイダー）
    print("\n[Step 3/3] DB再構築（Azure OpenAI + VertexAI）...")
    rebuild_dbs()

    print("\n" + "=" * 60)
    print("再構築完了！（Azure OpenAI + VertexAI 両方）")
    print("=" * 60)


if __name__ == "__main__":
    main()
