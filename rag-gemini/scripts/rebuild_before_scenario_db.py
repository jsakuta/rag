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

import json
import os
import shutil
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

PROVIDERS = ["azure_openai", "vertex_ai"]

EMBEDDING_MODELS = {
    "azure_openai": ("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"),
    "vertex_ai": ("VERTEX_AI_EMBEDDING_MODEL", "gemini-embedding-001"),
}


def delete_existing_dbs() -> bool:
    """既存のrev* DBディレクトリを削除"""
    db_base = PROJECT_ROOT / "reference" / "vector_db"

    for area in REVISION_AREAS:
        db_path = db_base / area
        if not db_path.exists():
            continue
        try:
            shutil.rmtree(db_path)
            logger.info(f"削除完了: {db_path}")
        except Exception as e:
            logger.error(f"削除エラー: {db_path} - {e}")
            return False

    return True


def reset_timestamps() -> bool:
    """タイムスタンプファイルからrev*エントリを削除"""
    timestamp_file = PROJECT_ROOT / "reference" / "vector_db" / "update_timestamps.json"

    if not timestamp_file.exists():
        logger.info("タイムスタンプファイルが存在しません")
        return True

    try:
        with open(timestamp_file, "r", encoding="utf-8") as f:
            timestamps = json.load(f)

        keys_to_remove = [k for k in timestamps if k.startswith("rev")]
        for key in keys_to_remove:
            del timestamps[key]
            logger.info(f"タイムスタンプ削除: {key}")

        with open(timestamp_file, "w", encoding="utf-8") as f:
            json.dump(timestamps, f, ensure_ascii=False, indent=2)

        logger.info("タイムスタンプファイル更新完了")
        return True
    except Exception as e:
        logger.error(f"タイムスタンプ更新エラー: {e}")
        return False


def rebuild_dbs() -> None:
    """全DBを両プロバイダーで再構築"""
    for provider in PROVIDERS:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"=== {provider} でDB構築開始 ===")
        logger.info(f"{'=' * 60}")

        config = SearchConfig(base_dir=str(PROJECT_ROOT))
        config.embedding_provider = provider

        env_key, default_model = EMBEDDING_MODELS[provider]
        config.embedding_model = os.getenv(env_key, default_model)

        with DynamicDBManager(config) as db_manager:
            business_areas = db_manager.analyze_reference_files()
            logger.info(f"検出された業務分野: {list(business_areas.keys())}")

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


def main() -> None:
    print("=" * 60)
    print("変更前シナリオDB 再構築スクリプト")
    print("（Azure OpenAI + VertexAI 両プロバイダー対応）")
    print("=" * 60)

    steps = [
        ("既存DBの削除", delete_existing_dbs),
        ("タイムスタンプのリセット", reset_timestamps),
    ]

    for i, (name, func) in enumerate(steps, 1):
        print(f"\n[Step {i}/3] {name}...")
        if not func():
            print(f"エラー: {name}に失敗しました")
            if i == 1:
                print("Streamlit UIが実行中の場合は停止してください")
            sys.exit(1)

    print("\n[Step 3/3] DB再構築（Azure OpenAI + VertexAI）...")
    rebuild_dbs()

    print("\n" + "=" * 60)
    print("再構築完了！（Azure OpenAI + VertexAI 両方）")
    print("=" * 60)


if __name__ == "__main__":
    main()
