#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DB構築スクリプト（Azure OpenAI + VertexAI 両プロバイダー対応）

使用方法:
    python scripts/build_db.py                          # 全業務分野（差分のみ構築）
    python scripts/build_db.py --force                   # 全業務分野（全再構築）
    python scripts/build_db.py --business naibujimu      # 特定業務分野のみ
    python scripts/build_db.py --revisions-only          # 改定別（rev*）のみ
    python scripts/build_db.py --no-revisions            # 通常業務のみ
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from config import SearchConfig
from src.utils.dynamic_db_manager import DynamicDBManager
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

PROVIDERS = ["azure_openai", "vertex_ai"]

EMBEDDING_MODELS = {
    "azure_openai": ("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"),
    "vertex_ai": ("VERTEX_AI_EMBEDDING_MODEL", "gemini-embedding-001"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DB構築（Azure OpenAI + VertexAI）")
    parser.add_argument("--force", action="store_true", help="既存DBを削除して全再構築")
    parser.add_argument("--business", type=str, default=None,
                        help="構築対象の業務分野（例: naibujimu, smile, rev02_souzoku）")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--revisions-only", action="store_true", help="改定別（rev*）のみ")
    group.add_argument("--no-revisions", action="store_true", help="通常業務のみ（rev*除外）")
    return parser.parse_args()


def delete_existing_dbs(db_base: Path, target_areas: list[str]) -> bool:
    for area in target_areas:
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


def reset_timestamps(timestamp_file: Path, target_areas: list[str]) -> bool:
    if not timestamp_file.exists():
        return True
    try:
        with open(timestamp_file, "r", encoding="utf-8") as f:
            timestamps = json.load(f)
        keys_to_remove = [k for k in timestamps
                          if any(k.startswith(f"{area}_") for area in target_areas)]
        for key in keys_to_remove:
            del timestamps[key]
            logger.info(f"タイムスタンプ削除: {key}")
        with open(timestamp_file, "w", encoding="utf-8") as f:
            json.dump(timestamps, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"タイムスタンプ更新エラー: {e}")
        return False


def build_dbs(args: argparse.Namespace) -> None:
    all_results = []
    include_revisions = not args.no_revisions

    for provider in PROVIDERS:
        print(f"\n{'=' * 60}")
        print(f"=== {provider} ===")
        print(f"{'=' * 60}")

        config = SearchConfig(base_dir=str(PROJECT_ROOT))
        config.embedding_provider = provider
        env_key, default_model = EMBEDDING_MODELS[provider]
        config.embedding_model = os.getenv(env_key, default_model)

        with DynamicDBManager(config) as db_manager:
            business_areas = db_manager.analyze_reference_files(
                include_revisions=include_revisions)

            # --revisions-only: rev* のみに絞る
            if args.revisions_only:
                business_areas = {k: v for k, v in business_areas.items()
                                  if k.startswith("rev")}

            # --business: 指定分野のみ
            if args.business:
                if args.business not in business_areas:
                    print(f"エラー: '{args.business}' が見つかりません")
                    print(f"利用可能: {list(business_areas.keys())}")
                    sys.exit(1)
                business_areas = {args.business: business_areas[args.business]}

            target_areas = list(business_areas.keys())
            logger.info(f"対象業務分野: {target_areas}")

            # --force: DB削除 + タイムスタンプリセット
            if args.force:
                db_base = PROJECT_ROOT / "data" / "vector_db"
                timestamp_file = db_base / "update_timestamps.json"
                print(f"  [Force] 既存DB削除（{len(target_areas)}件）...")
                if not delete_existing_dbs(db_base, target_areas):
                    print("エラー: DB削除失敗。Streamlit UIが実行中の場合は停止してください")
                    sys.exit(1)
                print("  [Force] タイムスタンプリセット...")
                if not reset_timestamps(timestamp_file, target_areas):
                    print("エラー: タイムスタンプリセット失敗")
                    sys.exit(1)

            for area in target_areas:
                print(f"\n--- {area} ({provider}) ---")
                start_time = time.time()
                try:
                    db_manager.update_business_db(area, business_areas[area])
                    elapsed = time.time() - start_time
                    db_path = db_manager.get_db_path_for_business(area)
                    sqlite_path = os.path.join(db_path, "chroma.sqlite3")
                    all_results.append({
                        "area": area, "provider": provider,
                        "status": "OK" if os.path.exists(sqlite_path) else "WARN",
                        "elapsed": elapsed,
                    })
                except Exception as e:
                    elapsed = time.time() - start_time
                    all_results.append({
                        "area": area, "provider": provider,
                        "status": "ERROR", "elapsed": elapsed, "error": str(e),
                    })
                    logger.error(f"{area} ({provider}) エラー: {e}")

        print(f"\n=== {provider} 完了 ===")

    # サマリ
    print("\n" + "=" * 60)
    print("構築結果サマリ")
    print("=" * 60)
    print(f"{'業務分野':<20} {'プロバイダー':<15} {'ステータス':<10} {'所要時間':<10}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['area']:<20} {r['provider']:<15} {r['status']:<10} {r['elapsed']:.1f}秒")
    print("=" * 60)

    if any(r["status"] == "ERROR" for r in all_results):
        sys.exit(1)


def main() -> None:
    args = parse_args()
    mode = "全再構築（--force）" if args.force else "差分構築"
    scope = (args.business if args.business
             else "改定別のみ" if args.revisions_only
             else "通常業務のみ" if args.no_revisions
             else "全業務分野")
    print("=" * 60)
    print(f"DB構築スクリプト | モード: {mode} | 対象: {scope}")
    print("=" * 60)
    build_dbs(args)


if __name__ == "__main__":
    main()
