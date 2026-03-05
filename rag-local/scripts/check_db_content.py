#!/usr/bin/env python3
"""
データベースの内容と重複を確認するスクリプト

使用方法:
    python scripts/check_db_content.py                    # 全コレクション確認
    python scripts/check_db_content.py --area smile       # 特定エリアのみ
    python scripts/check_db_content.py --detail           # 重複詳細も表示
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.vector_db import MetadataVectorDB

VECTOR_DB_BASE = PROJECT_ROOT / "data" / "vector_db"


def check_collection(db_path: str, area: str, provider: str, show_detail: bool) -> dict:
    """単一コレクションの内容を確認"""
    try:
        db = MetadataVectorDB(db_path=db_path, collection_name="default")
        count = db.collection.count()
        if count == 0:
            return {"area": area, "provider": provider, "count": 0, "unique": 0, "duplicates": 0}

        results = db.collection.get()
        documents = results["documents"]
        metadatas = results["metadatas"]
        ids = results["ids"]

        unique_count = len(set(documents))
        duplicate_count = len(documents) - unique_count

        # ソース別件数
        source_count = Counter(meta.get("source", "unknown") for meta in metadatas)

        info = {
            "area": area,
            "provider": provider,
            "count": count,
            "unique": unique_count,
            "duplicates": duplicate_count,
            "sources": dict(source_count),
        }

        if show_detail and duplicate_count > 0:
            doc_counter = Counter(documents)
            dupes = [(doc, cnt) for doc, cnt in doc_counter.items() if cnt > 1]
            info["duplicate_details"] = []
            for doc, cnt in dupes[:3]:
                doc_indices = [j for j, d in enumerate(documents) if d == doc]
                detail = {
                    "text": doc[:100],
                    "occurrences": cnt,
                    "entries": [
                        {"id": ids[idx], "source": metadatas[idx].get("source"), "row": metadatas[idx].get("row_index")}
                        for idx in doc_indices
                    ],
                }
                info["duplicate_details"].append(detail)

        return info
    except Exception as e:
        return {"area": area, "provider": provider, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="ベクトルDB内容確認")
    parser.add_argument("--area", type=str, default=None, help="確認対象エリア（例: smile, rev01_smile）")
    parser.add_argument("--detail", action="store_true", help="重複ドキュメントの詳細を表示")
    args = parser.parse_args()

    if not VECTOR_DB_BASE.exists():
        print(f"エラー: DBディレクトリが存在しません: {VECTOR_DB_BASE}")
        sys.exit(1)

    print("=" * 60)
    print("ベクトルDB内容確認")
    print("=" * 60)

    total_docs = 0
    total_duplicates = 0
    results = []

    for area_dir in sorted(VECTOR_DB_BASE.iterdir()):
        if not area_dir.is_dir():
            continue
        if args.area and area_dir.name != args.area:
            continue

        for provider_dir in sorted(area_dir.iterdir()):
            if not provider_dir.is_dir():
                continue

            info = check_collection(str(provider_dir), area_dir.name, provider_dir.name, args.detail)
            results.append(info)

            if "error" not in info:
                total_docs += info["count"]
                total_duplicates += info["duplicates"]

    # サマリーテーブル
    print(f"\n{'エリア':<25} {'プロバイダー':<15} {'件数':>6} {'ユニーク':>8} {'重複':>4}")
    print("-" * 65)
    for info in results:
        if "error" in info:
            print(f"{info['area']:<25} {info['provider']:<15} ERROR: {info['error']}")
        else:
            dup_mark = f" *{info['duplicates']}" if info["duplicates"] > 0 else ""
            print(f"{info['area']:<25} {info['provider']:<15} {info['count']:>6} {info['unique']:>8} {dup_mark:>4}")

    print("-" * 65)
    print(f"{'合計':<25} {'':15} {total_docs:>6} {'':>8} {total_duplicates:>4}")

    # ソース別詳細
    if any("sources" in info for info in results):
        print(f"\n{'=' * 60}")
        print("ソース別件数（azure_openaiのみ表示）")
        print("=" * 60)
        for info in results:
            if info.get("provider") != "azure_openai" or "sources" not in info:
                continue
            print(f"\n  {info['area']}:")
            for source, count in sorted(info["sources"].items()):
                print(f"    {source}: {count}件")

    # 重複詳細
    if args.detail:
        for info in results:
            if "duplicate_details" not in info:
                continue
            print(f"\n{'=' * 60}")
            print(f"重複詳細: {info['area']}/{info['provider']}")
            print("=" * 60)
            for detail in info["duplicate_details"]:
                print(f"\n  内容: {detail['text']}...")
                print(f"  出現回数: {detail['occurrences']}")
                for entry in detail["entries"]:
                    print(f"    ID={entry['id']}, Source={entry['source']}, Row={entry['row']}")

    if total_duplicates > 0:
        print(f"\n注意: 重複が {total_duplicates} 件あります。--detail で詳細を確認してください。")
    else:
        print("\n重複ドキュメントはありません。")


if __name__ == "__main__":
    main()
