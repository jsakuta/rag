#!/usr/bin/env python3
"""
データベースの内容と重複を確認するスクリプト
"""

from collections import Counter

from src.utils.vector_db import MetadataVectorDB


def check_db_content() -> None:
    """データベースの内容を確認"""
    db = MetadataVectorDB()
    results = db.collection.get()

    documents = results["documents"]
    metadatas = results["metadatas"]
    ids = results["ids"]

    unique_count = len(set(documents))
    duplicate_count = len(documents) - unique_count

    print("=== データベース内容確認 ===")
    print(f"総ドキュメント数: {len(documents)}")
    print(f"ユニークドキュメント数: {unique_count}")
    print(f"重複ドキュメント数: {duplicate_count}")

    # ソース別の件数
    print("\n=== ソース別件数 ===")
    source_count = Counter(meta.get("source", "unknown") for meta in metadatas)
    for source, count in source_count.items():
        print(f"  {source}: {count}件")

    # 重複ドキュメントの詳細
    print("\n=== 重複ドキュメントの詳細 ===")
    doc_counter = Counter(documents)
    duplicates = [(doc, cnt) for doc, cnt in doc_counter.items() if cnt > 1]

    if not duplicates:
        print("重複ドキュメントはありません")
        return

    print(f"重複しているドキュメント数: {len(duplicates)}")
    for i, (doc, count) in enumerate(duplicates[:5]):
        print(f"\n重複ドキュメント {i + 1} (出現回数: {count}):")
        print(f"内容: {doc[:100]}...")

        doc_indices = [j for j, d in enumerate(documents) if d == doc]
        print("メタデータ:")
        for idx in doc_indices:
            meta = metadatas[idx]
            print(f"  ID: {ids[idx]}, Source: {meta.get('source')}, Row: {meta.get('row_index')}")


if __name__ == "__main__":
    check_db_content()
