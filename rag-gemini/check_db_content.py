#!/usr/bin/env python3
"""
データベースの内容と重複を確認するスクリプト
"""

from utils.vector_db import MetadataVectorDB
from collections import Counter

def check_db_content():
    """データベースの内容を確認"""
    db = MetadataVectorDB()
    results = db.collection.get()
    
    documents = results['documents']
    metadatas = results['metadatas']
    ids = results['ids']
    
    print("=== データベース内容確認 ===")
    print(f"総ドキュメント数: {len(documents)}")
    print(f"ユニークドキュメント数: {len(set(documents))}")
    print(f"重複ドキュメント数: {len(documents) - len(set(documents))}")
    
    # ソース別の件数
    print("\n=== ソース別件数 ===")
    source_count = Counter()
    for meta in metadatas:
        source = meta.get('source', 'unknown')
        source_count[source] += 1
    
    for source, count in source_count.items():
        print(f"  {source}: {count}件")
    
    # 重複ドキュメントの詳細
    print("\n=== 重複ドキュメントの詳細 ===")
    doc_counter = Counter(documents)
    duplicates = {doc: count for doc, count in doc_counter.items() if count > 1}
    
    if duplicates:
        print(f"重複しているドキュメント数: {len(duplicates)}")
        for i, (doc, count) in enumerate(list(duplicates.items())[:5]):  # 最初の5件のみ表示
            print(f"\n重複ドキュメント {i+1} (出現回数: {count}):")
            print(f"内容: {doc[:100]}...")
            
            # このドキュメントのメタデータを確認
            doc_indices = [i for i, d in enumerate(documents) if d == doc]
            print("メタデータ:")
            for idx in doc_indices:
                meta = metadatas[idx]
                print(f"  ID: {ids[idx]}, Source: {meta.get('source')}, Row: {meta.get('row_index')}")
    else:
        print("重複ドキュメントはありません")
    
    # 特定の質問の重複を確認
    print("\n=== 特定質問の重複確認 ===")
    target_question = "本日通達の営業係預り物件自宅一時保管について"
    matching_docs = [(i, doc) for i, doc in enumerate(documents) if target_question in doc]
    
    if matching_docs:
        print(f"'{target_question}'を含むドキュメント数: {len(matching_docs)}")
        for idx, doc in matching_docs:
            meta = metadatas[idx]
            print(f"  ID: {ids[idx]}, Source: {meta.get('source')}, Row: {meta.get('row_index')}")
            print(f"  内容: {doc[:200]}...")
    else:
        print(f"'{target_question}'を含むドキュメントは見つかりませんでした")

if __name__ == "__main__":
    check_db_content()




