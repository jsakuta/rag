"""
空ディレクトリを検出・削除するスクリプト
"""
import os
from pathlib import Path

def remove_empty_directories(base_dir: str, dry_run: bool = False):
    """空ディレクトリを再帰的に削除"""
    base_path = Path(base_dir)
    
    # 除外するディレクトリ（.gitなど）
    exclude_dirs = {".git", ".venv", "venv", "node_modules", ".vscode"}
    
    removed_count = 0
    
    # ボトムアップで走査（子ディレクトリから削除）
    for dirpath, dirnames, filenames in os.walk(base_path, topdown=False):
        # 除外ディレクトリをスキップ
        dirpath_obj = Path(dirpath)
        if any(excluded in dirpath_obj.parts for excluded in exclude_dirs):
            continue
        
        # ディレクトリが空かチェック
        try:
            if not any(dirpath_obj.iterdir()):
                if dry_run:
                    print(f"[DRY RUN] 削除予定: {dirpath_obj.relative_to(base_path)}")
                else:
                    dirpath_obj.rmdir()
                    print(f"削除: {dirpath_obj.relative_to(base_path)}")
                removed_count += 1
        except Exception as e:
            # OSError: Directory not emptyなどは無視
            pass
    
    if dry_run:
        print(f"\n[DRY RUN] {removed_count}個の空ディレクトリが見つかりました")
    else:
        print(f"\n完了: {removed_count}個の空ディレクトリを削除しました")

if __name__ == "__main__":
    base_dir = r"C:\VSCode\rag\rag-gemini"
    
    # まずドライラン
    print("=== ドライラン（削除予定の確認） ===")
    remove_empty_directories(base_dir, dry_run=True)
    
    print("\n=== 実際の削除 ===")
    remove_empty_directories(base_dir, dry_run=False)
