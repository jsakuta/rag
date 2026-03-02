#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""引き継ぎパッケージ作成スクリプト（許可リスト方式）

Usage:
    python scripts/create_handover_package.py DEST
    python scripts/create_handover_package.py DEST --include-data
    python scripts/create_handover_package.py DEST --dry-run

DEST に許可リストのファイルのみをコピーし、秘密情報の混入がないことを検証します。
"""

import argparse
import fnmatch
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- 許可リスト（これらのみコピーする） ---
# 注: CLAUDE.md はリストに含まれないため自動的に除外される
INCLUDE = [
    # ソースコード
    "apps/",
    "src/",
    "ui/",
    "config.py",
    "config/",
    "scripts/",
    "prompt/",
    "tests/",
    # ドキュメント
    "README.md",
    "docs/",
    # 設定テンプレート
    ".env.example",
    "requirements.txt",
    "requirements-dev.txt",
    "pytest.ini",
    ".streamlit/",
    # データ（--include-data 時のみ実データ、デフォルトは空ディレクトリ構造のみ）
    "data/source/",
    "data/input/",
]

# --- 明示的に除外（許可リスト内でも除外） ---
EXCLUDE_PATTERNS = [
    "__pycache__",
    "*.pyc",
    ".pytest_cache",
]

# --- 秘密情報検出パターン ---
SECRET_PATTERNS = [
    ".env",
    "*credentials*",
    "*.key",
]

# data/ 配下で --include-data 時のみ実データをコピーする対象
DATA_DIRS = ["data/source/", "data/input/"]


def _is_excluded(path: Path) -> bool:
    """EXCLUDE_PATTERNS に該当するかチェック"""
    for pattern in EXCLUDE_PATTERNS:
        # ディレクトリ名またはファイル名がパターンにマッチ
        for part in path.parts:
            if fnmatch.fnmatch(part, pattern):
                return True
    return False


def _is_data_dir_entry(rel_str: str) -> bool:
    """data/ 配下のエントリかどうかを判定"""
    for data_dir in DATA_DIRS:
        if rel_str == data_dir.rstrip("/") or rel_str.startswith(data_dir):
            return True
    return False


def _collect_files(include_data: bool) -> list[tuple[Path, Path]]:
    """許可リストに基づいてコピー対象ファイルを収集する。

    Returns:
        list of (source_absolute, relative_to_project_root) tuples
    """
    files: list[tuple[Path, Path]] = []

    for entry in INCLUDE:
        source = PROJECT_ROOT / entry

        if entry.endswith("/"):
            # ディレクトリエントリ
            if not source.is_dir():
                continue

            is_data = _is_data_dir_entry(entry)

            if is_data and not include_data:
                # --include-data なし: 空ディレクトリ構造のみ（ファイル収集しない）
                continue

            for file_path in source.rglob("*"):
                if not file_path.is_file():
                    continue
                rel = file_path.relative_to(PROJECT_ROOT)
                if _is_excluded(rel):
                    continue
                files.append((file_path, rel))
        else:
            # 単一ファイルエントリ
            if not source.is_file():
                continue
            rel = source.relative_to(PROJECT_ROOT)
            if _is_excluded(rel):
                continue
            files.append((source, rel))

    return files


def _collect_data_empty_dirs() -> list[Path]:
    """data/ 配下の空ディレクトリ構造として作成すべきパスを収集する"""
    dirs: list[Path] = []
    for data_dir in DATA_DIRS:
        source = PROJECT_ROOT / data_dir
        if not source.is_dir():
            # ディレクトリ自体が存在しなくても、空ディレクトリとして作成
            dirs.append(Path(data_dir))
            continue
        # サブディレクトリ構造を走査
        has_subdir = False
        for d in source.rglob("*"):
            if d.is_dir():
                rel = d.relative_to(PROJECT_ROOT)
                if not _is_excluded(rel):
                    dirs.append(rel)
                    has_subdir = True
        if not has_subdir:
            # サブディレクトリがなければルートだけ作成
            dirs.append(Path(data_dir.rstrip("/")))
    return dirs


def _check_secrets(dest: Path) -> list[Path]:
    """DEST ディレクトリをスキャンし、秘密情報ファイルを検出する"""
    found: list[Path] = []
    for file_path in dest.rglob("*"):
        if not file_path.is_file():
            continue
        name = file_path.name
        for pattern in SECRET_PATTERNS:
            if fnmatch.fnmatch(name, pattern):
                found.append(file_path.relative_to(dest))
                break
    return found


def _format_size(size_bytes: int) -> str:
    """バイト数を人間が読みやすい形式に変換する"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def run(dest_path: Path, include_data: bool, dry_run: bool) -> int:
    """メイン処理。成功時0、警告・エラー時1を返す。"""
    dest = dest_path.resolve()

    if dest.exists():
        print(f"[ERROR] 出力先が既に存在します: {dest}")
        print("  既存のディレクトリを削除するか、別の出力先を指定してください。")
        return 1

    # ファイル収集
    files = _collect_files(include_data)

    if not files:
        print("[ERROR] コピー対象ファイルが見つかりません。")
        print(f"  プロジェクトルート: {PROJECT_ROOT}")
        return 1

    # --- dry-run モード ---
    if dry_run:
        print(f"[DRY-RUN] プロジェクトルート: {PROJECT_ROOT}")
        print(f"[DRY-RUN] 出力先: {dest}")
        print(f"[DRY-RUN] --include-data: {include_data}")
        print()

        total_size = 0
        for _, rel in sorted(files, key=lambda x: x[1]):
            src = PROJECT_ROOT / rel
            size = src.stat().st_size
            total_size += size
            print(f"  {rel}")

        if not include_data:
            empty_dirs = _collect_data_empty_dirs()
            if empty_dirs:
                print()
                print("[DRY-RUN] 空ディレクトリ（data/）:")
                for d in sorted(empty_dirs):
                    print(f"  {d}/")

        print()
        print(f"[DRY-RUN] ファイル数: {len(files)}")
        print(f"[DRY-RUN] 合計サイズ: {_format_size(total_size)}")
        return 0

    # --- 実コピー ---
    print(f"プロジェクトルート: {PROJECT_ROOT}")
    print(f"出力先: {dest}")
    print(f"--include-data: {include_data}")
    print()

    copied_count = 0
    total_size = 0

    for src, rel in files:
        dst = dest / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied_count += 1
        total_size += dst.stat().st_size

    # data/ 空ディレクトリ作成（--include-data なしの場合）
    if not include_data:
        empty_dirs = _collect_data_empty_dirs()
        for d in empty_dirs:
            (dest / d).mkdir(parents=True, exist_ok=True)

    # --- 秘密情報チェック ---
    secrets = _check_secrets(dest)

    # --- サマリ出力 ---
    print(f"コピー完了: {copied_count} ファイル ({_format_size(total_size)})")
    print(f"出力先: {dest}")

    if not include_data:
        empty_dirs = _collect_data_empty_dirs()
        if empty_dirs:
            print(f"空ディレクトリ作成: {len(empty_dirs)} ディレクトリ (data/)")

    if secrets:
        print()
        print("[WARNING] 秘密情報の可能性があるファイルが検出されました:")
        for s in secrets:
            print(f"  - {s}")
        print("これらのファイルを確認し、問題がある場合は手動で削除してください。")
        return 1

    print()
    print("[OK] 秘密情報チェック: 問題なし")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="引き継ぎパッケージ作成スクリプト（許可リスト方式）",
    )
    parser.add_argument(
        "dest",
        type=Path,
        help="出力先ディレクトリ（新規作成される）",
    )
    parser.add_argument(
        "--include-data",
        action="store_true",
        default=False,
        help="data/source/ と data/input/ の実データも含める（デフォルト: 空ディレクトリ構造のみ）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="コピーせずに対象ファイル一覧を表示",
    )

    args = parser.parse_args()
    sys.exit(run(args.dest, args.include_data, args.dry_run))


if __name__ == "__main__":
    main()
