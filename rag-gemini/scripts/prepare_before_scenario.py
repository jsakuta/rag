#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
変更前シナリオの前処理スクリプト

処理内容:
1. 「データ整理/変更前シナリオ」からExcelファイルを読み込み
2. 文字数列を削除（Lv1~Lv10, シナリオパスのみ残す）
3. ファイル名を変換して reference/scenario/ に配置

使用方法:
    python scripts/prepare_before_scenario.py [--dry-run]

オプション:
    --dry-run: 実際のファイル作成を行わず、処理内容を表示
"""

import os
import re
import sys
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

# プロジェクトルート
PROJECT_ROOT = Path(__file__).parent.parent

# 改定番号マッピング（丸数字 → revision_id）
REVISION_MAP = {
    "①": "01",
    "②": "02",
    "③": "03",
    "④": "04",
    "⑤": "05",
    "⑥": "06",
    "⑦": "07",
    "⑧": "08",
    "⑨": "09",
}

# ソースディレクトリ
SOURCE_DIR = PROJECT_ROOT / "データ整理" / "変更前シナリオ"

# 出力ディレクトリ
OUTPUT_DIR = PROJECT_ROOT / "reference" / "scenario"

# ファイル名パターン: ①変更前シナリオ_smile-bot.xlsx
FILE_PATTERN = re.compile(r"^([①-⑨])変更前シナリオ_([a-z]+)-bot\.xlsx$")


def find_source_files():
    """ソースファイルを検索"""
    files = []

    if not SOURCE_DIR.exists():
        print(f"エラー: ソースディレクトリが存在しません: {SOURCE_DIR}")
        return files

    for folder in SOURCE_DIR.iterdir():
        if not folder.is_dir():
            continue

        for file_path in folder.iterdir():
            if file_path.suffix != ".xlsx" or file_path.name.startswith("~$"):
                continue

            match = FILE_PATTERN.match(file_path.name)
            if match:
                revision_mark = match.group(1)  # ①, ②, etc.
                bot_name = match.group(2)  # smile, souzoku, etc.
                revision_id = REVISION_MAP.get(revision_mark, "00")

                files.append({
                    "path": file_path,
                    "revision_mark": revision_mark,
                    "revision_id": revision_id,
                    "bot_name": bot_name,
                })

    return files


def remove_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """不要な列を削除（文字数列、シナリオパス列）"""
    # 保持する列: Lv1~Lv10 のみ（シナリオパスは削除）
    keep_columns = []
    for col in df.columns:
        if col.startswith("Lv"):
            keep_columns.append(col)

    return df[keep_columns]


def generate_output_filename(revision_id: str, bot_name: str) -> str:
    """出力ファイル名を生成"""
    # 既存の命名規則に合わせる: {business}_シナリオデータ_{date}.xlsx
    today = datetime.now().strftime("%Y%m%d")
    return f"rev{revision_id}{bot_name}_シナリオデータ_{today}.xlsx"


def process_files(dry_run: bool = False):
    """ファイルを処理"""
    files = find_source_files()

    if not files:
        print("処理対象のファイルが見つかりません。")
        return

    print(f"検出されたファイル: {len(files)}件\n")

    # 出力ディレクトリの作成
    if not dry_run:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for info in sorted(files, key=lambda x: (x["revision_id"], x["bot_name"])):
        source_path = info["path"]
        output_filename = generate_output_filename(info["revision_id"], info["bot_name"])
        output_path = OUTPUT_DIR / output_filename

        print(f"処理中: {source_path.name}")
        print(f"  改定番号: {info['revision_mark']} (rev{info['revision_id']})")
        print(f"  ボット: {info['bot_name']}")
        print(f"  出力先: {output_path.name}")

        if dry_run:
            # ドライラン: 列情報を表示
            df = pd.read_excel(source_path)
            print(f"  元の列数: {len(df.columns)}")
            df_cleaned = remove_unnecessary_columns(df)
            print(f"  処理後の列数: {len(df_cleaned.columns)}")
            print(f"  保持する列: {list(df_cleaned.columns)}")
        else:
            # 実際の処理
            df = pd.read_excel(source_path)
            df_cleaned = remove_unnecessary_columns(df)
            df_cleaned.to_excel(output_path, index=False)
            print(f"  保存完了: {len(df_cleaned)}行")

        print()

    print("=" * 50)
    if dry_run:
        print(f"ドライラン完了: {len(files)}件のファイルが処理対象です")
        print("実際に処理するには --dry-run オプションを外して実行してください")
    else:
        print(f"処理完了: {len(files)}件のファイルを reference/scenario/ に配置しました")


def main():
    parser = argparse.ArgumentParser(
        description="変更前シナリオの前処理（文字数列削除・リネーム・配置）"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="実際のファイル作成を行わず、処理内容を表示"
    )
    args = parser.parse_args()

    print("=" * 50)
    print("変更前シナリオ 前処理スクリプト")
    print("=" * 50)
    print(f"ソース: {SOURCE_DIR}")
    print(f"出力先: {OUTPUT_DIR}")
    print(f"モード: {'ドライラン' if args.dry_run else '実行'}")
    print("=" * 50)
    print()

    process_files(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
