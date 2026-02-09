#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
事務改定前シナリオDB ソースファイル生成スクリプト

最新マージ版シナリオに各改定の「修正前」カテゴリファイルを適用し、
正確なDBソースファイルを生成する。

使用方法:
    python scripts/generate_db_source.py [オプション]

オプション:
    --dry-run      : 実際のファイル作成を行わず、処理内容を表示
    --validate     : 生成後の検証を実行
    --diff         : 既存ファイルとの差分を表示
    --revision X   : 特定の改定のみ処理（1-6）
    --backup       : 処理前にバックアップを作成
"""

import os
import sys
import argparse
import shutil
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

# プロジェクトルート
PROJECT_ROOT = Path(__file__).parent.parent

# ディレクトリ設定
MERGE_BASE_DIR = PROJECT_ROOT / "データ整理" / "最新_マージ版シナリオ"
REVISION_BASE_DIR = PROJECT_ROOT / "データ整理" / "事務改定前後_個別シナリオ"
OUTPUT_DIR = PROJECT_ROOT / "data" / "source" / "scenarios" / "revisions"
BACKUP_DIR = PROJECT_ROOT / "data" / "source" / "scenarios" / "backup"


@dataclass
class CategoryReplacement:
    """カテゴリ置換設定"""
    subfolder: Optional[str]  # 修正前ファイルのサブフォルダ（Noneの場合は直下）
    filename: str             # 修正前ファイル名
    lv1: str                  # 置換対象のLv1カテゴリ
    lv2: Optional[str] = None # 部分置換の場合のLv2（Noneで全体置換）


@dataclass
class RevisionConfig:
    """改定設定"""
    revision_id: str          # rev01, rev02, etc.
    folder_name: str          # 改定フォルダ名
    bot_name: str             # 出力ファイルのボット名
    base_file: str            # ベースファイル名
    replacements: list = field(default_factory=list)  # 置換設定リスト
    full_replace: bool = False  # True: 修正前ファイルでベース全体を置換
    add_sentakushi: bool = False  # True: 「選択肢の全体像」行を追加


# 改定設定
REVISION_CONFIGS = [
    # 改定①: スマイル機能変更
    RevisionConfig(
        revision_id="rev01",
        folder_name="①スマイル機能変更メンテ台帳20",
        bot_name="smile",
        base_file="マージ版シナリオ_smile-bot.xlsx",
        replacements=[
            CategoryReplacement(
                subfolder=None,
                filename="①スマイル機能変更_修正前_シナリオ_スマイルタブレット_諸届_20250718 (1).xlsx",
                lv1="諸届",
            ),
        ],
    ),
    # 改定②: 相続少額払い
    RevisionConfig(
        revision_id="rev02",
        folder_name="②相続少額払いメンテ台帳21",
        bot_name="souzoku",
        base_file="マージ版シナリオ_souzoku-bot.xlsx",
        replacements=[
            CategoryReplacement(
                subfolder=None,
                filename="②少額解約変更前_シナリオ_相続_預金_少額払い・簡易払い_20250213 (1).xlsx",
                lv1="預金",
                lv2="少額払い・簡易払い",
            ),
        ],
    ),
    # 改定③smile: 保険証→資格確認証（スマイル）
    RevisionConfig(
        revision_id="rev03",
        folder_name="③保険証→資格確認証メンテ台帳25.26.27.28.29.30.35.36",
        bot_name="smile",
        base_file="マージ版シナリオ_smile-bot.xlsx",
        replacements=[
            CategoryReplacement(
                subfolder="スマイル_カード関連",
                filename="シナリオ_スマイルタブレット_カード関連_20250717.xlsx",
                lv1="カード関連",
            ),
            CategoryReplacement(
                subfolder="スマイル_取引時確認",
                filename="シナリオ_スマイルタブレット_取引時確認_20250718.xlsx",
                lv1="取引時確認",
            ),
        ],
    ),
    # 改定③torikaku: 取引時確認（全体置換）
    RevisionConfig(
        revision_id="rev03",
        folder_name="③保険証→資格確認証メンテ台帳25.26.27.28.29.30.35.36",
        bot_name="torikaku",
        base_file="マージ版シナリオ_torikaku-bot.xlsx",
        replacements=[
            CategoryReplacement(
                subfolder="取引時確認",
                filename="シナリオ_取引時確認_20250228.xlsx",
                lv1="_FULL_REPLACE_",  # 特殊マーカー
            ),
        ],
        full_replace=True,
        add_sentakushi=True,
    ),
    # 改定③souzoku: 相続預金（少額払い部分）
    RevisionConfig(
        revision_id="rev03",
        folder_name="③保険証→資格確認証メンテ台帳25.26.27.28.29.30.35.36",
        bot_name="souzoku",
        base_file="マージ版シナリオ_souzoku-bot.xlsx",
        replacements=[
            CategoryReplacement(
                subfolder="相続_預金",
                filename="シナリオ_相続_預金_少額払い・簡易払い_20250910.xlsx",
                lv1="預金",
                lv2="少額払い・簡易払い",
            ),
        ],
    ),
    # 改定③naibujimu: 内部事務（喪失、届出事項変更、預金）
    RevisionConfig(
        revision_id="rev03",
        folder_name="③保険証→資格確認証メンテ台帳25.26.27.28.29.30.35.36",
        bot_name="naibujimu",
        base_file="マージ版シナリオ_naibujimu-bot.xlsx",
        replacements=[
            CategoryReplacement(
                subfolder="内部事務_喪失",
                filename="シナリオ_内部事務_喪失_20250718.xlsx",
                lv1="喪失",
            ),
            CategoryReplacement(
                subfolder="内部事務_届出事項変更",
                filename="シナリオ_内部事務_届出事項変更_20250718.xlsx",
                lv1="届出事項変更",
            ),
            CategoryReplacement(
                subfolder="内部事務_預金",
                filename="シナリオ_内部事務_預金_普通預金・貯蓄預金_20250813.xlsx",
                lv1="預金",
                lv2="普通預金・貯蓄預金",
            ),
        ],
    ),
    # 改定④: 0円新規開設可能
    RevisionConfig(
        revision_id="rev04",
        folder_name="④難易度高_0円新規開設可能メンテ台帳37",
        bot_name="naibujimu",
        base_file="マージ版シナリオ_naibujimu-bot.xlsx",
        replacements=[
            CategoryReplacement(
                subfolder=None,
                filename="シナリオ_内部事務_預金_普通預金・貯蓄預金_20250813.xlsx",
                lv1="預金",
                lv2="普通預金・貯蓄預金",
            ),
        ],
    ),
    # 改定⑤: AML→GPLEX
    RevisionConfig(
        revision_id="rev05",
        folder_name="⑤AMLフィルター→GPLEXメンテ台帳41.42",
        bot_name="smile",
        base_file="マージ版シナリオ_smile-bot.xlsx",
        replacements=[
            CategoryReplacement(
                subfolder="喪失",
                filename="シナリオ_スマイルタブレット_喪失_20250731.xlsx",
                lv1="喪失",
            ),
            CategoryReplacement(
                subfolder="預金関連",
                filename="シナリオ_スマイルタブレット_預金関連_20250919.xlsx",
                lv1="預金関連",
            ),
        ],
    ),
    # 改定⑥: DC→MDC
    RevisionConfig(
        revision_id="rev06",
        folder_name="⑥DC→MDCメンテ台帳43.44.45",
        bot_name="smile",
        base_file="マージ版シナリオ_smile-bot.xlsx",
        replacements=[
            CategoryReplacement(
                subfolder="カード関連",
                filename="シナリオ_スマイルタブレット_カード関連_20251203.xlsx",
                lv1="カード関連",
            ),
            CategoryReplacement(
                subfolder="喪失",
                filename="シナリオ_スマイルタブレット_喪失_20250731.xlsx",
                lv1="喪失",
            ),
            CategoryReplacement(
                subfolder="諸届",
                filename="シナリオ_スマイルタブレット_諸届_20250919.xlsx",
                lv1="諸届",
            ),
        ],
    ),
]

# 期待行数（検証用）
EXPECTED_ROWS = {
    "rev01smile": 555,
    "rev02souzoku": 262,
    "rev03smile": 555,
    "rev03torikaku": 105,
    "rev03souzoku": 269,
    "rev03naibujimu": 1384,
    "rev04naibujimu": 1384,
    "rev05smile": 555,
    "rev06smile": 555,
}


def load_excel(path: Path) -> pd.DataFrame:
    """Excelファイルを読み込み"""
    df = pd.read_excel(path)
    # Lv列のみ抽出
    lv_cols = [col for col in df.columns if col.startswith("Lv")]
    return df[lv_cols]


def get_category_order(df: pd.DataFrame) -> list:
    """Lv1カテゴリの出現順序を取得"""
    seen = []
    for val in df["Lv1"].dropna():
        if val not in seen:
            seen.append(val)
    return seen


def replace_category_full(base_df: pd.DataFrame, replacement_df: pd.DataFrame, lv1: str) -> pd.DataFrame:
    """Lv1カテゴリを全体置換"""
    # カテゴリ順序を取得
    category_order = get_category_order(base_df)

    if lv1 not in category_order:
        print(f"  警告: カテゴリ '{lv1}' がベースに存在しません")
        return base_df

    # 該当カテゴリ以外を保持
    parts = []
    current_cat = None
    start_idx = 0

    for i, row in base_df.iterrows():
        cat = row["Lv1"]
        if pd.notna(cat) and cat != current_cat:
            if current_cat is not None:
                if current_cat != lv1:
                    parts.append(base_df.iloc[start_idx:i])
                else:
                    # 置換対象カテゴリの位置を記録
                    insert_pos = len(parts)
            current_cat = cat
            start_idx = i

    # 最後のセグメント
    if current_cat is not None:
        if current_cat != lv1:
            parts.append(base_df.iloc[start_idx:])
        else:
            insert_pos = len(parts)

    # 置換カテゴリを挿入
    parts.insert(insert_pos, replacement_df)

    return pd.concat(parts, ignore_index=True)


def replace_category_partial(base_df: pd.DataFrame, replacement_df: pd.DataFrame, lv1: str, lv2: str) -> pd.DataFrame:
    """Lv1+Lv2で部分置換"""
    mask = (base_df["Lv1"] == lv1) & (base_df["Lv2"] == lv2)

    if not mask.any():
        print(f"  警告: カテゴリ '{lv1}/{lv2}' がベースに存在しません")
        return base_df

    indices = base_df[mask].index.tolist()
    before = base_df.iloc[:indices[0]]
    after = base_df.iloc[indices[-1] + 1:]

    return pd.concat([before, replacement_df, after], ignore_index=True)


def get_replacement_path(config: RevisionConfig, repl: CategoryReplacement) -> Path:
    """置換ファイルのパスを取得"""
    base = REVISION_BASE_DIR / config.folder_name / "修正前"
    if repl.subfolder:
        return base / repl.subfolder / repl.filename
    return base / repl.filename


def process_revision(config: RevisionConfig, dry_run: bool = False) -> dict:
    """改定を処理"""
    result = {
        "config": config,
        "success": False,
        "rows": 0,
        "message": "",
    }

    # ベースファイルのパス
    base_path = MERGE_BASE_DIR / config.base_file
    if not base_path.exists():
        result["message"] = f"ベースファイルが見つかりません: {base_path}"
        return result

    # ベースファイル読み込み
    print(f"  ベースファイル: {config.base_file}")
    base_df = load_excel(base_path)
    print(f"    行数: {len(base_df)}")

    # 全体置換モード
    if config.full_replace:
        repl = config.replacements[0]
        repl_path = get_replacement_path(config, repl)

        if not repl_path.exists():
            result["message"] = f"修正前ファイルが見つかりません: {repl_path}"
            return result

        print(f"  全体置換: {repl.filename}")
        result_df = load_excel(repl_path)
        print(f"    行数: {len(result_df)}")

        if config.add_sentakushi:
            sentakushi_row = base_df[base_df["Lv1"].str.contains("選択肢の全体像", na=False)]
            if not sentakushi_row.empty:
                result_df = pd.concat([result_df, sentakushi_row], ignore_index=True)
                print("    「選択肢の全体像」行を追加")
    else:
        result_df = base_df.copy()

        for repl in config.replacements:
            repl_path = get_replacement_path(config, repl)

            if not repl_path.exists():
                result["message"] = f"修正前ファイルが見つかりません: {repl_path}"
                return result

            repl_df = load_excel(repl_path)
            category_label = f"{repl.lv1}/{repl.lv2}" if repl.lv2 else repl.lv1
            print(f"  置換: {category_label}")
            print(f"    ファイル: {repl.filename} ({len(repl_df)}行)")

            if repl.lv2:
                result_df = replace_category_partial(result_df, repl_df, repl.lv1, repl.lv2)
            else:
                result_df = replace_category_full(result_df, repl_df, repl.lv1)

    # 空行を削除
    before_count = len(result_df)
    result_df = result_df[result_df["Lv1"].notna()]
    after_count = len(result_df)
    if before_count != after_count:
        print(f"  空行削除: {before_count - after_count}行")

    result["rows"] = len(result_df)
    result["df"] = result_df
    result["success"] = True

    return result


def save_result(result: dict, dry_run: bool = False) -> Optional[Path]:
    """結果を保存"""
    config = result["config"]
    output_name = f"{config.revision_id}{config.bot_name}_シナリオデータ_{datetime.now().strftime('%Y%m%d')}.xlsx"
    output_path = OUTPUT_DIR / output_name

    if not dry_run:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        result["df"].to_excel(output_path, index=False)

    return output_path


def validate_result(result: dict) -> bool:
    """結果を検証"""
    config = result["config"]
    key = f"{config.revision_id}{config.bot_name}"
    expected = EXPECTED_ROWS.get(key)

    if expected is None:
        print(f"  検証: 期待行数が定義されていません")
        return True

    actual = result["rows"]
    if actual == expected:
        print(f"  検証: OK ({actual}行)")
        return True
    else:
        print(f"  検証: NG (期待: {expected}行, 実際: {actual}行)")
        return False


def create_backup():
    """バックアップを作成"""
    if not OUTPUT_DIR.exists():
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR.parent / f"scenario_backup_{timestamp}"
    shutil.copytree(OUTPUT_DIR, backup_path)
    print(f"バックアップ作成: {backup_path}")


def main():
    parser = argparse.ArgumentParser(description="事務改定前シナリオDB ソースファイル生成")
    parser.add_argument("--dry-run", action="store_true", help="実際のファイル作成を行わない")
    parser.add_argument("--validate", action="store_true", help="生成後の検証を実行")
    parser.add_argument("--diff", action="store_true", help="既存ファイルとの差分を表示")
    parser.add_argument("--revision", type=int, choices=[1, 2, 3, 4, 5, 6], help="特定の改定のみ処理")
    parser.add_argument("--backup", action="store_true", help="処理前にバックアップを作成")
    args = parser.parse_args()

    print("=" * 60)
    print("事務改定前シナリオDB ソースファイル生成")
    print("=" * 60)
    print(f"モード: {'ドライラン' if args.dry_run else '実行'}")
    if args.revision:
        print(f"対象改定: {args.revision}")
    print("=" * 60)
    print()

    # バックアップ
    if args.backup and not args.dry_run:
        create_backup()
        print()

    # 処理対象のフィルタリング
    configs = REVISION_CONFIGS
    if args.revision:
        rev_id = f"rev{args.revision:02d}"
        configs = [c for c in configs if c.revision_id == rev_id]

    # 処理
    results = []
    all_success = True

    for config in configs:
        print(f"[{config.revision_id}{config.bot_name}] 処理中...")
        result = process_revision(config, args.dry_run)
        results.append(result)

        if result["success"]:
            output_path = save_result(result, args.dry_run)
            print(f"  出力: {output_path.name} ({result['rows']}行)")

            if args.validate:
                if not validate_result(result):
                    all_success = False
        else:
            print(f"  エラー: {result['message']}")
            all_success = False

        print()

    # サマリー
    print("=" * 60)
    print("サマリー")
    print("=" * 60)
    for result in results:
        config = result["config"]
        status = "✓" if result["success"] else "✗"
        print(f"  {status} {config.revision_id}{config.bot_name}: {result['rows']}行")

    if args.dry_run:
        print()
        print("ドライランモードです。実際にファイルを作成するには --dry-run を外してください。")

    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
