"""
正解ID対応表生成スクリプト

事務改定内容フォルダのmdファイルと事務改定差分.mdから
正解IDを抽出し、multi_stage_input.xlsxを生成する。

正解ID形式: {ボット名}_{Excel行番号}
例: smile-bot_129, naibujimu-bot_96
"""

import re
import os
from pathlib import Path
import pandas as pd

# プロジェクトルートからの相対パス
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "データ整理"
OUTPUT_DIR = PROJECT_ROOT / "input"


def main():
    """メイン処理"""
    results = []

    # 事務改定内容フォルダのmdファイルを処理
    revision_dir = DATA_DIR / "事務改定内容"

    if not revision_dir.exists():
        print(f"エラー: {revision_dir} が存在しません")
        return

    for md_file in sorted(revision_dir.glob("*.md")):
        # ①②③...の番号を取得
        prefix = md_file.stem[0]  # "①", "②" など

        print(f"処理中: {md_file.name}")

        # 対応する事務改定差分.mdを探す
        diff_md = find_diff_md(prefix)
        if not diff_md:
            print(f"  警告: {prefix}事務改定差分.md が見つかりません")
            continue

        # 改定内容を読み込み
        revision_content = md_file.read_text(encoding='utf-8').strip()

        # 事務改定差分.mdから正解IDを抽出
        correct_ids = extract_correct_ids(diff_md)

        if not correct_ids:
            print(f"  警告: 正解IDが見つかりません")
            continue

        print(f"  正解ID: {correct_ids}")

        results.append({
            '番号': prefix,
            '改定内容': revision_content,
            '正解ID': ', '.join(correct_ids)
        })

    if not results:
        print("エラー: 処理対象が見つかりません")
        return

    # 出力ディレクトリを作成
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Excel出力
    output_path = OUTPUT_DIR / "multi_stage_input.xlsx"
    df = pd.DataFrame(results)
    df.to_excel(output_path, index=False)
    print(f"\n出力完了: {output_path}")
    print(f"件数: {len(results)}件")

    # 結果を表示
    print("\n--- 生成結果 ---")
    for row in results:
        print(f"[{row['番号']}] {row['改定内容'][:50]}...")
        print(f"    正解ID: {row['正解ID']}")


def find_diff_md(prefix: str) -> Path | None:
    """プレフィックスに対応する事務改定差分.mdを検索"""
    for folder in DATA_DIR.iterdir():
        if folder.is_dir() and folder.name.startswith(prefix):
            diff_file = folder / f"{prefix}事務改定差分.md"
            if diff_file.exists():
                return diff_file
    return None


def extract_correct_ids(diff_md: Path) -> list[str]:
    """
    事務改定差分.mdから正解IDを抽出

    対応フォーマット:
    - ## smile-bot → ボット名
    - 行番号: 129, 185 → Excel行番号リスト
    - **カテゴリ内行X** (Excel行Y) → Excel行番号（変更箇所セクション）
    """
    content = diff_md.read_text(encoding='utf-8')

    correct_ids = []
    current_bot = None

    for line in content.split('\n'):
        # ボット名セクション: ## smile-bot
        if line.startswith('## ') and '-bot' in line:
            current_bot = line[3:].strip()
            continue

        if not current_bot:
            continue

        # 行番号リスト形式: - 行番号: 129, 185
        # または: 行番号: 129, 185
        if '行番号:' in line:
            # コロンの後の部分を取得
            match = re.search(r'行番号:\s*(.+)', line)
            if match:
                numbers_str = match.group(1).strip()
                numbers = parse_row_numbers(numbers_str)
                for num in numbers:
                    correct_id = f"{current_bot}_{num}"
                    if correct_id not in correct_ids:
                        correct_ids.append(correct_id)

        # 変更箇所セクション形式: **カテゴリ内行26** (Excel行129)
        excel_match = re.search(r'\(Excel行(\d+)\)', line)
        if excel_match:
            num = int(excel_match.group(1))
            correct_id = f"{current_bot}_{num}"
            if correct_id not in correct_ids:
                correct_ids.append(correct_id)

    return correct_ids


def parse_row_numbers(text: str) -> list[int]:
    """
    行番号文字列をパース

    対応フォーマット:
    - カンマ/読点区切り: 1、5、12、13 または 129, 185
    - 範囲指定: 18～25（全角チルダ）または 18-25
    - 混在: 54～62、64
    - 省略表現（スキップ）: 他20行
    """
    # 全角→半角変換
    text = text.translate(str.maketrans(
        '０１２３４５６７８９、～',
        '0123456789,-'
    ))

    numbers = []

    # カンマで分割
    parts = [p.strip() for p in text.split(',')]

    for part in parts:
        # "他X行"のような省略表現はスキップ
        if '他' in part or not part:
            continue

        # 範囲指定: 18-25
        if '-' in part:
            range_match = re.match(r'(\d+)\s*-\s*(\d+)', part)
            if range_match:
                start = int(range_match.group(1))
                end = int(range_match.group(2))
                numbers.extend(range(start, end + 1))
            continue

        # 単一の数値
        if part.isdigit():
            numbers.append(int(part))

    return numbers


if __name__ == "__main__":
    main()
