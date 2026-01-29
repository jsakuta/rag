"""
正解ID対応表生成スクリプト

メンテナンス管理台帳から行番号を読み取り、
個別シナリオファイル（修正前）のテキスト内容とマージ版シナリオを
テキストマッチングして正しい行番号を特定する。

正解ID形式: {ボット名}_{マージ版Excel行番号}
例: smile-bot_130, naibujimu-bot_641
"""

import re
from pathlib import Path
import pandas as pd

# プロジェクトルートからの相対パス
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "データ整理"
OUTPUT_DIR = PROJECT_ROOT / "input"

# ボット名の変換
BOT_MAP = {
    'スマイル': 'smile-bot',
    '内部事務': 'naibujimu-bot',
    '相続': 'souzoku-bot',
    '取引時確認': 'torikaku-bot'
}

# 事務改定の定義
# row: メンテナンス管理台帳の行番号（header=3でのindex）
# merge: マージ版シナリオファイル名
# individual_folder: 事務改定フォルダ名
# individual_subfolder: 修正前フォルダ内のサブフォルダ名
# individual_pattern: 個別シナリオファイル名のパターン
REVISIONS = {
    '①': [
        {
            'row': 19,
            'merge': '①マージ版シナリオ_smile-bot.xlsx',
            'individual_folder': '①スマイル機能変更メンテ台帳20',
            'individual_subfolder': '',
            'individual_pattern': '*諸届*.xlsx',
        },
    ],
    '②': [
        {
            'row': 20,
            'merge': '②マージ版シナリオ_souzoku-bot.xlsx',
            'individual_folder': '②相続少額払いメンテ台帳21',
            'individual_subfolder': '',
            'individual_pattern': '*少額*.xlsx',
        },
    ],
    '③': [
        {
            'row': 24,  # 台帳25: 内部事務/喪失
            'merge': '③マージ版シナリオ_naibujimu-bot.xlsx',
            'individual_folder': '③保険証→資格確認証メンテ台帳25.26.27.28.29.30.35.36',
            'individual_subfolder': '内部事務_喪失',
            'individual_pattern': '*喪失*.xlsx',
        },
        {
            'row': 25,  # 台帳26: 内部事務/届出事項変更
            'merge': '③マージ版シナリオ_naibujimu-bot.xlsx',
            'individual_folder': '③保険証→資格確認証メンテ台帳25.26.27.28.29.30.35.36',
            'individual_subfolder': '内部事務_届出事項変更',
            'individual_pattern': '*届出事項変更*.xlsx',
        },
        {
            'row': 26,  # 台帳27: 内部事務/預金/普通預金・貯蓄預金
            'merge': '③マージ版シナリオ_naibujimu-bot.xlsx',
            'individual_folder': '③保険証→資格確認証メンテ台帳25.26.27.28.29.30.35.36',
            'individual_subfolder': '内部事務_預金',
            'individual_pattern': '*普通預金・貯蓄預金*.xlsx',
        },
        {
            'row': 28,  # 台帳29: 相続/預金/少額払い・簡易払い
            'merge': '③マージ版シナリオ_souzoku-bot.xlsx',
            'individual_folder': '③保険証→資格確認証メンテ台帳25.26.27.28.29.30.35.36',
            'individual_subfolder': '相続_預金',
            'individual_pattern': '*少額払い*.xlsx',
        },
        {
            'row': 29,  # 台帳30: 取引時確認
            'merge': '③マージ版シナリオ_torikaku-bot.xlsx',
            'individual_folder': '③保険証→資格確認証メンテ台帳25.26.27.28.29.30.35.36',
            'individual_subfolder': '取引時確認',
            'individual_pattern': 'シナリオ_取引時確認*.xlsx',
        },
        {
            'row': 34,  # 台帳35: スマイル/取引時確認
            'merge': '③マージ版シナリオ_smile-bot.xlsx',
            'individual_folder': '③保険証→資格確認証メンテ台帳25.26.27.28.29.30.35.36',
            'individual_subfolder': 'スマイル_取引時確認',
            'individual_pattern': '*取引時確認*.xlsx',
        },
        {
            'row': 35,  # 台帳36: スマイル/カード関連
            'merge': '③マージ版シナリオ_smile-bot.xlsx',
            'individual_folder': '③保険証→資格確認証メンテ台帳25.26.27.28.29.30.35.36',
            'individual_subfolder': 'スマイル_カード関連',
            'individual_pattern': '*カード関連*.xlsx',
        },
    ],
    '④': [
        {
            'row': 36,  # 台帳37: 内部事務/預金/普通預金・貯蓄預金
            'merge': '④マージ版シナリオ_naibujimu-bot.xlsx',
            'individual_folder': '④難易度高_0円新規開設可能メンテ台帳37',
            'individual_subfolder': '',
            'individual_pattern': '*普通預金・貯蓄預金*.xlsx',
        },
    ],
    '⑤': [
        {
            'row': 40,  # 台帳41: スマイル/預金関連
            'merge': '⑤マージ版シナリオ_smile-bot.xlsx',
            'individual_folder': '⑤AMLフィルター→GPLEXメンテ台帳41.42',
            'individual_subfolder': '預金関連',
            'individual_pattern': '*預金関連*.xlsx',
        },
        {
            'row': 41,  # 台帳42: スマイル/喪失
            'merge': '⑤マージ版シナリオ_smile-bot.xlsx',
            'individual_folder': '⑤AMLフィルター→GPLEXメンテ台帳41.42',
            'individual_subfolder': '喪失',
            'individual_pattern': '*喪失*.xlsx',
        },
    ],
    '⑥': [
        {
            'row': 42,  # 台帳43: スマイル/諸届
            'merge': '⑥マージ版シナリオ_smile-bot.xlsx',
            'individual_folder': '⑥DC→MDCメンテ台帳43.44.45',
            'individual_subfolder': '諸届',
            'individual_pattern': '*諸届*.xlsx',
        },
        {
            'row': 43,  # 台帳44: スマイル/喪失
            'merge': '⑥マージ版シナリオ_smile-bot.xlsx',
            'individual_folder': '⑥DC→MDCメンテ台帳43.44.45',
            'individual_subfolder': '喪失',
            'individual_pattern': '*喪失*.xlsx',
        },
        {
            'row': 44,  # 台帳45: スマイル/カード関連
            'merge': '⑥マージ版シナリオ_smile-bot.xlsx',
            'individual_folder': '⑥DC→MDCメンテ台帳43.44.45',
            'individual_subfolder': 'カード関連',
            'individual_pattern': '*カード関連*.xlsx',
        },
    ],
}

# キャッシュ
_merge_cache = {}
_individual_cache = {}


def main():
    """メイン処理"""
    # メンテナンス管理台帳を読み込み
    ledger_path = DATA_DIR / "シナリオボットメンテナンス管理台帳.xlsx"
    ledger = pd.read_excel(ledger_path, sheet_name='メンテナンス管理台帳', header=3)
    ledger.columns = ['idx', 'No', 'ステータス', '受付日', '区分', 'ボット名', '大分類', '変更内容',
                      '修正種別', 'WF通番', '事務改定実施日', '担当部署', '起案者', '対象シナリオファイル',
                      '作業中シナリオファイル', '提出予定日', '更新予定日時', '精査担当', '更新担当']

    results = []

    # 事務改定内容フォルダのmdファイルを処理
    revision_dir = DATA_DIR / "事務改定内容"

    if not revision_dir.exists():
        print(f"エラー: {revision_dir} が存在しません")
        return

    for md_file in sorted(revision_dir.glob("*.md")):
        prefix = md_file.stem[0]

        print(f"処理中: {md_file.name}")

        if prefix not in REVISIONS:
            print(f"  警告: {prefix} の定義がありません")
            continue

        revision_content = md_file.read_text(encoding='utf-8').strip()
        correct_ids = generate_correct_ids(ledger, prefix)

        if not correct_ids:
            print(f"  警告: 正解IDが見つかりません")
            continue

        print(f"  正解ID ({len(correct_ids)}件): {correct_ids[:5]}{'...' if len(correct_ids) > 5 else ''}")

        results.append({
            '番号': prefix,
            '改定内容': revision_content,
            '正解ID': ', '.join(correct_ids)
        })

    if not results:
        print("エラー: 処理対象が見つかりません")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "multi_stage_input.xlsx"
    df = pd.DataFrame(results)
    df.to_excel(output_path, index=False)
    print(f"\n出力完了: {output_path}")
    print(f"件数: {len(results)}件")

    print("\n--- 生成結果 ---")
    for row in results:
        print(f"[{row['番号']}] {row['改定内容'][:50]}...")
        print(f"    正解ID: {row['正解ID']}")


def generate_correct_ids(ledger: pd.DataFrame, prefix: str) -> list[str]:
    """正解IDを生成"""
    correct_ids = []

    for rev_def in REVISIONS[prefix]:
        row_idx = rev_def['row']
        merge_file = rev_def['merge']

        row = ledger.iloc[row_idx]
        bot_name = row['ボット名']
        content = str(row['変更内容'])

        cat_rows = parse_row_numbers(content)
        if not cat_rows:
            continue

        bot = BOT_MAP.get(bot_name, bot_name)

        # テキストマッチングで正解IDを取得
        merge_rows = find_rows_by_text_matching(
            rev_def['individual_folder'],
            rev_def.get('individual_subfolder', ''),
            rev_def['individual_pattern'],
            merge_file,
            cat_rows
        )

        for row_num in merge_rows:
            correct_id = f"{bot}_{row_num}"
            if correct_id not in correct_ids:
                correct_ids.append(correct_id)

    return correct_ids


def find_rows_by_text_matching(
    individual_folder: str,
    individual_subfolder: str,
    individual_pattern: str,
    merge_file: str,
    target_rows: list[int]
) -> list[int]:
    """テキストマッチングでマージ版の行番号を特定"""

    # 個別シナリオファイルを読み込み
    individual_dir = DATA_DIR / "事務改定前後_個別シナリオ" / individual_folder / "修正前"
    if individual_subfolder:
        individual_dir = individual_dir / individual_subfolder

    files = list(individual_dir.glob(individual_pattern))

    if not files:
        print(f"  警告: 個別シナリオファイルが見つかりません: {individual_dir / individual_pattern}")
        return []

    individual_file = files[0]
    cache_key = str(individual_file)

    if cache_key not in _individual_cache:
        _individual_cache[cache_key] = pd.read_excel(individual_file)
    df_individual = _individual_cache[cache_key]

    # マージ版シナリオを読み込み
    if merge_file not in _merge_cache:
        merge_path = DATA_DIR / "事務改定前_マージ版シナリオ" / merge_file
        if not merge_path.exists():
            print(f"  警告: マージ版シナリオが見つかりません: {merge_path}")
            return []
        _merge_cache[merge_file] = pd.read_excel(merge_path)
    df_merge = _merge_cache[merge_file]

    # テキストマッチング用の列
    key_cols = ['Lv1', 'Lv2', 'Lv3', 'Lv4', 'Lv5', 'Lv6', 'Lv7', 'Lv8', 'Lv9', 'Lv10']

    def make_key(row):
        """行からマッチングキーを作成"""
        parts = []
        for col in key_cols:
            val = row.get(col)
            if pd.notna(val):
                parts.append(str(val).strip())
        return '|'.join(parts)

    # マージ版のキーを事前計算
    merge_keys = {}
    for merge_idx, merge_row in df_merge.iterrows():
        key = make_key(merge_row)
        if key not in merge_keys:
            merge_keys[key] = []
        merge_keys[key].append(merge_idx + 2)  # Excel行番号

    # 各行をマッチング
    # 台帳の行番号の解釈は事務改定によって異なる：
    # - Excel行番号（ヘッダー含む）: idx = row_num - 2
    # - カテゴリ内行番号（1始まり）: idx = row_num - 1
    # 両方試してマッチする方を採用
    matched_rows = []
    for row_num in target_rows:
        found = False

        # パターン1: Excel行番号として解釈
        idx1 = row_num - 2
        if 0 <= idx1 < len(df_individual):
            key1 = make_key(df_individual.iloc[idx1])
            if key1 in merge_keys:
                matched_rows.append(merge_keys[key1][0])
                found = True
                continue

        # パターン2: カテゴリ内行番号として解釈
        idx2 = row_num - 1
        if 0 <= idx2 < len(df_individual):
            key2 = make_key(df_individual.iloc[idx2])
            if key2 in merge_keys:
                matched_rows.append(merge_keys[key2][0])
                found = True
                continue

        if not found:
            # 新規追加行など、修正前ファイルに存在しない行はスキップ
            pass

    return matched_rows


def parse_row_numbers(text: str) -> list[int]:
    """変更内容テキストから行番号を抽出"""
    if pd.isna(text):
        return []

    text = text.translate(str.maketrans(
        '０１２３４５６７８９、～：',
        '0123456789,-:'
    ))

    numbers = []
    for match in re.finditer(r'行番号[:\s]*([0-9,\-~\s]+)', text):
        nums_str = match.group(1)
        for part in re.split(r'[,\s]+', nums_str):
            part = part.strip()
            if not part:
                continue
            if '-' in part or '~' in part:
                m = re.match(r'(\d+)\s*[-~]\s*(\d+)', part)
                if m:
                    numbers.extend(range(int(m.group(1)), int(m.group(2)) + 1))
            elif part.isdigit():
                numbers.append(int(part))

    return sorted(set(numbers))


if __name__ == "__main__":
    main()
