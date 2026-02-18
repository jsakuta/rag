import os
import re
import pandas as pd
from typing import List
import pymupdf4llm

def is_indented_major_header(line: str) -> bool:
    """インデントされた見出しかどうかを判定"""
    stripped = line.lstrip()
    major_header_patterns = [
        r"^\d+[．\.](?!.*[（\(]注[）\)]).*",  # 例: "1．" または "1." ただし（注）を含まない
        r"^第\d+章\s+.*",          # 例: "第1章 "
        r"^第\d+節\s+.*",          # 例: "第1節 "
    ]
    return any(re.match(pattern, stripped) for pattern in major_header_patterns)

def split_by_hierarchical_headers(text: str) -> List[str]:
    """テキストを階層的な見出しで分割する"""
    sub_header_patterns = [
        r"^\d+[-‐－‑‒–—―]\d+\s+.*", # 例: "1-1 "（全角・半角のハイフンを許可）
        r"^[A-Z][．\.].*",         # 例: "A．" または "A."
        r"^\[\d+\].*",             # 例: "[1]"
        r"^\［\d+\］.*"           # 例: "［1］"（全角）
    ]

    lines = text.split("\n")
    chunks = []
    current_major_header = None
    current_sub_header = None
    current_content = []
    major_header_found = False
    in_note_section = False
    current_note_number = None
    last_number_in_note = None
    continuing_header = False  # 見出しの継続フラグ
    note_content = []  # 注釈セクションの内容を保持
    in_numbered_list = False  # 番号付きリスト内かどうかのフラグ
    note_section_indentation = None  # 注釈セクションのインデント量

    def save_current_chunk():
        if not current_content and not note_content:
            return
        
        # 注釈セクションがある場合はそれを優先
        if note_content:
            chunk_text = "\n".join(note_content).strip()
            note_content.clear()
        else:
            chunk_text = "\n".join(current_content).strip()
            
        headers = []
        if current_major_header:
            headers.append(current_major_header)
        if current_sub_header:
            headers.append(current_sub_header)
        full_text = "\n".join(headers + [chunk_text] if chunk_text else headers)
        if full_text.strip():
            chunks.append(full_text)
        current_content.clear()

    def get_line_indentation(line: str) -> int:
        """行のインデント量を取得する"""
        return len(line) - len(line.lstrip())

    def extract_number_from_note(text: str) -> int:
        """注釈番号を抽出する（より柔軟な形式に対応）"""
        # 全角数字を半角に変換
        zen_to_han = str.maketrans('０１２３４５６７８９', '0123456789')
        text = text.translate(zen_to_han)
        
        # (注)1. や （注）1. のパターンを検出
        note_pattern = r'[（\(]\s*注\s*[）\)]\s*(\d+)[\.\．]'
        match = re.search(note_pattern, text)
        if match:
            return int(match.group(1))
        
        # 単独の数字 + ．または. のパターンを検出（注釈セクション内の連番用）
        if in_note_section:
            num_pattern = r'^\s*(\d+)[\.\．]'
            match = re.search(num_pattern, text)
            if match:
                return int(match.group(1))
        
        return None

    def is_note_start(line: str) -> bool:
        """注釈セクションの開始をチェック"""
        cleaned_line = line.strip()
        # より柔軟な注釈開始パターン
        pattern = r'^[\s　]*[（\(]\s*注\s*[）\)]\s*[0-9０-９]+\s*[．\.]'
        is_start = bool(re.match(pattern, cleaned_line))
        if is_start:
            print(f"\n=== 注釈セクション開始: {cleaned_line} ===")
            nonlocal note_section_indentation
            note_section_indentation = get_line_indentation(line)
        return is_start

    def is_in_note_section(line: str, indent: int) -> bool:
        """現在の行が注釈セクション内かどうかを判定"""
        if not in_note_section:
            return False
        # インデントベースの判定を追加
        current_indent = get_line_indentation(line)
        # インデントが同じか深い場合は同じセクション内と判断
        return current_indent >= note_section_indentation

    def is_sequential_number(line: str) -> bool:
        """連番かどうかをチェック"""
        if not in_note_section:
            return False
            
        number = extract_number_from_note(line)
        if number is None:
            return False
            
        # 最後の番号がない場合は、current_note_numberとの比較
        if last_number_in_note is None:
            return number == current_note_number + 1
            
        # 最後の番号がある場合は、その番号との比較
        return number == last_number_in_note + 1

    def should_exit_note_section(line: str) -> bool:
        """注釈セクションを抜けるべきかどうかを判定"""
        if not line.strip():
            return False

        # インデントベースの判定
        if not is_in_note_section(line, note_section_indentation):
            print(f"\n=== 注釈セクション終了 (インデント変更): {line.strip()} ===")
            return True

        # 番号付きリストの場合は継続
        if is_numbered_list_item(line):
            return False

        # 新しい見出しパターンの場合は終了
        if any(re.match(pattern, line.strip()) for pattern in sub_header_patterns):
            if not is_sequential_number(line):
                print(f"\n=== 注釈セクション終了 (見出し検出): {line.strip()} ===")
                return True

        return False

    def is_numbered_list_item(line: str) -> bool:
        """番号付きリストのアイテムかどうかをチェック"""
        stripped_line = line.strip()
        patterns = [
            r'^\d+[\.\．]',  # 1. or 1．
            r'^[①-⑳]',    # 丸数字
            r'^\([0-9]+\)', # (1)
            r'^（[0-9]+）', # （1）
        ]
        return any(re.match(pattern, stripped_line) for pattern in patterns)

    # ファイルの先頭を強制的に最初の見出しとして設定
    if lines:
        # 最初の行から```を除去
        first_line = lines[0].strip()
        if first_line == "```":
            if len(lines) > 1:
                current_major_header = lines[1].strip()
                lines = lines[1:]
        else:
            current_major_header = first_line
        major_header_found = True
        print(f"最初の見出しを '{current_major_header}' に設定しました。")

    for idx, line in enumerate(lines[1:], start=2):
        original_line = line
        stripped_line = line.strip()

        # ```だけの行はスキップ
        if stripped_line == "```":
            continue

        if is_note_start(line):
            save_current_chunk()
            in_note_section = True
            current_note_number = extract_number_from_note(line)
            last_number_in_note = current_note_number
            note_content = [original_line]
            continuing_header = False
            print(f"注釈番号を設定: {current_note_number}")
            continue

        if in_note_section:
            if should_exit_note_section(original_line):
                in_note_section = False
                current_note_number = None
                last_number_in_note = None
                note_section_indentation = None
                save_current_chunk()
                print("注釈番号をリセット")
                
                # 新しい見出しの処理
                if is_indented_major_header(line):
                    save_current_chunk()
                    current_major_header = stripped_line
                    continuing_header = False
                elif any(re.match(pattern, stripped_line) for pattern in sub_header_patterns):
                    save_current_chunk()
                    current_sub_header = stripped_line
                    continuing_header = False
                else:
                    current_content.append(original_line)
            else:
                number = extract_number_from_note(line)
                if number is not None:
                    if last_number_in_note is not None:
                        print(f"注釈番号を更新: {last_number_in_note} -> {number}")
                    last_number_in_note = number
                note_content.append(original_line)
        else:
            if continuing_header and not is_indented_major_header(line) and not any(re.match(pattern, stripped_line) for pattern in sub_header_patterns):
                # 見出しの継続処理
                if current_major_header:
                    current_major_header += " " + stripped_line
                continue

            if is_indented_major_header(line):
                save_current_chunk()
                current_major_header = stripped_line
                current_sub_header = None
                continuing_header = False
            elif any(re.match(pattern, stripped_line) for pattern in sub_header_patterns):
                save_current_chunk()
                current_sub_header = stripped_line
                continuing_header = False
            else:
                current_content.append(original_line)

    # 最後のチャンクを保存する前に、最後の```行を除去
    if current_content and current_content[-1].strip() == "```":
        current_content.pop()
    if note_content and note_content[-1].strip() == "```":
        note_content.pop()
    
    save_current_chunk()
    return chunks

def process_manuals(input_dir: str, output_excel: str):
    """manuals内のファイルを処理してExcelに出力する"""
    data = []
    max_chunks = 0  # 最大チャンク数を記録
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            if filename.lower().endswith('.pdf'):
                # PDFファイルを読み込みマークダウンに変換
                md_text = pymupdf4llm.to_markdown(file_path, page_chunks=False)
                if isinstance(md_text, list):
                    md_text = "\n".join(md_text)

                # PDFから読み取った内容を表示
                print(f"\n===== {filename} の内容 =====")
                print(md_text)
                print("================================\n")

                # チャンク分割
                chunk_texts = split_by_hierarchical_headers(md_text)
                # データを追加
                row = [filename] + chunk_texts
                data.append(row)
                if len(chunk_texts) > max_chunks:
                    max_chunks = len(chunk_texts)
                    
    # カラム名を作成
    columns = ['ファイル名'] + [f'チャンク{i+1}' for i in range(max_chunks)]
    # チャンク数が少ない行を空文字で埋める
    padded_data = []
    for row in data:
        padded_row = row + [''] * (max_chunks - (len(row) - 1))
        padded_data.append(padded_row)
    # データフレームを作成してExcelに出力
    df = pd.DataFrame(padded_data, columns=columns)
    df.to_excel(output_excel, index=False)
    print(f"処理が完了しました。結果は '{output_excel}' に保存されています。")

if __name__ == "__main__":
    process_manuals('manuals', 'output_chunks.xlsx')