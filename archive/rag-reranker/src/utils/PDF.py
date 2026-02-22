import os
import pandas as pd
import pymupdf4llm

def extract_text_from_pdf(file_path: str) -> str:
    """PDFファイルからテキストを抽出する"""
    md_text = pymupdf4llm.to_markdown(file_path, page_chunks=False)
    if isinstance(md_text, list):
        md_text = "\n".join(md_text)
    return md_text

def process_pdfs(input_dir: str, output_excel: str):
    """PDFファイルを処理してExcelに出力する"""
    data = []
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            if filename.lower().endswith('.pdf'):
                # PDFファイルからテキストを抽出
                text = extract_text_from_pdf(file_path)
                # データを追加
                data.append([filename, text])
                # PDFから読み取った内容を表示
                print(f"\n===== {filename} の内容 =====")
                print(text)
                print("================================\n")
    
    # データフレームを作成してExcelに出力
    df = pd.DataFrame(data, columns=['ファイル名', '内容'])
    df.to_excel(output_excel, index=False)
    print(f"処理が完了しました。結果は '{output_excel}' に保存されています。")

# 使用例
process_pdfs('manuals/保証', 'output_pdfs.xlsx')