#!/usr/bin/env python3
"""
Markdown → Word変換スクリプト(設計書デザインポリシー準拠版 v8)

修正点:
- XMLエスケープ(&→&amp;)を確実に実行
- 表紙: プロジェクト名行を削除、紺線を上下2本、メタ情報は表ではなくテキスト
- 目次を確実に生成
- MDメタ情報を表紙に使用し本文から削除
- 斜体を完全削除(スタイル定義・本文両方)
- 見出しを全て黒色に統一
- 表を中央揃え
- 画像を中央揃え
- Heading1〜6の全見出しを太字に設定

使用方法:
    python convert.py <入力.md> <出力.docx>
"""

import sys
import os
import re
import subprocess
import shutil
import tempfile

OOXML_SCRIPTS = os.path.join(os.path.dirname(__file__), 'ooxml')

# pandoc executable (winget install location)
PANDOC = None
_winget_pandoc = os.path.expandvars(
    r'%LOCALAPPDATA%\Microsoft\WinGet\Packages'
    r'\JohnMacFarlane.Pandoc_Microsoft.Winget.Source_8wekyb3d8bbwe'
    r'\pandoc-3.9\pandoc.exe')
if os.path.exists(_winget_pandoc):
    PANDOC = _winget_pandoc
else:
    PANDOC = shutil.which('pandoc') or 'pandoc'

# 色定義
NAVY_COLOR = '2B5292'
LINK_COLOR = '0563C1'
BLACK_COLOR = '000000'
WHITE_COLOR = 'FFFFFF'
GRAY_COLOR = '666666'

# フォントサイズ(half-points: 1pt = 2 half-points)
SIZE_DEFAULT = '21'      # 10.5pt
SIZE_TITLE = '44'        # 22pt
SIZE_H1 = '32'           # 16pt
SIZE_H2 = '28'           # 14pt
SIZE_H3 = '24'           # 12pt
SIZE_H4 = '22'           # 11pt
SIZE_HEADER_FOOTER = '18'  # 9pt
SIZE_META = '21'         # 10.5pt

MEIRYO_FONTS = '<w:rFonts w:ascii="Meiryo UI" w:hAnsi="Meiryo UI" w:eastAsia="Meiryo UI" w:cs="Meiryo UI"/>'


def xml_escape(text):
    """XMLの特殊文字をエスケープ"""
    if not text:
        return text
    text = text.replace('&', '&amp;')
    text = text.replace('<', '&lt;')
    text = text.replace('>', '&gt;')
    text = text.replace('"', '&quot;')
    text = text.replace("'", '&apos;')
    return text


def parse_md_metadata(md_content):
    """MDからメタデータを抽出"""
    metadata = {
        'title': '',
        'version': '',
        'created_date': '',
        'updated_date': '',
        'author': ''
    }
    
    # BOMを削除
    md_content = md_content.lstrip('\ufeff')
    
    # H1タイトルを抽出
    h1_match = re.search(r'^#\s+(.+?)$', md_content, re.MULTILINE)
    if h1_match:
        title = h1_match.group(1).strip()
        # **を削除(太字マークダウン)
        title = re.sub(r'\*\*', '', title)
        metadata['title'] = title
    
    # メタ情報を抽出(**key**: value形式)
    patterns = {
        'version': r'\*\*(文書バージョン|版数)\*\*:\s*(.+)',
        'created_date': r'\*\*作成日\*\*:\s*(.+)',
        'updated_date': r'\*\*最終更新日\*\*:\s*(.+)',
        'author': r'\*\*作成者\*\*:\s*(.+)',
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, md_content, re.MULTILINE)
        if match:
            metadata[key] = match.group(match.lastindex).strip()
    
    return metadata


def preprocess_md(input_path, output_path):
    """MDを前処理: メタ情報と目次セクションを削除、NBSP変換"""
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # BOMを削除
    content = content.lstrip('\ufeff')
    
    # CRLF を LF に統一
    content = content.replace('\r\n', '\n')
    
    # NBSP(Non-Breaking Space、U+00A0)を通常のスペースに変換
    # これがないとpandocがリストの階層構造を正しく認識できない
    # ※正常なファイル(NBSPを含まない)には影響なし
    content = content.replace('\u00a0', ' ')
    
    # 見出しから**を全て削除(## **タイトル** → ## タイトル)
    # これがないと目次に太字が反映されてしまう
    # ※正常なファイル(見出しに**がない)には影響なし
    # 見出し行を検出して、その行から全ての**を削除
    def remove_bold_from_heading(match):
        heading_line = match.group(0)
        return heading_line.replace('**', '')
    
    content = re.sub(r'^#{1,6}\s+.+$', remove_bold_from_heading, content, flags=re.MULTILINE)
    
    # H1タイトル行を削除(表紙で使用するため)
    content = re.sub(r'^#\s+.+?\n', '', content, count=1)
    
    # メタ情報ブロックを削除(**key**: value形式の連続行)
    content = re.sub(
        r'(\*\*(文書バージョン|版数|文書ID|作成日|最終更新日|作成者|関連文書)\*\*:\s*.+\s*)+',
        '', content)
    
    # 手動で書かれた目次セクションを削除(**付きの見出しも対応)
    content = re.sub(r'##\s*\*?\*?目次\*?\*?\s*\n(.*?)(?=\n##\s)', '\n## ', content, flags=re.DOTALL)
    
    # 最初の---(水平線)も削除(表紙の区切り用)
    content = re.sub(r'^\s*---\s*$', '', content, count=1, flags=re.MULTILINE)
    
    # 先頭の空行を削除
    content = content.lstrip('\n')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return content


def convert_md_to_word(input_md, output_docx):
    """メイン変換処理"""
    # メタデータを抽出
    with open(input_md, 'r', encoding='utf-8') as f:
        original_content = f.read()
    
    metadata = parse_md_metadata(original_content)
    doc_title = metadata['title']
    version = metadata['version']
    created_date = metadata['created_date']
    author = metadata['author']
    
    work_dir = tempfile.mkdtemp(prefix='md_convert_')
    unpacked_dir = os.path.join(work_dir, 'unpacked')
    
    try:
        # 前処理
        processed_md = os.path.join(work_dir, 'processed.md')
        preprocess_md(input_md, processed_md)
        
        # pandocで変換(目次付き)
        temp_docx = os.path.join(work_dir, 'temp.docx')
        subprocess.run([
            PANDOC, processed_md, '-o', temp_docx,
            '--toc', '--toc-depth=4',
            '--resource-path', os.path.dirname(os.path.abspath(input_md)),
            '-f', 'markdown', '-t', 'docx'
        ], check=True)
        
        # unpack
        subprocess.run([
            'python', f'{OOXML_SCRIPTS}/unpack.py',
            temp_docx, unpacked_dir
        ], capture_output=True, check=True)
        
        # スタイル更新
        update_styles(unpacked_dir)
        
        # ヘッダー・フッター作成
        add_header(unpacked_dir, doc_title)
        add_footer(unpacked_dir)
        header_rid, footer_rid = update_relationships(unpacked_dir)
        
        # 文書内容更新
        update_document_content(unpacked_dir, header_rid, footer_rid,
                               doc_title, created_date, version, author)
        
        # 表スタイル適用
        apply_table_borders(unpacked_dir)
        apply_table_header_style(unpacked_dir)
        center_tables(unpacked_dir)
        
        # 画像を中央揃え
        center_images(unpacked_dir)
        
        # 斜体削除
        remove_italics(unpacked_dir)
        
        # フォントテーブル更新(Meiryo UI追加)
        update_font_table(unpacked_dir)
        
        # XMLエスケープ確認(最終処理)
        ensure_xml_escape(unpacked_dir)
        
        # pack
        subprocess.run([
            'python', f'{OOXML_SCRIPTS}/pack.py',
            unpacked_dir, output_docx
        ], check=True)
        
        print(f"変換完了: {output_docx}")
        
    finally:
        shutil.rmtree(work_dir)


def update_styles(unpacked_dir):
    """スタイル更新: フォント統一、見出し黒色化、斜体削除"""
    styles_path = os.path.join(unpacked_dir, 'word/styles.xml')
    with open(styles_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 全フォントをMeiryo UIに統一
    content = re.sub(r'<w:rFonts[^/]*/>', MEIRYO_FONTS, content)
    
    # スタイル定義から斜体を削除
    content = re.sub(r'<w:i/>', '', content)
    content = re.sub(r'<w:iCs/>', '', content)
    content = re.sub(r'<w:i\s*/>', '', content)
    content = re.sub(r'<w:iCs\s*/>', '', content)
    
    # デフォルトフォントサイズを10.5ptに
    content = re.sub(
        r'(<w:rPrDefault>\s*<w:rPr>.*?<w:sz w:val=\")(\d+)(\")',
        rf'\g<1>{SIZE_DEFAULT}\g<3>', content, flags=re.DOTALL)
    content = re.sub(
        r'(<w:rPrDefault>\s*<w:rPr>.*?<w:szCs w:val=\")(\d+)(\")',
        rf'\g<1>{SIZE_DEFAULT}\g<3>', content, flags=re.DOTALL)
    
    # 全てのテーマカラー参照を黒に置換
    content = re.sub(r'<w:color w:themeColor=\"[^\"]*\"[^/]*/>', 
                     f'<w:color w:val=\"{BLACK_COLOR}\"/>', content)
    content = re.sub(r'<w:color w:themeColor=\"[^\"]*\" w:themeShade=\"[^\"]*\" w:val=\"[^\"]*\"/>', 
                     f'<w:color w:val=\"{BLACK_COLOR}\"/>', content)
    
    # 見出しスタイルのサイズ設定と色を黒に統一、太字を追加
    style_configs = [
        ('Title', SIZE_TITLE, True),
        ('Heading1', SIZE_H1, True),
        ('Heading2', SIZE_H2, True),
        ('Heading3', SIZE_H3, True),
        ('Heading4', SIZE_H4, True),
        ('Heading5', SIZE_DEFAULT, True),  # 10.5pt、太字
        ('Heading6', SIZE_DEFAULT, True),  # 10.5pt、太字
    ]
    
    for style_id, size, bold in style_configs:
        # サイズ設定
        content = re.sub(
            rf'(<w:style[^>]*w:styleId=\"{style_id}\"[^>]*>.*?<w:sz w:val=\")(\d+)(\")',
            rf'\g<1>{size}\g<3>', content, flags=re.DOTALL)
        content = re.sub(
            rf'(<w:style[^>]*w:styleId=\"{style_id}\"[^>]*>.*?<w:szCs w:val=\")(\d+)(\")',
            rf'\g<1>{size}\g<3>', content, flags=re.DOTALL)
        
        # 色を黒に設定、太字を追加
        style_pattern = rf'(<w:style[^>]*w:styleId=\"{style_id}\"[^>]*>)(.*?)(</w:style>)'
        match = re.search(style_pattern, content, flags=re.DOTALL)
        if match:
            style_start, style_content, style_end = match.groups()
            # 既存の色設定を黒に置換
            style_content = re.sub(r'<w:color[^/]*/>', f'<w:color w:val=\"{BLACK_COLOR}\"/>', style_content)
            # 太字を追加(rPr内に<w:b/>がなければ追加)
            if bold and '<w:b/>' not in style_content:
                style_content = re.sub(r'(<w:rPr>)', r'\1<w:b/><w:bCs/>', style_content)
            content = content[:match.start()] + style_start + style_content + style_end + content[match.end():]
    
    # Hyperlinkスタイル
    hyperlink_style = f'''
  <w:style w:type=\"character\" w:styleId=\"Hyperlink\">
    <w:name w:val=\"Hyperlink\"/>
    <w:basedOn w:val=\"DefaultParagraphFont\"/>
    <w:uiPriority w:val=\"99\"/>
    <w:unhideWhenUsed/>
    <w:rPr>
      {MEIRYO_FONTS}
      <w:color w:val=\"{LINK_COLOR}\"/>
      <w:u w:val=\"single\"/>
    </w:rPr>
  </w:style>'''
    
    if 'w:styleId=\"Hyperlink\"' in content:
        content = re.sub(r'<w:style[^>]*w:styleId=\"Hyperlink\"[^>]*>.*?</w:style>',
                        hyperlink_style.strip(), content, flags=re.DOTALL)
    else:
        content = content.replace('</w:styles>', hyperlink_style + '\n</w:styles>')
    
    with open(styles_path, 'w', encoding='utf-8') as f:
        f.write(content)


def add_header(unpacked_dir, doc_title):
    """ヘッダー作成(右寄せ、グレー、文書タイトル)"""
    escaped_title = xml_escape(doc_title)
    header_xml = f'''<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>
<w:hdr xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\"
       xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\">
  <w:p>
    <w:pPr><w:jc w:val=\"right\"/></w:pPr>
    <w:r>
      <w:rPr>
        {MEIRYO_FONTS}
        <w:color w:val=\"{GRAY_COLOR}\"/>
        <w:sz w:val=\"{SIZE_HEADER_FOOTER}\"/>
        <w:szCs w:val=\"{SIZE_HEADER_FOOTER}\"/>
      </w:rPr>
      <w:t>{escaped_title}</w:t>
    </w:r>
  </w:p>
</w:hdr>'''
    with open(os.path.join(unpacked_dir, 'word/header1.xml'), 'w', encoding='utf-8') as f:
        f.write(header_xml)


def add_footer(unpacked_dir):
    """フッター作成(中央、ページ番号「- N -」形式)"""
    footer_xml = f'''<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>
<w:ftr xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\"
       xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\">
  <w:p>
    <w:pPr><w:jc w:val=\"center\"/></w:pPr>
    <w:r>
      <w:rPr>{MEIRYO_FONTS}<w:sz w:val=\"{SIZE_HEADER_FOOTER}\"/><w:szCs w:val=\"{SIZE_HEADER_FOOTER}\"/></w:rPr>
      <w:t xml:space=\"preserve\">- </w:t>
    </w:r>
    <w:r>
      <w:rPr>{MEIRYO_FONTS}<w:sz w:val=\"{SIZE_HEADER_FOOTER}\"/><w:szCs w:val=\"{SIZE_HEADER_FOOTER}\"/></w:rPr>
      <w:fldChar w:fldCharType=\"begin\"/>
    </w:r>
    <w:r>
      <w:rPr>{MEIRYO_FONTS}<w:sz w:val=\"{SIZE_HEADER_FOOTER}\"/><w:szCs w:val=\"{SIZE_HEADER_FOOTER}\"/></w:rPr>
      <w:instrText>PAGE</w:instrText>
    </w:r>
    <w:r>
      <w:rPr>{MEIRYO_FONTS}<w:sz w:val=\"{SIZE_HEADER_FOOTER}\"/><w:szCs w:val=\"{SIZE_HEADER_FOOTER}\"/></w:rPr>
      <w:fldChar w:fldCharType=\"separate\"/>
    </w:r>
    <w:r>
      <w:rPr>{MEIRYO_FONTS}<w:sz w:val=\"{SIZE_HEADER_FOOTER}\"/><w:szCs w:val=\"{SIZE_HEADER_FOOTER}\"/></w:rPr>
      <w:t>1</w:t>
    </w:r>
    <w:r>
      <w:rPr>{MEIRYO_FONTS}<w:sz w:val=\"{SIZE_HEADER_FOOTER}\"/><w:szCs w:val=\"{SIZE_HEADER_FOOTER}\"/></w:rPr>
      <w:fldChar w:fldCharType=\"end\"/>
    </w:r>
    <w:r>
      <w:rPr>{MEIRYO_FONTS}<w:sz w:val=\"{SIZE_HEADER_FOOTER}\"/><w:szCs w:val=\"{SIZE_HEADER_FOOTER}\"/></w:rPr>
      <w:t xml:space=\"preserve\"> -</w:t>
    </w:r>
  </w:p>
</w:ftr>'''
    with open(os.path.join(unpacked_dir, 'word/footer1.xml'), 'w', encoding='utf-8') as f:
        f.write(footer_xml)


def update_relationships(unpacked_dir):
    """リレーションシップ更新"""
    rels_path = os.path.join(unpacked_dir, 'word/_rels/document.xml.rels')
    with open(rels_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 既存の最大rId番号を取得
    rids = re.findall(r'Id=\"rId(\d+)\"', content)
    max_rid = max(int(r) for r in rids) if rids else 0
    
    header_rid = f"rId{max_rid + 1}"
    footer_rid = f"rId{max_rid + 2}"
    
    header_rel = f'<Relationship Id=\"{header_rid}\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/header\" Target=\"header1.xml\"/>'
    footer_rel = f'<Relationship Id=\"{footer_rid}\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer\" Target=\"footer1.xml\"/>'
    
    content = content.replace('</Relationships>', f'{header_rel}\n{footer_rel}\n</Relationships>')
    
    with open(rels_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Content_Types更新
    ct_path = os.path.join(unpacked_dir, '[Content_Types].xml')
    with open(ct_path, 'r', encoding='utf-8') as f:
        ct_content = f.read()
    
    if 'PartName=\"/word/header1.xml\"' not in ct_content:
        header_override = '<Override PartName=\"/word/header1.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.header+xml\"/>'
        ct_content = ct_content.replace('</Types>', f'{header_override}\n</Types>')
    if 'PartName=\"/word/footer1.xml\"' not in ct_content:
        footer_override = '<Override PartName=\"/word/footer1.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml\"/>'
        ct_content = ct_content.replace('</Types>', f'{footer_override}\n</Types>')
    
    with open(ct_path, 'w', encoding='utf-8') as f:
        f.write(ct_content)
    
    return header_rid, footer_rid


def create_cover_page(doc_title, created_date, version, author):
    """
    表紙を作成
    構造:
    - 上部余白(4行)
    - 紺線(上)
    - タイトル(中央、22pt太字、黒)
    - 紺線(下)
    - 中間余白(6行)
    - メタ情報(テキスト、表ではない)
    - 改ページ
    """
    escaped_title = xml_escape(doc_title)
    escaped_date = xml_escape(created_date)
    escaped_version = xml_escape(version)
    escaped_author = xml_escape(author)
    
    # メタ情報(テキスト形式、中央揃え、太字)
    meta_paragraphs = ''
    if created_date:
        meta_paragraphs += f'''
    <w:p>
      <w:pPr><w:jc w:val=\"center\"/><w:spacing w:after=\"120\"/></w:pPr>
      <w:r><w:rPr>{MEIRYO_FONTS}<w:b/><w:bCs/><w:sz w:val=\"{SIZE_META}\"/><w:szCs w:val=\"{SIZE_META}\"/><w:color w:val=\"{BLACK_COLOR}\"/></w:rPr><w:t>作成日 : {escaped_date}</w:t></w:r>
    </w:p>'''
    if version:
        meta_paragraphs += f'''
    <w:p>
      <w:pPr><w:jc w:val=\"center\"/><w:spacing w:after=\"120\"/></w:pPr>
      <w:r><w:rPr>{MEIRYO_FONTS}<w:b/><w:bCs/><w:sz w:val=\"{SIZE_META}\"/><w:szCs w:val=\"{SIZE_META}\"/><w:color w:val=\"{BLACK_COLOR}\"/></w:rPr><w:t>版数 : {escaped_version}</w:t></w:r>
    </w:p>'''
    if author:
        meta_paragraphs += f'''
    <w:p>
      <w:pPr><w:jc w:val=\"center\"/><w:spacing w:after=\"120\"/></w:pPr>
      <w:r><w:rPr>{MEIRYO_FONTS}<w:b/><w:bCs/><w:sz w:val=\"{SIZE_META}\"/><w:szCs w:val=\"{SIZE_META}\"/><w:color w:val=\"{BLACK_COLOR}\"/></w:rPr><w:t>作成者 : {escaped_author}</w:t></w:r>
    </w:p>'''
    
    # 空行を生成
    empty_line = '<w:p><w:pPr><w:spacing w:after=\"0\"/></w:pPr></w:p>\n    '
    
    return f'''
    {empty_line * 4}
    <w:p><w:pPr><w:pBdr><w:bottom w:val=\"single\" w:sz=\"24\" w:space=\"1\" w:color=\"{NAVY_COLOR}\"/></w:pBdr><w:spacing w:after=\"200\"/><w:jc w:val=\"center\"/></w:pPr></w:p>
    <w:p>
      <w:pPr><w:jc w:val=\"center\"/><w:spacing w:before=\"200\" w:after=\"200\"/></w:pPr>
      <w:r><w:rPr>{MEIRYO_FONTS}<w:b/><w:bCs/><w:sz w:val=\"{SIZE_TITLE}\"/><w:szCs w:val=\"{SIZE_TITLE}\"/><w:color w:val=\"{BLACK_COLOR}\"/></w:rPr><w:t>{escaped_title}</w:t></w:r>
    </w:p>
    <w:p><w:pPr><w:pBdr><w:top w:val=\"single\" w:sz=\"24\" w:space=\"1\" w:color=\"{NAVY_COLOR}\"/></w:pBdr><w:spacing w:before=\"0\" w:after=\"0\"/><w:jc w:val=\"center\"/></w:pPr></w:p>
    {empty_line * 6}
    {meta_paragraphs}
    <w:p><w:r><w:br w:type=\"page\"/></w:r></w:p>
'''


def update_document_content(unpacked_dir, header_rid, footer_rid,
                           doc_title, created_date, version, author):
    """文書内容を更新"""
    doc_path = os.path.join(unpacked_dir, 'word/document.xml')
    with open(doc_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # sectPrを完全に書き換え
    new_sectpr = f'''<w:sectPr>
  <w:headerReference w:type=\"default\" r:id=\"{header_rid}\"/>
  <w:footerReference w:type=\"default\" r:id=\"{footer_rid}\"/>
  <w:pgSz w:w=\"11906\" w:h=\"16838\"/>
  <w:pgMar w:top=\"1440\" w:right=\"1440\" w:bottom=\"1440\" w:left=\"1440\" w:header=\"708\" w:footer=\"708\" w:gutter=\"0\"/>
  <w:cols w:space=\"720\"/>
  <w:docGrid w:linePitch=\"360\"/>
</w:sectPr>'''
    
    content = re.sub(r'<w:sectPr[^>]*/>', new_sectpr, content)
    content = re.sub(r'<w:sectPr[^>]*>.*?</w:sectPr>', new_sectpr, content, flags=re.DOTALL)
    
    # 目次タイトルを「目次」に変更
    content = re.sub(r'Table of Contents', '目次', content)
    
    # ★★★ 目次SDTをクリーンアップ(pandocが本文を含めてしまう問題の修正)★★★
    content = cleanup_toc_content(content)
    
    # 水平線を改ページに変換
    hr_pattern = r'<w:p[^>]*>\s*<w:r>\s*<w:pict>\s*<v:rect[^>]*o:hr=\"t\"[^/]*/>\s*</w:pict>\s*</w:r>\s*</w:p>'
    content = re.sub(hr_pattern, '<w:p><w:r><w:br w:type=\"page\"/></w:r></w:p>', content, flags=re.DOTALL)
    
    # 表紙挿入: <w:body>直後、目次(<w:sdt>)の前に挿入
    body_start = content.find('<w:body>') + len('<w:body>')
    sdt_start = content.find('<w:sdt>')
    
    if body_start > 0:
        # 既存のTitle/Subtitleスタイルの段落を削除(pandocが生成したもの)
        if sdt_start > 0:
            before_sdt = content[body_start:sdt_start]
            before_sdt = re.sub(r'<w:p[^>]*>\s*<w:pPr>\s*<w:pStyle w:val=\"Title\"[^/]*/>\s*</w:pPr>.*?</w:p>', '', before_sdt, flags=re.DOTALL)
            before_sdt = re.sub(r'<w:p[^>]*>\s*<w:pPr>\s*<w:pStyle w:val=\"Subtitle\"[^/]*/>\s*</w:pPr>.*?</w:p>', '', before_sdt, flags=re.DOTALL)
            before_sdt = re.sub(r'<w:p[^>]*>\s*<w:pPr>\s*<w:pStyle w:val=\"FirstParagraph\"[^/]*/>\s*</w:pPr>.*?</w:p>', '', before_sdt, flags=re.DOTALL)
            before_sdt = re.sub(r'<w:bookmarkStart[^/]*/>\s*<w:bookmarkEnd[^/]*/>', '', before_sdt)
            before_sdt = before_sdt.strip()
        
        cover_xml = create_cover_page(doc_title, created_date, version, author)
        
        if sdt_start > 0:
            # 目次の終了位置を見つける
            sdt_end_marker = '</w:sdtContent></w:sdt>'
            sdt_end = content.find(sdt_end_marker, sdt_start)
            if sdt_end < 0:
                sdt_end_marker = '</w:sdtContent>\n    </w:sdt>'
                sdt_end = content.find(sdt_end_marker, sdt_start)
            
            if sdt_end > 0:
                sdt_end += len(sdt_end_marker)
                # 表紙 + 目次 + 改ページ + 本文
                content = (content[:body_start] + 
                          cover_xml + 
                          content[sdt_start:sdt_end] + 
                          '<w:p><w:r><w:br w:type=\"page\"/></w:r></w:p>' + 
                          content[sdt_end:])
        else:
            # 目次がない場合は表紙のみ挿入
            content = content[:body_start] + cover_xml + content[body_start:]
    
    # 章の前に改ページ(2つ目以降のHeading1)
    h1_pattern = r'(<w:pStyle w:val=\"Heading1\"/>)'
    matches = list(re.finditer(h1_pattern, content))
    for match in reversed(matches[1:]):
        content = content[:match.end()] + '<w:pageBreakBefore/>' + content[match.end():]
    
    with open(doc_path, 'w', encoding='utf-8') as f:
        f.write(content)


def apply_table_borders(unpacked_dir):
    """表に黒罫線を適用"""
    doc_path = os.path.join(unpacked_dir, 'word/document.xml')
    with open(doc_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    borders_xml = f'''<w:tblBorders>
      <w:top w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"{BLACK_COLOR}\"/>
      <w:left w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"{BLACK_COLOR}\"/>
      <w:bottom w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"{BLACK_COLOR}\"/>
      <w:right w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"{BLACK_COLOR}\"/>
      <w:insideH w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"{BLACK_COLOR}\"/>
      <w:insideV w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"{BLACK_COLOR}\"/>
    </w:tblBorders>'''
    
    def add_borders(match):
        tbl_pr = match.group(0)
        if 'w:val=\"nil\"' in tbl_pr or '<w:tblBorders>' in tbl_pr:
            return tbl_pr
        return tbl_pr.replace('</w:tblPr>', borders_xml + '</w:tblPr>')
    
    content = re.sub(r'<w:tblPr>.*?</w:tblPr>', add_borders, content, flags=re.DOTALL)
    
    with open(doc_path, 'w', encoding='utf-8') as f:
        f.write(content)


def center_tables(unpacked_dir):
    """表を中央揃えにする"""
    doc_path = os.path.join(unpacked_dir, 'word/document.xml')
    with open(doc_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 中央揃え用のXML
    center_jc = '<w:jc w:val=\"center\"/>'
    
    def add_center_to_table(match):
        tbl_pr = match.group(0)
        # 表紙のメタ表(罫線なし)はスキップ
        if 'w:val=\"nil\"' in tbl_pr:
            return tbl_pr
        # 既に中央揃えがある場合はスキップ
        if '<w:jc' in tbl_pr:
            return tbl_pr
        # </w:tblPr>の前に中央揃えを追加
        return tbl_pr.replace('</w:tblPr>', center_jc + '</w:tblPr>')
    
    content = re.sub(r'<w:tblPr>.*?</w:tblPr>', add_center_to_table, content, flags=re.DOTALL)
    
    with open(doc_path, 'w', encoding='utf-8') as f:
        f.write(content)


def center_images(unpacked_dir):
    """画像を中央揃えにする(画像を含む段落を中央揃えに)"""
    doc_path = os.path.join(unpacked_dir, 'word/document.xml')
    with open(doc_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 画像(w:drawing)を含む段落を検出して中央揃えにする
    def center_paragraph_with_image(match):
        paragraph = match.group(0)
        # 画像が含まれていない場合はそのまま
        if '<w:drawing>' not in paragraph:
            return paragraph
        
        center_jc = '<w:jc w:val=\"center\"/>'
        
        # 既に段落プロパティがある場合
        if '<w:pPr>' in paragraph:
            # 既に中央揃えがある場合はスキップ
            if '<w:jc' in paragraph:
                return paragraph
            # </w:pPr>の前に中央揃えを追加
            paragraph = re.sub(r'(</w:pPr>)', center_jc + r'\1', paragraph, count=1)
        else:
            # 段落プロパティがない場合は追加
            paragraph = paragraph.replace('<w:p>', f'<w:p><w:pPr>{center_jc}</w:pPr>', 1)
            paragraph = paragraph.replace('<w:p ', f'<w:p><w:pPr>{center_jc}</w:pPr><w:p_placeholder ', 1)
            paragraph = paragraph.replace('<w:p_placeholder ', '', 1)
        
        return paragraph
    
    content = re.sub(r'<w:p[^>]*>.*?</w:p>', center_paragraph_with_image, content, flags=re.DOTALL)
    
    with open(doc_path, 'w', encoding='utf-8') as f:
        f.write(content)


def apply_table_header_style(unpacked_dir):
    """表ヘッダー行に紺背景+白太字を適用"""
    doc_path = os.path.join(unpacked_dir, 'word/document.xml')
    with open(doc_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    def process_tables(content):
        parts = re.split(r'(<w:tbl>|<w:tbl\s[^>]*>)', content)
        result = []
        i = 0
        while i < len(parts):
            part = parts[i]
            if part.startswith('<w:tbl'):
                result.append(part)
                i += 1
                if i < len(parts):
                    table_content = parts[i]
                    tbl_end_idx = table_content.find('</w:tbl>')
                    if tbl_end_idx != -1:
                        table_body = table_content[:tbl_end_idx]
                        after_table = table_content[tbl_end_idx:]
                        
                        # 罫線なし表(表紙のメタ表など)はスキップ
                        if 'w:val=\"nil\"' in table_body:
                            result.append(table_content)
                            i += 1
                            continue
                        
                        # 最初の行をヘッダーとしてスタイル適用
                        first_tr = re.search(r'(<w:tr[^>]*>)(.*?)(</w:tr>)', table_body, re.DOTALL)
                        if first_tr:
                            tr_start, tr_content, tr_end = first_tr.groups()
                            
                            def style_cell(m):
                                tc_start, tc_content, tc_end = m.groups()
                                shading = f'<w:shd w:val=\"clear\" w:color=\"auto\" w:fill=\"{NAVY_COLOR}\"/>'
                                if '<w:tcPr/>' in tc_content:
                                    tc_content = tc_content.replace('<w:tcPr/>', f'<w:tcPr>{shading}</w:tcPr>')
                                elif '<w:tcPr>' in tc_content:
                                    tc_content = re.sub(r'(<w:tcPr[^/>]*>)', r'\1' + shading, tc_content, count=1)
                                else:
                                    tc_content = f'<w:tcPr>{shading}</w:tcPr>' + tc_content
                                
                                def add_white_bold(rm):
                                    r_tag, r_content, r_end = rm.groups()
                                    style = f'<w:color w:val=\"{WHITE_COLOR}\"/><w:b/><w:bCs/>'
                                    if '<w:rPr>' in r_content:
                                        r_content = re.sub(r'(<w:rPr[^>]*>)', r'\1' + style, r_content, count=1)
                                    else:
                                        r_content = f'<w:rPr>{style}</w:rPr>' + r_content
                                    return r_tag + r_content + r_end
                                
                                tc_content = re.sub(r'(<w:r(?:\s[^>]*)?>)(.*?)(</w:r>)', add_white_bold, tc_content, flags=re.DOTALL)
                                return tc_start + tc_content + tc_end
                            
                            styled_tr = re.sub(r'(<w:tc[^/>]*>)(.*?)(</w:tc>)', style_cell, tr_content, flags=re.DOTALL)
                            table_body = table_body.replace(first_tr.group(0), tr_start + styled_tr + tr_end, 1)
                        
                        result.append(table_body + after_table)
                    else:
                        result.append(table_content)
            else:
                result.append(part)
            i += 1
        return ''.join(result)
    
    content = process_tables(content)
    
    with open(doc_path, 'w', encoding='utf-8') as f:
        f.write(content)


def remove_italics(unpacked_dir):
    """斜体を完全削除(本文から)"""
    doc_path = os.path.join(unpacked_dir, 'word/document.xml')
    with open(doc_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # <w:i/>と<w:iCs/>を削除(斜体を一切使用しない)
    content = re.sub(r'<w:i/>', '', content)
    content = re.sub(r'<w:iCs/>', '', content)
    content = re.sub(r'<w:i\s*/>', '', content)
    content = re.sub(r'<w:iCs\s*/>', '', content)
    
    with open(doc_path, 'w', encoding='utf-8') as f:
        f.write(content)


def cleanup_toc_content(content):
    """
    pandocが生成するTOC(目次)SDTをクリーンアップ。
    
    pandocの問題: TOCフィールド内に見出しだけでなく本文も含めてしまう
    解決策: TOC SDTの内容を、TOCフィールドとプレースホルダーのみに簡略化
    """
    # TOC SDTの開始と終了を特定
    sdt_start = content.find('<w:sdt>')
    if sdt_start < 0:
        return content
    
    sdt_end = content.find('</w:sdt>', sdt_start)
    if sdt_end < 0:
        return content
    sdt_end += len('</w:sdt>')
    
    # 元のSDT内容を取得
    original_sdt = content[sdt_start:sdt_end]
    
    # sdtPr部分を抽出(目次設定を保持)
    sdtpr_match = re.search(r'<w:sdtPr>.*?</w:sdtPr>', original_sdt, re.DOTALL)
    if not sdtpr_match:
        return content
    sdtpr = sdtpr_match.group(0)
    
    # 新しいシンプルなTOC SDTを作成
    # TOCHeadingとTOCフィールドのみを含む最小構成
    new_sdt = f'''<w:sdt>
      {sdtpr}
      <w:sdtContent>
        <w:p>
          <w:pPr>
            <w:pStyle w:val=\"TOCHeading\"/>
          </w:pPr>
          <w:r>
            <w:t xml:space=\"preserve\">目次</w:t>
          </w:r>
        </w:p>
        <w:p>
          <w:r>
            <w:fldChar w:fldCharType=\"begin\" w:dirty=\"true\"/>
          </w:r>
          <w:r>
            <w:instrText xml:space=\"preserve\">TOC \\\\o \"1-4\" \\\\h \\\\z \\\\u</w:instrText>
          </w:r>
          <w:r>
            <w:fldChar w:fldCharType=\"separate\"/>
          </w:r>
          <w:r>
            <w:t>(Wordで目次を更新してください: 右クリック→「フィールドの更新」またはF9キー)</w:t>
          </w:r>
          <w:r>
            <w:fldChar w:fldCharType=\"end\"/>
          </w:r>
        </w:p>
      </w:sdtContent>
    </w:sdt>'''
    
    return content[:sdt_start] + new_sdt + content[sdt_end:]


def ensure_xml_escape(unpacked_dir):
    """XMLエスケープを確認・修正"""
    for xml_file in ['word/document.xml', 'word/header1.xml', 'word/footer1.xml']:
        file_path = os.path.join(unpacked_dir, xml_file)
        if not os.path.exists(file_path):
            continue
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # エスケープされていない&を修正
        # &amp; &lt; &gt; &quot; &apos; &#数字; は除外
        pattern = r'&(?!(amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)'
        content = re.sub(pattern, '&amp;', content)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)


def update_font_table(unpacked_dir):
    """fontTable.xmlにMeiryo UIを追加(フォント適用に必須)"""
    font_table_path = os.path.join(unpacked_dir, 'word/fontTable.xml')
    with open(font_table_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Meiryo UIが既に存在するか確認
    if 'Meiryo UI' in content:
        return
    
    # Meiryo UIフォント定義を追加
    meiryo_font = '''  <w:font w:name=\"Meiryo UI\">
    <w:altName w:val=\"Meiryo UI\"/>
    <w:panose1 w:val=\"020B0604030504040204\"/>
    <w:charset w:val=\"80\"/>
    <w:family w:val=\"swiss\"/>
    <w:pitch w:val=\"variable\"/>
    <w:sig w:usb0=\"E00002FF\" w:usb1=\"6AC7FFFF\" w:usb2=\"08000012\" w:usb3=\"00000000\" w:csb0=\"0002009F\" w:csb1=\"00000000\"/>
  </w:font>
'''
    
    # </w:fonts>の前に追加
    content = content.replace('</w:fonts>', meiryo_font + '</w:fonts>')
    
    with open(font_table_path, 'w', encoding='utf-8') as f:
        f.write(content)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("使用方法: python convert.py <入力.md> <出力.docx>")
        sys.exit(1)
    
    convert_md_to_word(sys.argv[1], sys.argv[2])
