from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
import pandas as pd
import os

def process_pdf_with_chunks(file_path: str, endpoint: str, key: str, output_excel: str):
    """PDFをチャンク化してExcelに出力する"""
    # ドキュメントの読み込み
    loader = AzureAIDocumentIntelligenceLoader(
        api_endpoint=endpoint,
        api_key=key,
        file_path=file_path,
        api_model="prebuilt-layout"
    )

    # Markdownヘッダーに基づくチャンキング
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3")
    ]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    documents = loader.load()
    chunks = splitter.split_text(documents[0].page_content)

    # チャンクデータの整形
    data = []
    for chunk in chunks:
        chunk_data = {
            'ファイル名': os.path.basename(file_path),
            'Header 1': chunk.metadata.get('Header 1', ''),
            'Header 2': chunk.metadata.get('Header 2', ''),
            'Header 3': chunk.metadata.get('Header 3', ''),
            '内容': chunk.page_content
        }
        data.append(chunk_data)
        
        # 処理状況の表示
        print(f"\n===== チャンク {len(data)} =====")
        print(f"見出し1: {chunk_data['Header 1']}")
        print(f"見出し2: {chunk_data['Header 2']}")
        print(f"見出し3: {chunk_data['Header 3']}")
        print(f"内容: {chunk_data['内容'][:200]}...")
        print("=" * 50)

    # データフレーム作成とExcel出力
    df = pd.DataFrame(data)
    df.to_excel(output_excel, index=False)
    print(f"\n処理が完了しました。結果は '{output_excel}' に保存されています。")

def process_directory(input_dir: str, endpoint: str, key: str, output_excel: str):
    """ディレクトリ内のPDFファイルを処理"""
    all_data = []
    
    # Document Intelligence クライアントの初期化
    document_intelligence_client = DocumentIntelligenceClient(
        endpoint=endpoint, 
        credential=AzureKeyCredential(key)
    )

    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(root, filename)
                print(f"\n処理中: {filename}")
                
                try:
                    # 単一PDFの処理
                    loader = AzureAIDocumentIntelligenceLoader(
                        api_endpoint=endpoint,
                        api_key=key,
                        file_path=file_path,
                        api_model="prebuilt-layout"
                    )
                    
                    documents = loader.load()
                    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
                        ("#", "Header 1"),
                        ("##", "Header 2"),
                        ("###", "Header 3")
                    ])
                    chunks = splitter.split_text(documents[0].page_content)
                    
                    # チャンクデータの追加
                    for chunk in chunks:
                        chunk_data = {
                            'ファイル名': filename,
                            'Header 1': chunk.metadata.get('Header 1', ''),
                            'Header 2': chunk.metadata.get('Header 2', ''),
                            'Header 3': chunk.metadata.get('Header 3', ''),
                            '内容': chunk.page_content
                        }
                        all_data.append(chunk_data)
                        
                        print(f"チャンク追加: {chunk_data['Header 1']}")

                except Exception as e:
                    print(f"エラー発生 ({filename}): {str(e)}")
                    continue

    # 全データをExcelに出力
    if all_data:
        df = pd.DataFrame(all_data)
        df.to_excel(output_excel, index=False)
        print(f"\n処理が完了しました。結果は '{output_excel}' に保存されています。")
    else:
        print("\n処理可能なデータがありませんでした。")

if __name__ == "__main__":
    # 環境変数から認証情報を取得
    import os
    endpoint = os.getenv("AZURE_FORM_RECOGNIZER_ENDPOINT", "your_endpoint_here")
    key = os.getenv("AZURE_FORM_RECOGNIZER_KEY", "your_key_here")
    
    # ディレクトリ内の全PDFを処理
    process_directory(
        "manuals",
        endpoint,
        key,
        "output_chunks.xlsx"
    )