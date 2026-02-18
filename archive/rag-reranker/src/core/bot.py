import os
import platform
from typing import List, Dict, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain.schema import Document, HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from config import *
from models.metadata import DocumentMetadata, SearchParameters
from models.embedding import create_embedding_model
from models.reranker import RerankerModel
from utils.text import extract_keywords, split_by_hierarchical_headers, process_pdf_content, process_excel_content
from utils.file import (find_documents_recursively, get_relative_path, 
                       save_pickle, load_pickle, save_json, load_json, load_text)

# ANSIカラーコード
class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    GRAY = '\033[90m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

class LoanAssistantBot:
    """融資業務Q&Aボットの主要クラス"""

    def __init__(self, 
                 model_provider: str = MODEL_PROVIDER,
                 model_name: str = MODEL_NAME,
                 embedding_provider: str = EMBEDDING_PROVIDER,
                 embedding_model_name: str = EMBEDDING_MODEL,
                 reranker_model_name: str = RERANKER_MODEL,
                 prompt_path: str = DEFAULT_PROMPT_FILE):
        """初期化"""
        # AIモデルと埋め込みモデルの初期化
        self.chat_model = self._create_chat_model(model_provider, model_name)
        self.embeddings = create_embedding_model(embedding_provider, embedding_model_name)
        self.reranker = RerankerModel(reranker_model_name)

        # データ保持用の変数初期化
        self.chunks = []
        self.vectors = []
        self.doc_sources = {}
        self.file_paths = {}

        # OSの種類を検出
        self.is_windows = platform.system() == "Windows"

        # プロンプトテンプレートの読み込み
        if prompt_path:
            self.prompt_template = load_text(prompt_path)

        # 検索パラメータの初期化と読み込み
        self.search_params = SearchParameters()
        self._load_search_params()

    def _create_chat_model(self, model_provider: str, model_name: str):
        """チャットモデルを作成"""
        if model_provider == "anthropic":
            return ChatAnthropic(
                anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
                model=model_name,
                temperature=0
            )
        elif model_provider == "openai":
            return ChatOpenAI(
                openai_api_key=os.environ.get("OPENAI_API_KEY"),
                model_name=model_name,
                temperature=0
            )
        elif model_provider == "gemini":
            return ChatGoogleGenerativeAI(
                api_key=os.environ.get("GOOGLE_API_KEY"),
                model=model_name,
                temperature=0
            )
        else:
            raise ValueError(f"Unsupported model provider: {model_provider}")

    def _create_file_link(self, file_path: str) -> str:
        """OSに応じたファイルリンクを生成"""
        if self.is_windows:
            # Windowsの場合、バックスラッシュに変換
            file_path = file_path.replace('/', '\\')
            return f"{Colors.BLUE}{Colors.UNDERLINE}file://{file_path}{Colors.END}"
        else:
            # Unix系の場合
            return f"{Colors.BLUE}{Colors.UNDERLINE}file://{file_path}{Colors.END}"

    def _load_search_params(self) -> None:
        """保存された検索パラメータを読み込む"""
        try:
            if os.path.exists(PARAMS_FILE):
                data = load_json(PARAMS_FILE)
                self.search_params = SearchParameters.from_dict(data)
                print(f"{Colors.GREEN}検索パラメータを読み込みました{Colors.END}")
        except Exception as e:
            print(f"{Colors.YELLOW}検索パラメータの読み込み中にエラーが発生しました: {e}")
            print("デフォルトの検索パラメータを使用します{Colors.END}")

    def save_search_params(self) -> None:
        """検索パラメータをファイルに保存"""
        try:
            save_json(self.search_params.to_dict(), PARAMS_FILE)
            print(f"{Colors.GREEN}検索パラメータを保存しました{Colors.END}")
        except Exception as e:
            print(f"{Colors.YELLOW}検索パラメータの保存中にエラーが発生しました: {e}{Colors.END}")

    def update_search_params(self,
                           initial_top_k: Optional[int] = None,
                           final_top_k: Optional[int] = None,
                           semantic_weight: Optional[float] = None) -> None:
        """検索パラメータを更新"""
        if initial_top_k is not None:
            self.search_params.initial_top_k = initial_top_k
        if final_top_k is not None:
            self.search_params.final_top_k = final_top_k
        if semantic_weight is not None:
            self.search_params.semantic_weight = semantic_weight
        
        try:
            self.search_params.validate()
            self.save_search_params()
        except ValueError as e:
            print(f"{Colors.YELLOW}パラメータの検証に失敗しました: {e}{Colors.END}")
            raise

    def show_search_params(self) -> None:
        """現在の検索パラメータを表示"""
        print(f"\n{Colors.BOLD}=== 現在の検索パラメータ ==={Colors.END}")
        print(f"初期検索件数 (initial_top_k): {Colors.GREEN}{self.search_params.initial_top_k}{Colors.END}")
        print(f"最終検索件数 (final_top_k): {Colors.GREEN}{self.search_params.final_top_k}{Colors.END}")
        print(f"意味的類似度の重み (semantic_weight): {Colors.GREEN}{self.search_params.semantic_weight:.2f}{Colors.END}")

    def vectorize_chunks(self) -> None:
        """チャンクをベクトル化"""
        print(f"{Colors.GRAY}Vectorizing {len(self.chunks)} chunks...{Colors.END}")
        self.vectors = []

        for i, chunk in enumerate(self.chunks):
            try:
                embedding = self.embeddings.embed_query(chunk.page_content)
                self.vectors.append(embedding)

                if (i + 1) % 10 == 0:
                    print(f"{Colors.GRAY}Vectorized {i + 1}/{len(self.chunks)} chunks{Colors.END}")

            except Exception as e:
                print(f"{Colors.YELLOW}Error vectorizing chunk {i}: {str(e)}{Colors.END}")
                self.vectors.append([0.0] * self.embeddings.get_vector_dimension())

    def find_relevant_chunks_hybrid(self, question: str) -> List[Dict]:
        """意味検索とキーワード検索を組み合わせたハイブリッド検索を実行"""
        try:
            params = self.search_params
            semantic_weight = params.semantic_weight
            keyword_weight = 1.0 - semantic_weight

            if not self.chunks or not self.vectors:
                print(f"{Colors.YELLOW}警告: チャンクまたはベクトルが存在しません{Colors.END}")
                return []

            # Semantic search
            question_embedding = self.embeddings.embed_query(question)
            semantic_scores = [
                float(cosine_similarity([question_embedding], [vector])[0][0])
                for vector in self.vectors
            ]

            # Keyword search
            keyword_scores = []
            keywords = extract_keywords(question)
            
            for chunk in self.chunks:
                chunk_keywords = extract_keywords(chunk.page_content)
                if not keywords:
                    score = 0.0
                else:
                    matches = sum(1 for kw in keywords if any(
                        kw.lower() in ck.lower() for ck in chunk_keywords
                    ))
                    score = matches / len(keywords)
                keyword_scores.append(score)

            # Normalize scores
            max_semantic = max(semantic_scores) if semantic_scores else 1.0
            max_keyword = max(keyword_scores) if keyword_scores else 1.0
            
            semantic_scores = [s / max_semantic if max_semantic > 0 else 1.0 for s in semantic_scores]
            keyword_scores = [k / max_keyword if max_keyword > 0 else 1.0 for k in keyword_scores]

            # Combine scores
            combined_scores = [
                (semantic_weight * s) + (keyword_weight * k)
                for s, k in zip(semantic_scores, keyword_scores)
            ]

            # Convert scores to numpy array and get indices
            scores_array = np.array(combined_scores)
            top_k = min(params.initial_top_k, len(scores_array))
            top_indices = np.argsort(scores_array)[-top_k:][::-1]
            
            # Convert indices to integer type
            top_indices = top_indices.astype(int)

            initial_results = []
            for idx in top_indices:
                source = self.doc_sources.get(idx, "Unknown Source")
                initial_results.append({
                    'content': self.chunks[idx].page_content,
                    'source': source,
                    'file_path': self.file_paths.get(source, ""),
                    'combined_similarity': float(combined_scores[idx]),  # 明示的に float に変換
                    'semantic_similarity': float(semantic_scores[idx]),  # 明示的に float に変換
                    'keyword_similarity': float(keyword_scores[idx])     # 明示的に float に変換
                })

            # Rerank results
            if initial_results:
                reranked_results = self.reranker.rerank(question, initial_results)
                final_top_k = min(params.final_top_k, len(reranked_results))
                return reranked_results[:final_top_k]
            else:
                return []

        except Exception as e:
            print(f"{Colors.YELLOW}検索処理中にエラーが発生しました: {str(e)}{Colors.END}")
            return []

    def _get_files_metadata(self) -> DocumentMetadata:
        """現在のドキュメントディレクトリの状態を取得"""
        files = []
        last_modified_times = {}
        
        documents = find_documents_recursively(DOCS_DIRECTORY)
        
        for file_paths in documents.values():
            for file_path in file_paths:
                relative_path = get_relative_path(file_path, DOCS_DIRECTORY)
                files.append(relative_path)
                last_modified_times[relative_path] = os.path.getmtime(file_path)
        
        return DocumentMetadata(
            files=sorted(files),
            last_modified_times=last_modified_times,
            total_chunks=len(self.chunks) if self.chunks else 0,
            model_name=self.embeddings.model_name
        )

    def _should_update_vectors(self) -> bool:
        """ベクトルの更新が必要かどうかを判定"""
        if not os.path.exists(METADATA_FILE):
            return True
        
        try:
            stored_metadata = DocumentMetadata.from_dict(load_json(METADATA_FILE))
            current_metadata = self._get_files_metadata()
            
            return (stored_metadata.files != current_metadata.files or
                    stored_metadata.last_modified_times != current_metadata.last_modified_times or
                    stored_metadata.model_name != current_metadata.model_name)
            
        except Exception as e:
            print(f"{Colors.YELLOW}メタデータの検証中にエラーが発生しました: {e}{Colors.END}")
            return True

    def initialize(self) -> None:
        """ボットの初期化処理を実行"""
        if self._should_update_vectors() or not self.load_vectors():
            print(f"{Colors.YELLOW}ドキュメントの変更を検出したため、ベクトルを更新します...{Colors.END}")
            self.load_documents_from_directory()
            if self.chunks:
                self.vectorize_chunks()
                self.save_vectors()
        else:
            print(f"{Colors.GREEN}キャッシュされたベクトルを使用します{Colors.END}")

    def load_documents_from_directory(self) -> None:
        """ディレクトリから文書を再帰的に読み込む"""
        self.chunks = []
        self.doc_sources = {}
        self.file_paths = {}
        
        documents = find_documents_recursively(DOCS_DIRECTORY)
        print(f"{Colors.GRAY}Found {len(documents['pdf'])} PDF files and {len(documents['excel'])} Excel files{Colors.END}")
        
        # PDFファイルの処理
        for file_path in documents['pdf']:
            relative_path = get_relative_path(file_path, DOCS_DIRECTORY)
            try:
                self._load_single_pdf(file_path)
                print(f"{Colors.GREEN}Successfully loaded PDF: {relative_path}{Colors.END}")
            except Exception as e:
                print(f"{Colors.YELLOW}Error loading PDF {relative_path}: {str(e)}{Colors.END}")
        
        # Excelファイルの処理
        for file_path in documents['excel']:
            relative_path = get_relative_path(file_path, DOCS_DIRECTORY)
            try:
                self._load_single_excel(file_path)
                print(f"{Colors.GREEN}Successfully loaded Excel: {relative_path}{Colors.END}")
            except Exception as e:
                print(f"{Colors.YELLOW}Error loading Excel {relative_path}: {str(e)}{Colors.END}")

    def _load_single_pdf(self, pdf_path: str) -> None:
        """PDFファイルを処理してチャンクに変換"""
        relative_path = get_relative_path(pdf_path, DOCS_DIRECTORY)
        absolute_path = os.path.abspath(pdf_path)
        
        try:
            # PDFをテキストに変換
            full_text = process_pdf_content(pdf_path)
            # テキストを階層的な見出しで分割
            chunks = split_by_hierarchical_headers(full_text)
            start_index = len(self.chunks)
            
            # 各チャンクをDocumentオブジェクトに変換
            for i, chunk_data in enumerate(chunks):
                if len(chunk_data["text"].strip()) > 0:
                    chunk = Document(
                        page_content=f"【見出し】{chunk_data['header']}\n\n{chunk_data['text']}",
                        metadata={
                            'header': chunk_data['header'],
                            'source': relative_path
                        }
                    )
                    self.chunks.append(chunk)
                    self.doc_sources[start_index + i] = relative_path
                    self.file_paths[relative_path] = absolute_path
            
            print(f"{Colors.GRAY}Added {len(chunks)} hierarchical header-based chunks from {relative_path}{Colors.END}")
            
        except Exception as e:
            print(f"{Colors.YELLOW}Error processing text from {relative_path}: {str(e)}{Colors.END}")
            try:
                # フォールバック：単一のチャンクとして処理
                chunk = Document(
                    page_content=full_text,
                    metadata={'source': relative_path}
                )
                self.chunks.append(chunk)
                self.doc_sources[len(self.chunks) - 1] = relative_path
                self.file_paths[relative_path] = absolute_path
                print(f"{Colors.GRAY}Added single chunk from {relative_path} using fallback method{Colors.END}")
            except Exception as e2:
                print(f"{Colors.YELLOW}Fallback also failed for {relative_path}: {str(e2)}{Colors.END}")

    def _load_single_excel(self, excel_path: str) -> None:
        """Excelファイルを処理してチャンクに変換"""
        relative_path = get_relative_path(excel_path, DOCS_DIRECTORY)
        absolute_path = os.path.abspath(excel_path)
        start_index = len(self.chunks)
        
        try:
            new_chunks = process_excel_content(excel_path)
            for chunk in new_chunks:
                chunk.metadata['source'] = relative_path
                self.chunks.append(chunk)
                self.doc_sources[start_index + len(self.chunks) - start_index - 1] = relative_path
                self.file_paths[relative_path] = absolute_path
            
            print(f"{Colors.GRAY}Added {len(new_chunks)} chunks from {relative_path}{Colors.END}")
            
        except Exception as e:
            print(f"{Colors.YELLOW}Error processing Excel file {relative_path}: {str(e)}{Colors.END}")

    def save_vectors(self) -> None:
        """ベクトルとメタデータを保存"""
        try:
            # チャンク、ドキュメントソース、ファイルパスの保存
            save_pickle({
                'chunks': self.chunks,
                'doc_sources': self.doc_sources,
                'file_paths': self.file_paths
            }, CHUNKS_FILE)
            
            # ベクトルの保存
            save_pickle(self.vectors, VECTORS_FILE)
            
            # メタデータの保存
            metadata = self._get_files_metadata()
            save_json(metadata.to_dict(), METADATA_FILE)
            
            print(f"{Colors.GREEN}ベクトルとメタデータを保存しました{Colors.END}")
            
        except Exception as e:
            print(f"{Colors.YELLOW}保存中にエラーが発生しました: {e}{Colors.END}")

    def load_vectors(self) -> bool:
        """保存されたベクトルとメタデータを読み込み"""
        try:
            if not all(os.path.exists(f) for f in [CHUNKS_FILE, VECTORS_FILE, METADATA_FILE]):
                return False
            
            # チャンク、ドキュメントソース、ファイルパスの読み込み
            data = load_pickle(CHUNKS_FILE)
            self.chunks = data['chunks']
            self.doc_sources = data['doc_sources']
            self.file_paths = data.get('file_paths', {})
            
            # ベクトルの読み込み
            self.vectors = load_pickle(VECTORS_FILE)
            
            print(f"{Colors.GREEN}保存されたベクトルとメタデータを読み込みました{Colors.END}")
            return True
            
        except Exception as e:
            print(f"{Colors.YELLOW}読み込み中にエラーが発生しました: {e}{Colors.END}")
            return False

    def ask(self, question: str) -> str:
        """質問に対する回答を生成"""
        if not self.chunks:
            return f"ドキュメントが読み込まれていません。ファイルを {DOCS_DIRECTORY} に配置してください。"
        
        try:
            relevant_chunks = self.find_relevant_chunks_hybrid(question)
            if not relevant_chunks:
                return "申し訳ありません。関連する情報が見つかりませんでした。"

            context = ""
            for chunk in relevant_chunks:
                context += f"\n\n[{chunk['source']}から抽出された情報]\n{chunk['content']}"
            
            if not hasattr(self, 'prompt_template'):
                raise ValueError("Prompt template not loaded.")

            prompt = self.prompt_template + f"""
            [参考情報]
            {context}

            [質問]
            {question}
            """
            
            if isinstance(self.chat_model, ChatGoogleGenerativeAI):
                messages = [HumanMessage(content=prompt)]
            else:
                messages = [
                    SystemMessage(content="あなたは融資業務のアシスタントです。"),
                    HumanMessage(content=prompt)
                ]
            
            response = self.chat_model.invoke(messages)
            
            reference_info = f"\n\n{Colors.BOLD}=== 参照情報の詳細 ==={Colors.END}"
            for i, chunk in enumerate(relevant_chunks, 1):
                file_path = chunk.get('file_path', '')
                source = chunk['source']
                
                reference_info += f"\n\n{Colors.YELLOW}[参照チャンク {i}]{Colors.END}\n"
                if file_path:
                    reference_info += f"ファイル: {self._create_file_link(file_path)} ({Colors.GREEN}{source}{Colors.END})\n"
                else:
                    reference_info += f"ファイル: {Colors.GREEN}{source}{Colors.END}\n"
                
                reference_info += f"{Colors.GRAY}リランキングスコア: {chunk['reranking_score']:.4f}\n"
                reference_info += f"初期類似度: {chunk['combined_similarity']:.2%}\n"
                reference_info += f"意味的類似度: {chunk['semantic_similarity']:.2%}\n"
                reference_info += f"キーワード類似度: {chunk['keyword_similarity']:.2%}{Colors.END}\n"
                reference_info += f"内容:\n{chunk['content']}\n"
                reference_info += Colors.GRAY + "-" * 80 + Colors.END
            
            return response.content + reference_info

        except Exception as e:
            print(f"{Colors.YELLOW}回答生成中にエラーが発生しました: {str(e)}{Colors.END}")
            return f"エラーが発生しました: {str(e)}"