import os
import fitz  # PyMuPDFライブラリ
import pandas as pd
from typing import List, Dict, Optional, Set, Union, NamedTuple, Tuple
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from langchain.schema import Document
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
import pymupdf4llm
from dotenv import load_dotenv
import pickle
import hashlib
import json
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
from abc import ABC, abstractmethod
import argparse
from sudachipy import tokenizer
from sudachipy import dictionary
import os
import pandas as pd
from typing import List, Dict, Optional
import time

# 環境変数の読み込み
load_dotenv()

# システム全体で使用する定数の定義
DOCS_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manuals")
MODEL_PROVIDER = "anthropic"  # 使用するモデルプロバイダー
MODEL_NAME = "claude-3-5-sonnet-20240620"  # 使用するモデル名
EMBEDDING_PROVIDER =  "openai" #"sentence_transformers"# 使用する埋め込みモデルプロバイダー
EMBEDDING_MODEL = "text-embedding-3-large"#"intfloat/multilingual-e5-base"  # 使用する埋め込みモデル名
RERANKER_MODEL = "hotchpotch/japanese-reranker-cross-encoder-large-v1"  # 使用するリランカーモデル名


# 検索パラメータのデフォルト値
DEFAULT_INITIAL_TOP_K = 10
DEFAULT_FINAL_TOP_K = 4
DEFAULT_SEMANTIC_WEIGHT = 0.7


class ExcelQAProcessor:
    """Excelファイルを使用したQ&A処理クラス"""
    
    def __init__(self, bot: 'LoanAssistantBot', input_dir: str = "input", max_rows: int = 0):
        """初期化
        
        Args:
            bot: LoanAssistantBotインスタンス
            input_dir: 入力ファイルのディレクトリ
            max_rows: 処理する最大行数（0の場合は全行処理）
        """
        self.bot = bot
        self.max_rows = max_rows
        self.input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), input_dir)
        if not os.path.exists(self.input_dir):
            os.makedirs(self.input_dir)
            print(f"Created input directory: {self.input_dir}")

    def _get_excel_files(self) -> List[str]:
        """inputディレクトリからExcelファイルを取得
        
        Returns:
            List[str]: Excelファイルのパスのリスト
        """
        excel_files = []
        for file in os.listdir(self.input_dir):
            if file.lower().endswith(('.xlsx', '.xls')):
                excel_files.append(os.path.join(self.input_dir, file))
        return sorted(excel_files)  # ファイル名でソート

    def process_files(self):
        """入力ディレクトリ内のExcelファイルを処理"""
        excel_files = self._get_excel_files()
        if not excel_files:
            print(f"No Excel files found in {self.input_dir}")
            return

        print(f"\n検索パラメータ:")
        print(f"- 初期検索件数: {self.bot.search_params.initial_top_k}")
        print(f"- 最終検索件数: {self.bot.search_params.final_top_k}")
        print(f"- 意味検索の重み: {self.bot.search_params.semantic_weight}")
        print(f"- 処理行数: {'全行' if self.max_rows == 0 else self.max_rows}")

        for file_path in excel_files:
            try:
                print(f"\nProcessing file: {file_path}")
                
                # Excelファイルの読み込み
                df = pd.read_excel(file_path)
                
                # 処理行数の制限
                if self.max_rows > 0:
                    df = df.head(self.max_rows)
                
                # C列を'Question'として処理
                df = df.rename(columns={df.columns[1]: 'Question'})
                
                print(f"処理対象行数: {len(df)}")
                
                # Q&A処理の実行
                result_df = self._process_answers(df)
                
                # 結果の保存
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                file_name = os.path.basename(file_path)
                base_name, ext = os.path.splitext(file_name)
                output_path = os.path.join(
                    self.input_dir, 
                    f"{base_name}_processed_{timestamp}{ext}"
                )
                
                result_df.to_excel(output_path, index=False)
                print(f"Results saved to: {output_path}")
                
            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")

    def _process_answers(self, df: pd.DataFrame) -> pd.DataFrame:
        """DataFrameの各行に対してQ&A処理を実行
        
        Args:
            df: 処理対象のDataFrame
        
        Returns:
            処理結果を含むDataFrame
        """
        # 必要な列が存在することを確認し、なければ追加
        required_columns = ['Question', 'Answer', 'Reference1', 'Reference2', 'Reference3', 'Reference4', 'Parameters']
        for col in required_columns:
            if col not in df.columns:
                df[col] = ''
        
        total_rows = len(df)
        
        # 各質問に対して回答を生成
        for idx, row in df.iterrows():
            question = row['Question']
            if pd.isna(question) or not question.strip():
                continue
            
            print(f"\nProcessing question {idx + 1}/{total_rows}: {question}")
            try:
                # 回答の生成
                answer = self.bot.ask(question)
                
                # 回答と参照情報を分離
                main_answer = answer.split("\n=== 参照ファイル一覧 ===")[0]
                
                # 参照チャンクの抽出
                chunks = []
                if "=== 参照情報の詳細 ===" in answer:
                    ref_section = answer.split("=== 参照情報の詳細 ===")[1]
                    chunk_sections = ref_section.split("-" * 80)
                    
                    for section in chunk_sections:
                        if "内容:" in section:
                            chunk_content = section.split("内容:")[1].strip()
                            if chunk_content:
                                chunks.append(chunk_content)
                
                # パラメータ情報の作成
                params = f"初期検索件数: {self.bot.search_params.initial_top_k}, "
                params += f"最終検索件数: {self.bot.search_params.final_top_k}, "
                params += f"意味検索の重み: {self.bot.search_params.semantic_weight}"
                
                # 結果をDataFrameに格納
                df.at[idx, 'Answer'] = main_answer
                
                # 参照チャンクを格納
                for i, chunk in enumerate(chunks[:4], 1):
                    df.at[idx, f'Reference{i}'] = chunk
                
                # パラメータ情報を格納
                df.at[idx, 'Parameters'] = params
                
            except Exception as e:
                print(f"Error processing question {idx + 1}: {str(e)}")
                df.at[idx, 'Answer'] = f"エラーが発生しました: {str(e)}"
                continue
            
            # 進捗表示を追加
            if (idx + 1) % 5 == 0:
                print(f"Progress: {idx + 1}/{total_rows} questions processed")
        
        return df


@dataclass
class SearchParameters:
    """検索パラメータを管理するデータクラス"""

    initial_top_k: int
    final_top_k: int
    semantic_weight: float

    @classmethod
    def from_input(cls) -> "SearchParameters":
        """ユーザー入力から検索パラメータを設定"""
        print("\n検索パラメータの設定")
        print("-" * 50)

        while True:
            try:
                initial_top_k = int(
                    input(f"初期検索件数 (デフォルト: {DEFAULT_INITIAL_TOP_K}): ")
                    or DEFAULT_INITIAL_TOP_K
                )
                if initial_top_k < 1:
                    print("初期検索件数は1以上の値を指定してください。")
                    continue

                final_top_k = int(
                    input(f"最終検索件数 (デフォルト: {DEFAULT_FINAL_TOP_K}): ")
                    or DEFAULT_FINAL_TOP_K
                )
                if final_top_k < 1 or final_top_k > initial_top_k:
                    print(
                        "最終検索件数は1以上、かつ初期検索件数以下の値を指定してください。"
                    )
                    continue

                semantic_weight = float(
                    input(
                        f"意味検索の重み (0-1の範囲, デフォルト: {DEFAULT_SEMANTIC_WEIGHT}): "
                    )
                    or DEFAULT_SEMANTIC_WEIGHT
                )
                if not 0 <= semantic_weight <= 1:
                    print("意味検索の重みは0から1の間で指定してください。")
                    continue

                return cls(
                    initial_top_k=initial_top_k,
                    final_top_k=final_top_k,
                    semantic_weight=semantic_weight,
                )

            except ValueError as e:
                print(f"不正な入力です: {e}")
                print("もう一度入力してください。")

    def to_dict(self) -> Dict:
        """パラメータを辞書形式に変換"""
        return {
            "initial_top_k": self.initial_top_k,
            "final_top_k": self.final_top_k,
            "semantic_weight": self.semantic_weight,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SearchParameters":
        """辞書形式のデータからSearchParametersインスタンスを生成"""
        return cls(
            initial_top_k=data.get("initial_top_k", DEFAULT_INITIAL_TOP_K),
            final_top_k=data.get("final_top_k", DEFAULT_FINAL_TOP_K),
            semantic_weight=data.get("semantic_weight", DEFAULT_SEMANTIC_WEIGHT),
        )


@dataclass
class DocumentMetadata:
    """ドキュメントのメタデータを管理するデータクラス"""

    files: List[str]
    last_modified_times: Dict[str, float]
    total_chunks: int
    model_name: str

    def to_dict(self):
        """メタデータを辞書形式に変換"""
        return {
            "files": self.files,
            "last_modified_times": self.last_modified_times,
            "total_chunks": self.total_chunks,
            "model_name": self.model_name,
        }

    @classmethod
    def from_dict(cls, data):
        """辞書形式のデータからDocumentMetadataインスタンスを生成"""
        return cls(
            files=data["files"],
            last_modified_times=data["last_modified_times"],
            total_chunks=data["total_chunks"],
            model_name=data["model_name"],
        )


class EmbeddingModel(ABC):
    """埋め込みモデルの抽象基底クラス"""

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """テキストを埋め込みベクトルに変換する

        Args:
            text: 入力テキスト

        Returns:
            List[float]: 埋め込みベクトル
        """
        pass

    @abstractmethod
    def get_vector_dimension(self) -> int:
        """埋め込みベクトルの次元数を返す

        Returns:
            int: ベクトルの次元数
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """モデル名を返す

        Returns:
            str: モデル名
        """
        pass


class OpenAIEmbeddingModel(EmbeddingModel):
    """OpenAI APIを使用した埋め込みモデル"""

    def __init__(self, model_name: str = "text-embedding-3-small"):
        """初期化

        Args:
            model_name: OpenAIの埋め込みモデル名
        """
        self._model = OpenAIEmbeddings(model=model_name)
        self._model_name = model_name
        # ベクトルの次元数はモデルによって異なる
        self._dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }

    def embed_query(self, text: str) -> List[float]:
        return self._model.embed_query(text)

    def get_vector_dimension(self) -> int:
        return self._dimensions.get(self._model_name, 1536)

    @property
    def model_name(self) -> str:
        return self._model_name


class SentenceTransformerEmbeddingModel(EmbeddingModel):
    """SentenceTransformersを使用した埋め込みモデル"""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-base"):
        """初期化

        Args:
            model_name: SentenceTransformersのモデル名
        """
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name

    def embed_query(self, text: str) -> List[float]:
        # クエリの場合はprefixを追加
        if not text.startswith(("query:", "passage:")):
            text = f"query: {text}"
        embeddings = self._model.encode([text], normalize_embeddings=True)
        return embeddings[0].tolist()

    def get_vector_dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    @property
    def model_name(self) -> str:
        return self._model_name


class RerankerModel:
    """クロスエンコーダーを使用したリランキングモデル"""

    def __init__(self, model_name: str = RERANKER_MODEL):
        """初期化

        Args:
            model_name: クロスエンコーダーモデル名
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(model_name, max_length=512, device=self.device)

        # GPUが利用可能な場合は半精度に変換
        if self.device == "cuda":
            self.model.model.half()

    def rerank(self, query: str, passages: List[Dict]) -> List[Dict]:
        """パッセージをリランキング

        Args:
            query: 検索クエリ
            passages: リランキングするパッセージのリスト

        Returns:
            リランキングされたパッセージのリスト
        """
        # クエリとパッセージのペアを作成
        pairs = [(query, passage["content"]) for passage in passages]

        # リランキングスコアを計算
        scores = self.model.predict(pairs)

        # スコアとパッセージを組み合わせてソート
        scored_passages = [(score, passage) for score, passage in zip(scores, passages)]
        scored_passages.sort(reverse=True, key=lambda x: x[0])

        # リランキングスコアを追加して結果を返す
        reranked_passages = []
        for score, passage in scored_passages:
            passage_with_score = passage.copy()
            passage_with_score["reranking_score"] = float(
                score
            )  # Convert torch.Tensor to float if necessary
            reranked_passages.append(passage_with_score)

        return reranked_passages


class LoanAssistantBot:
    """融資業務Q&Aボットの主要クラス"""

    def __init__(
        self,
        model_provider: str = MODEL_PROVIDER,
        model_name: str = MODEL_NAME,
        embedding_provider: str = EMBEDDING_PROVIDER,
        embedding_model_name: str = EMBEDDING_MODEL,
        reranker_model_name: str = RERANKER_MODEL,
        prompt_path: str = None,
        search_params: Optional[SearchParameters] = None,
    ):
        """初期化

        Args:
            model_provider: 使用するAIモデルのプロバイダー名
            model_name: 使用するモデル名
            embedding_provider: 使用する埋め込みモデルのプロバイダー名
            embedding_model_name: 使用する埋め込みモデル名
            reranker_model_name: 使用するリランカーモデル名
            prompt_path: プロンプトテンプレートファイルのパス
            search_params: 検索パラメータ
        """
        self.search_params = search_params or SearchParameters(
            initial_top_k=DEFAULT_INITIAL_TOP_K,
            final_top_k=DEFAULT_FINAL_TOP_K,
            semantic_weight=DEFAULT_SEMANTIC_WEIGHT,
        )

        # ドキュメント保存用ディレクトリの作成
        if not os.path.exists(DOCS_DIRECTORY):
            os.makedirs(DOCS_DIRECTORY)
            print(f"Created directory: {DOCS_DIRECTORY}")

        # AIモデルと埋め込みモデルの初期化
        self.chat_model = self._create_chat_model(model_provider, model_name)
        self.embeddings = self._create_embedding_model(
            embedding_provider, embedding_model_name
        )
        self.reranker = RerankerModel(reranker_model_name)

        # データ保持用の変数初期化
        self.chunks = []
        self.vectors = []
        self.doc_sources = {}

        # プロンプトテンプレートの読み込み
        if prompt_path:
            self.prompt_template = self._load_prompt_template(prompt_path)

        # キャッシュディレクトリとファイルパスの設定
        self.cache_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "cache"
        )
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        self.vectors_file = os.path.join(self.cache_dir, "vectors.pkl")
        self.chunks_file = os.path.join(self.cache_dir, "chunks.pkl")
        self.metadata_file = os.path.join(self.cache_dir, "metadata.json")
        self.params_file = os.path.join(self.cache_dir, "search_params.json")

    def _create_chat_model(self, model_provider: str, model_name: str):
        """チャットモデルを作成"""
        if model_provider == "anthropic":
            return ChatAnthropic(
                anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
                model=model_name,
                temperature=0,
            )
        elif model_provider == "openai":
            return ChatOpenAI(
                openai_api_key=os.environ.get("OPENAI_API_KEY"),
                model_name=model_name,
                temperature=0,
            )
        elif model_provider == "gemini":
            return ChatGoogleGenerativeAI(
                api_key=os.environ.get("GOOGLE_API_KEY"),
                model=model_name,
                temperature=0,
            )
        else:
            raise ValueError(f"Unsupported model provider: {model_provider}")

    def _create_embedding_model(self, provider: str, model_name: str) -> EmbeddingModel:
        """埋め込みモデルを作成

        Args:
            provider: プロバイダー名
            model_name: モデル名

        Returns:
            EmbeddingModel: 埋め込みモデルのインスタンス

        Raises:
            ValueError: サポートされていないプロバイダーが指定された場合
        """
        if provider == "openai":
            return OpenAIEmbeddingModel(model_name)
        elif provider == "sentence_transformers":
            return SentenceTransformerEmbeddingModel(model_name)
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

    def _load_prompt_template(self, prompt_path: str) -> str:
        """プロンプトテンプレートファイルを読み込む"""
        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"Prompt template file not found at: {prompt_path}")

        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def vectorize_chunks(self) -> None:
        """チャンクをベクトル化"""
        print(f"Vectorizing {len(self.chunks)} chunks...")
        self.vectors = []

        for i, chunk in enumerate(self.chunks):
            try:
                # テキストを埋め込みベクトルに変換
                embedding = self.embeddings.embed_query(chunk.page_content)
                self.vectors.append(embedding)

                # 進捗表示
                if (i + 1) % 10 == 0:
                    print(f"Vectorized {i + 1}/{len(self.chunks)} chunks")

            except Exception as e:
                print(f"Error vectorizing chunk {i}: {str(e)}")
                # エラー時は0ベクトルを追加
                self.vectors.append([0.0] * self.embeddings.get_vector_dimension())

    def _get_files_metadata(self) -> DocumentMetadata:
        """現在のドキュメントディレクトリの状態を取得"""
        files = []
        last_modified_times = {}

        # 再帰的にファイルを検索
        documents = self._find_documents_recursively(DOCS_DIRECTORY)

        # PDFとExcelファイルの情報を収集
        for file_paths in documents.values():
            for file_path in file_paths:
                relative_path = self._get_relative_path(file_path)
                files.append(relative_path)
                last_modified_times[relative_path] = os.path.getmtime(file_path)

        return DocumentMetadata(
            files=sorted(files),
            last_modified_times=last_modified_times,
            total_chunks=len(self.chunks) if self.chunks else 0,
            model_name=self.embeddings.model_name,
        )

    def _find_documents_recursively(self, directory: str) -> Dict[str, List[str]]:
        """指定されたディレクトリ以下のPDFとExcelファイルを再帰的に検索"""
        documents = {"pdf": [], "excel": []}

        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                lower_file = file.lower()

                if lower_file.endswith(".pdf"):
                    documents["pdf"].append(file_path)
                elif lower_file.endswith((".xlsx", ".xls")):
                    documents["excel"].append(file_path)

        return documents

    def _get_relative_path(self, file_path: str) -> str:
        """DOCS_DIRECTORYからの相対パスを取得"""
        return os.path.relpath(file_path, DOCS_DIRECTORY)

    

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
        
        def is_indented_major_header(line: str) -> bool:
            """インデントされた見出しかどうかを判定"""
            stripped = line.lstrip()
            major_header_patterns = [
                r"^\d+[．\.](?!.*[（\(]注[）\)]).*",  # 例: "1．" または "1." ただし（注）を含まない
                r"^第\d+章\s+.*",          # 例: "第1章 "
                r"^第\d+節\s+.*",          # 例: "第1節 "
            ]
            return any(re.match(pattern, stripped) for pattern in major_header_patterns)

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

    def _load_single_excel_hierarchical(self, excel_path: str) -> None:
        """階層構造を持つExcelファイルを相対パスでチャンク化"""
        relative_path = self._get_relative_path(excel_path)
        df = pd.read_excel(excel_path)

        level_columns = [col for col in df.columns if col.startswith("Lv")]
        content_column = None

        for i in range(len(level_columns) - 1, 0, -1):
            if df[level_columns[i]].isna().all():
                content_column = level_columns[i - 1]
                break

        if content_column is None:
            content_column = level_columns[-1]

        print(f"Content column identified: {content_column} in {relative_path}")

        content_column_index = level_columns.index(content_column)
        hierarchy_columns = level_columns[:content_column_index]
        hierarchy = {level: None for level in hierarchy_columns}

        start_index = len(self.chunks)
        current_chunks = []

        for idx, row in df.iterrows():
            for level in hierarchy_columns:
                if pd.notna(row[level]) and row[level]:
                    hierarchy[level] = row[level]
                    level_num = int(level.replace("Lv", ""))
                    for lower_level in hierarchy_columns:
                        if int(lower_level.replace("Lv", "")) > level_num:
                            hierarchy[lower_level] = None

            path_elements = [v for v in hierarchy.values() if pd.notna(v) and v]
            if pd.notna(row[content_column]):
                path_elements.append(str(row[content_column]))
            path_text = " > ".join(path_elements)

            if path_text:
                chunk = Document(
                    page_content=path_text,
                    metadata={
                        "source": relative_path,
                        "hierarchy": {
                            k: v for k, v in hierarchy.items() if v is not None
                        },
                    },
                )
                self.chunks.append(chunk)
                self.doc_sources[start_index + len(current_chunks)] = relative_path
                current_chunks.append(chunk)

        print(f"Added {len(current_chunks)} path-based chunks from {relative_path}")

    def _load_single_pdf(self, pdf_path: str) -> None:
        """PDFファイルを階層的な見出しごとにチャンク化"""
        relative_path = self._get_relative_path(pdf_path)
        md_text = pymupdf4llm.to_markdown(pdf_path, page_chunks=False)
        
        if isinstance(md_text, list):
            full_text = "\n".join(md_text)
        else:
            full_text = str(md_text)

        try:
            chunks = self.split_by_hierarchical_headers(full_text)
            start_index = len(self.chunks)
            
            for i, chunk_data in enumerate(chunks):
                if chunk_data["text"].strip():
                    chunk = Document(
                        page_content=f"【見出し】{chunk_data['header']}\n\n{chunk_data['text']}",
                        metadata={
                            "header": chunk_data["header"],
                            "source": relative_path,
                        },
                    )
                    self.chunks.append(chunk)
                    self.doc_sources[start_index + i] = relative_path
            
            print(f"Added {len(chunks)} hierarchical header-based chunks from {relative_path}")
        except Exception as e:
            print(f"Error processing text from {relative_path}: {str(e)}")
            # フォールバック処理
            try:
                chunk = Document(
                    page_content=full_text,
                    metadata={"source": relative_path}
                )
                self.chunks.append(chunk)
                self.doc_sources[len(self.chunks) - 1] = relative_path
                print(f"Added single chunk from {relative_path} using fallback method")
            except Exception as e2:
                print(f"Fallback also failed for {relative_path}: {str(e2)}")

    def _should_update_vectors(self) -> bool:
        """ベクトルの更新が必要かどうかを判定"""
        if not os.path.exists(self.metadata_file):
            return True

        try:
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                stored_metadata = DocumentMetadata.from_dict(json.load(f))

            current_metadata = self._get_files_metadata()

            return (
                stored_metadata.files != current_metadata.files
                or stored_metadata.last_modified_times
                != current_metadata.last_modified_times
                or stored_metadata.model_name != current_metadata.model_name
            )

        except Exception as e:
            print(f"メタデータの検証中にエラーが発生しました: {e}")
            return True

    def save_vectors(self) -> None:
        """チャンクとベクトルをファイルに保存"""
        try:
            with open(self.chunks_file, "wb") as f:
                pickle.dump({"chunks": self.chunks, "doc_sources": self.doc_sources}, f)

            with open(self.vectors_file, "wb") as f:
                pickle.dump(self.vectors, f)

            metadata = self._get_files_metadata()
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata.to_dict(), f, ensure_ascii=False, indent=2)

            print("ベクトルとチャンクを保存しました")

        except Exception as e:
            print(f"ベクトルの保存中にエラーが発生しました: {e}")

    def load_vectors(self) -> bool:
        """保存されたチャンクとベクトルを読み込み"""
        try:
            if not all(
                os.path.exists(f)
                for f in [self.chunks_file, self.vectors_file, self.metadata_file]
            ):
                return False

            with open(self.chunks_file, "rb") as f:
                data = pickle.load(f)
                self.chunks = data["chunks"]
                self.doc_sources = data["doc_sources"]

            with open(self.vectors_file, "rb") as f:
                self.vectors = pickle.load(f)

            print("保存されたベクトルとチャンクを読み込みました")
            return True

        except Exception as e:
            print(f"ベクトルの読み込み中にエラーが発生しました: {e}")
            return False

    def initialize(self) -> None:
        """ボットの初期化処理を実行"""
        if self._should_update_vectors() or not self.load_vectors():
            print("ドキュメントの変更を検出したため、ベクトルを更新します...")
            self.load_documents_from_directory()
            if self.chunks:
                self.vectorize_chunks()
                self.save_vectors()
        else:
            print("キャッシュされたベクトルを使用します")

    def load_documents_from_directory(self) -> None:
        """ディレクトリから文書を再帰的に読み込む"""
        self.chunks = []
        self.doc_sources = {}

        documents = self._find_documents_recursively(DOCS_DIRECTORY)
        print(
            f"Found {len(documents['pdf'])} PDF files and {len(documents['excel'])} Excel files"
        )

        for file_path in documents["pdf"]:
            relative_path = self._get_relative_path(file_path)
            try:
                self._load_single_pdf(file_path)
                print(f"Successfully loaded PDF: {relative_path}")
            except Exception as e:
                print(f"Error loading PDF {relative_path}: {str(e)}")

        for file_path in documents["excel"]:
            relative_path = self._get_relative_path(file_path)
            try:
                self._load_single_excel_hierarchical(file_path)
                print(f"Successfully loaded Excel: {relative_path}")
            except Exception as e:
                print(f"Error loading Excel {relative_path}: {str(e)}")

    def _load_important_terms(self) -> Set[str]:
        """専門用語をJSONファイルから読み込む"""
        terms_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "terms", "finance_terms.json"
        )
        try:
            with open(terms_path, "r", encoding="utf-8") as f:
                terms_data = json.load(f)

            # 全てのタイプの用語を統合
            all_terms = set()
            for term_category in terms_data.values():
                all_terms.update(term_category)
            return all_terms

        except Exception as e:
            print(f"専門用語の読み込み中にエラーが発生しました: {e}")

    def _extract_keywords(
        self, text: str, context_window: int = 5
    ) -> List[Tuple[str, float]]:
        """文脈を考慮してテキストからキーワードを抽出

        Args:
            text: 入力テキスト
            context_window: 文脈を考慮する単語数

        Returns:
            List[Tuple[str, float]]: (キーワード, 重要度) のリスト
        """
        # 専門用語を読み込み
        important_terms = self._load_important_terms()

        # ストップワードの定義
        stop_words = {
            # 一般的な助詞
            "は",
            "が",
            "の",
            "に",
            "を",
            "で",
            "へ",
            "と",
            "から",
            "より",
            "まで",
            # 助動詞
            "です",
            "ます",
            "でした",
            "ました",
            "である",
            "だ",
            "な",
            "なる",
            # 接続詞
            "そして",
            "また",
            "および",
            "かつ",
            "ならびに",
            "でも"
            # 疑問表現
            "ください",
            "教えて",
            "下さい",
            "思います",
            # 指示語
            "これ",
            "それ",
            "あれ",
            "この",
            "その",
            "あの",
            "ここ",
            "そこ",
            "あそこ",
            "こちら",
            # その他不要語
            "について",
            "における",
            "による",
            "ついて",
            "おける",
            "よる",
            "ござい",
            "申し訳",
            "する",
            "ため",
            "場合",
            "こと",
            "よろしい",
            "もの",
            "ところ"
        }

        # テキストの前処理
        text = text.replace("？", "。").replace("！", "。")
        text = re.sub(r"【.*?】", "", text)  # 【】で囲まれた部分を除去
        text = re.sub(r"[、。,.!?:;\(\)\[\]{}]", " ", text)
        text = re.sub(r"\s+", " ", text)

        # 形態素解析の実行
        tokenizer_obj = dictionary.Dictionary().create()
        mode = tokenizer.Tokenizer.SplitMode.C  # モードはA,B,C
        tokens = tokenizer_obj.tokenize(text, mode)

        # 有効な単語の抽出
        valid_words = []
        for token in tokens:
            word = token.surface()
            pos = token.part_of_speech()[0]

            # 数字のみの単語は除外
            if word.isdigit():
                continue

            # 2文字以上で品詞が名詞・動詞・形容詞、かつストップワードでない単語を抽出
            if (
                len(word) >= 2
                and pos in ["名詞", "動詞", "形容詞"]
                and word.lower() not in stop_words
                and not word.isspace()
            ):
                valid_words.append((word, pos))

            # 重要な複合語を抽出
            if len(word) >= 4:
                for term in important_terms:
                    if term in word and word != term:
                        valid_words.append((word, pos))

        # キーワードごとの重要度を計算
        keyword_scores = []
        for i, (word, pos) in enumerate(valid_words):
            # 1. 位置による重要度（文頭に近いほど重要）
            position_weight = 1.0 - (i / len(valid_words))

            # 2. 品詞による重み付け
            pos_weights = {"名詞": 2.0, "動詞": 0.2, "形容詞": 0.1}
            pos_weight = pos_weights.get(pos, 0.5)

            # 3. 重要語句との一致
            term_importance = 1.5 if word in important_terms else 1.0

            # 4. 共起語の分析
            start_idx = max(0, i - context_window)
            end_idx = min(len(valid_words), i + context_window + 1)
            context_words = set(
                w for w, _ in valid_words[start_idx:i] + valid_words[i + 1 : end_idx]
            )

            # 重要語句との共起スコアを計算
            context_importance = sum(1.0 for w in context_words if w in important_terms)
            context_score = (len(context_words) + context_importance) / (
                2 * context_window
            )

            # 5. 文字長による重み付け（より長い複合語を優先）
            length_weight = min(len(word) / 10.0, 1.5)

            # 6. 質問文中の疑問詞に関連する単語の重要度を上げる
            question_weight = (
                1.1 if re.search(r"(どの|何|いつ|どう|どちら|いかが)", text) else 1.0
            )

            # 7. 数値を含む表現の重要度を上げる
            numeric_weight = 1.1 if re.search(r"\d+", word) else 1.0

            # 最終スコアの計算
            final_score = (
                (
                    position_weight * 0.1
                    + pos_weight * 0.2
                    + term_importance * 0.25
                    + context_score * 0.15
                    + length_weight * 0.2
                )* question_weight* numeric_weight
            )

            keyword_scores.append((word, final_score))

        # スコアでソートして返す
        scored_keywords = sorted(keyword_scores, key=lambda x: x[1], reverse=True)

        # 重複を除去（同じ単語が含まれる場合は最もスコアが高いものを残す）
        seen = set()
        unique_keywords = []
        for word, score in scored_keywords:
            if not any(word in seen_word or seen_word in word for seen_word in seen):
                seen.add(word)
                unique_keywords.append((word, score))

        # 表示するキーワード数を調整（最大10件）
        return unique_keywords[:10]

    def find_relevant_chunks_hybrid(self, question: str) -> List[Dict]:
        """文脈を考慮した意味検索とキーワード検索のハイブリッド検索"""
        params = self.search_params
        if not 0 <= params.semantic_weight <= 1:
            raise ValueError("semantic_weight must be between 0 and 1")

        keyword_weight = 1.0 - params.semantic_weight

        # 意味的検索スコアの計算
        question_embedding = self.embeddings.embed_query(question)
        semantic_scores = [
            float(cosine_similarity([question_embedding], [vector])[0][0])
            for vector in self.vectors
        ]
            
        # 文脈を考慮したキーワード検索
        question_keywords = self._extract_keywords(question)
        print("\n=== 検索キーワード（重要度順）===")
        for keyword, score in question_keywords:  # すべてのキーワードを表示
            print(f"キーワード: {keyword} (重要度: {score:.4f})")
        print("=" * 40)

        # 各チャンクのキーワードスコアを計算
        keyword_scores = []
        for chunk in self.chunks:
            chunk_keywords = self._extract_keywords(chunk.page_content)
            chunk_keywords_dict = dict(chunk_keywords)

            # キーワードマッチングスコアの計算
            score = 0.0
            matches = 0
            for q_keyword, q_score in question_keywords:
                for c_keyword, c_score in chunk_keywords:
                    if (
                        q_keyword.lower() in c_keyword.lower()
                        or c_keyword.lower() in q_keyword.lower()
                    ):
                        # キーワードの重要度を考慮したスコア
                        score += q_score * c_score
                        matches += 1
                        break

            if matches > 0:
                # スコアを正規化（マッチした数で割る）
                score = score / matches

            keyword_scores.append(score)

        # スコアの正規化
        max_semantic = max(semantic_scores) if semantic_scores else 1.0
        max_keyword = max(keyword_scores) if keyword_scores else 1.0

        if max_semantic == 0:
            semantic_scores = [1.0] * len(semantic_scores)
        else:
            semantic_scores = [s / max_semantic for s in semantic_scores]

        if max_keyword == 0:
            keyword_scores = [1.0] * len(keyword_scores)
        else:
            keyword_scores = [k / max_keyword for k in keyword_scores]

        # ハイブリッドスコアの計算
        combined_scores = [
            (params.semantic_weight * s) + (keyword_weight * k)
            for s, k in zip(semantic_scores, keyword_scores)
        ]

        if params.initial_top_k == 0:
            return []

        # 初期の候補を選択
        top_indices = np.argsort(combined_scores)[-params.initial_top_k :][::-1]

        initial_results = []
        for idx in top_indices:
            initial_results.append(
                {
                    "content": self.chunks[idx].page_content,
                    "source": self.doc_sources.get(idx, "Unknown Source"),
                    "combined_similarity": combined_scores[idx],
                    "semantic_similarity": semantic_scores[idx],
                    "keyword_similarity": keyword_scores[idx],
                }
            )

        # リランキングを適用
        reranked_results = self.reranker.rerank(question, initial_results)

        # 最終的な結果を返す
        return reranked_results[: params.final_top_k]

    def save_search_params(self) -> None:
        """検索パラメータをファイルに保存"""
        try:
            with open(self.params_file, "w", encoding="utf-8") as f:
                json.dump(self.search_params.to_dict(), f, ensure_ascii=False, indent=2)
            print("検索パラメータを保存しました")
        except Exception as e:
            print(f"検索パラメータの保存中にエラーが発生しました: {e}")

    def load_search_params(self) -> bool:
        """保存された検索パラメータを読み込み"""
        try:
            if not os.path.exists(self.params_file):
                return False

            with open(self.params_file, "r", encoding="utf-8") as f:
                params_dict = json.load(f)
                self.search_params = SearchParameters.from_dict(params_dict)
            print("保存された検索パラメータを読み込みました")
            return True
        except Exception as e:
            print(f"検索パラメータの読み込み中にエラーが発生しました: {e}")
            return False

    def ask(self, question: str) -> str:
        """質問に対する回答を生成"""
        if not self.chunks:
            return f"ドキュメントが読み込まれていません。ファイルを {DOCS_DIRECTORY} に配置してください。"

        # キーワード表示部分を削除し、直接find_relevant_chunks_hybridを呼び出す
        relevant_chunks = self.find_relevant_chunks_hybrid(question)

        # 参照ファイルの一覧を作成（重複を除去）
        referenced_files = {chunk["source"] for chunk in relevant_chunks}

        # ファイルリンクのセクションを作成
        file_links = "\n=== 参照ファイル一覧 ===\n"
        for source in sorted(referenced_files):
            # DOCSディレクトリの絶対パスを使用
            full_path = os.path.join(DOCS_DIRECTORY, source).replace(
                "/", "\\"
            )  # Windowsパス形式に変換
            file_links += f"ファイル: {source}\n"
            file_links += f"パス: {full_path}\n\n"

        # コンテキストの作成
        context = ""
        for chunk in relevant_chunks:
            context += f"\n\n[{chunk['source']}から抽出された情報]\n{chunk['content']}"

        if hasattr(self, "prompt_template"):
            prompt = (
                self.prompt_template
                + f"""
            [参考情報]
            {context}

            [質問]
            {question}
            """
            )
        else:
            raise ValueError(
                "Prompt template not loaded. Please provide a prompt template file path when initializing the bot."
            )

        if isinstance(self.chat_model, ChatGoogleGenerativeAI):
            messages = [HumanMessage(content=prompt)]
        else:
            messages = [SystemMessage(content=""), HumanMessage(content=prompt)]

        response = self.chat_model.invoke(messages)

        # 参照情報の詳細を作成
        reference_info = f"\n{file_links}\n=== 参照情報の詳細 ==="
        for i, chunk in enumerate(relevant_chunks, 1):
            # DOCSディレクトリの絶対パスを使用
            full_path = os.path.join(DOCS_DIRECTORY, chunk["source"]).replace(
                "/", "\\"
            )  # Windowsパス形式に変換
            reference_info += f"\n\n[参照チャンク {i}]\n"
            reference_info += f"ファイル: {chunk['source']}\n"
            reference_info += f"パス: {full_path}\n"
            reference_info += f"リランキングスコア: {chunk['reranking_score']:.4f}\n"
            reference_info += f"初期類似度: {chunk['combined_similarity']:.2%}\n"
            reference_info += f"意味的類似度: {chunk['semantic_similarity']:.2%}\n"
            reference_info += f"キーワード類似度: {chunk['keyword_similarity']:.2%}\n"
            reference_info += f"内容:\n{chunk['content']}\n"
            reference_info += "-" * 80

        return response.content + reference_info


def main():
    """メイン実行関数"""
    prompt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", "v2.txt")
    
    parser = argparse.ArgumentParser(description='融資業務Q&Aボット')
    parser.add_argument('--initial-top-k', type=int, help=f'初期検索件数 (デフォルト: {DEFAULT_INITIAL_TOP_K})')
    parser.add_argument('--final-top-k', type=int, help=f'最終検索件数 (デフォルト: {DEFAULT_FINAL_TOP_K})')
    parser.add_argument('--semantic-weight', type=float, help=f'意味検索の重み (0-1の範囲, デフォルト: {DEFAULT_SEMANTIC_WEIGHT})')
    args = parser.parse_args()
    
    try:
        # 処理行数の入力
        while True:
            try:
                max_rows = int(input("処理する行数を入力してください（0を入力すると全行処理）: "))
                if max_rows < 0:
                    print("0以上の数値を入力してください。")
                    continue
                break
            except ValueError:
                print("有効な数値を入力してください。")
        
        # コマンドライン引数から検索パラメータを設定
        if any([args.initial_top_k, args.final_top_k, args.semantic_weight]):
            search_params = SearchParameters(
                initial_top_k=args.initial_top_k or DEFAULT_INITIAL_TOP_K,
                final_top_k=args.final_top_k or DEFAULT_FINAL_TOP_K,
                semantic_weight=args.semantic_weight or DEFAULT_SEMANTIC_WEIGHT
            )
        else:
            # ユーザー入力で検索パラメータを設定
            search_params = SearchParameters.from_input()
        
        bot = LoanAssistantBot(
            model_provider=MODEL_PROVIDER,
            model_name=MODEL_NAME,
            embedding_provider=EMBEDDING_PROVIDER,
            embedding_model_name=EMBEDDING_MODEL,
            reranker_model_name=RERANKER_MODEL,
            prompt_path=prompt_path,
            search_params=search_params
        )
        bot.initialize()
        
        # バッチ処理の実行
        processor = ExcelQAProcessor(bot, max_rows=max_rows)
        processor.process_files()
                
    except FileNotFoundError as e:
        print(f"エラー: {e}")
        print("プロンプトテンプレートファイルが見つかりません。")
        print(f"以下のパスにプロンプトファイルを配置してください: {prompt_path}")
    except Exception as e:
        print(f"エラーが発生しました: {e}")


if __name__ == "__main__":
    main()
