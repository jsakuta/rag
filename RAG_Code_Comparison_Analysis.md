# RAG システムコード比較分析

## 概要
この文書は、3つのRAGシステム（RAG、rag_v1.0、RAG_yokin）の関数レベルでの詳細な比較分析を示します。

## 1. システム構成の比較

### RAG フォルダ
- **目的**: 融資業務向けQ&Aボット
- **特徴**: 複数のファイル形式（PDF、Excel）対応、複雑な階層構造
- **UI**: コンソールベースの対話型インターフェース

### rag_v1.0 フォルダ
- **目的**: 一般的な質問応答システム
- **特徴**: バッチ処理とインタラクティブモード、Streamlit UI
- **UI**: Streamlit Webアプリケーション

### RAG_yokin フォルダ
- **目的**: 預金業務向け質問応答システム
- **特徴**: シンプルな構造、Excel特化
- **UI**: Streamlit Webアプリケーション

## 2. 主要関数の比較

### 2.1 メイン実行関数

#### RAG/main.py
```python
def main():
    # Windows環境のANSIエスケープシーケンス対応
    # LoanAssistantBotの初期化
    # 対話型ループ（コンソールベース）
    # パラメータ設定機能
```

#### rag_v1.0/main.py
```python
def main():
    # 設定の初期化
    # インタラクティブモード判定
    # StreamlitまたはProcessorの起動
```

#### RAG_yokin/main.py
```python
def main():
    # 設定の初期化
    # インタラクティブモード判定
    # StreamlitまたはExcelVectorProcessorの起動
```

### 2.2 検索関数

#### RAG/bot.py
```python
def find_relevant_chunks_hybrid(self, question: str) -> List[Dict]:
    # 意味検索とキーワード検索のハイブリッド実装
    # cosine_similarity使用
    # リランキング機能付き
    # 正規化されたスコア計算
```

#### rag_v1.0/searcher.py
```python
def search(self, input_number: str, query_text: str, original_answer: str) -> list:
    # LLMによるテキスト要約
    # キーワード抽出（Sudachi使用）
    # ベクトル類似度計算
    # ハイブリッドスコア計算
```

#### RAG_yokin/search.py
```python
def _get_hybrid_search_results(self, query_text: str, summarized_text: str, 
                              reference_texts: List[str], reference_vectors: np.ndarray,
                              top_k: int = 3) -> List[Tuple[int, float]]:
    # キーワード抽出（Sudachi使用）
    # ベクトル類似度計算
    # 重み付きスコア計算
```

### 2.3 データ処理関数

#### RAG/bot.py
```python
def _load_single_pdf(self, pdf_path: str) -> None:
    # PDFからテキスト抽出
    # 階層的見出しによる分割
    # Documentオブジェクトへの変換

def _load_single_excel(self, excel_path: str) -> None:
    # Excelファイルの処理
    # チャンクへの変換
```

#### rag_v1.0/processor.py
```python
def process_data(self, mode: str = "batch"):
    # 入力データの読み込み
    # 検索準備（ベクトル化）
    # 進捗バー表示
    # 結果の保存
```

#### RAG_yokin/processor.py
```python
def process_files(self):
    # ファイルの取得と読み込み
    # 列名の検証
    # リファレンスデータのベクトル化
    # キャッシュ機能
    # Excel書式設定
```

### 2.4 キーワード抽出関数

#### RAG/utils/text.py（推定）
```python
def extract_keywords(text: str):
    # 基本的なキーワード抽出
```

#### rag_v1.0/searcher.py
```python
def _extract_keywords(self, text: str, top_k: int = 5) -> list[str]:
    # Sudachi形態素解析
    # 名詞の抽出
    # 重要度による重み付け
    # ストップワード除去
```

#### RAG_yokin/search.py
```python
def _extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
    # Sudachi形態素解析
    # 名詞の抽出（固有名詞、一般名詞）
    # TF-IDFスコア考慮
    # 位置による重み付け
```

### 2.5 類似度計算関数

#### RAG/bot.py
```python
def find_relevant_chunks_hybrid(self, question: str):
    # コサイン類似度
    # 正規化処理
    # 重み付き結合
    # リランキング
```

#### rag_v1.0/searcher.py
```python
def _calculate_keyword_similarity(self, query_keywords: list[str], reference_text: str) -> float:
    # Jaccard類似度
    # 位置による重み付け
    # 正規化処理
```

#### RAG_yokin/search.py
```python
def _calculate_keyword_similarity(self, query_keywords: List[str], reference_text: str) -> float:
    # 重み付きJaccard類似度
    # 位置による重み付け
    # 正規化処理
```

## 3. 主要な違い

### 3.1 アーキテクチャ
- **RAG**: 単一の大きなクラス（LoanAssistantBot）
- **rag_v1.0**: モジュール化されたProcessor-Searcher構造
- **RAG_yokin**: Mixinパターンを使用した階層構造

### 3.2 UI方式
- **RAG**: コンソールベースの対話型
- **rag_v1.0**: Streamlit Webアプリケーション
- **RAG_yokin**: Streamlit Webアプリケーション

### 3.3 データ処理
- **RAG**: PDF/Excel両対応、階層的見出し処理
- **rag_v1.0**: 汎用的なデータ処理、進捗表示
- **RAG_yokin**: Excel特化、キャッシュ機能

### 3.4 検索機能
- **RAG**: リランキング付きハイブリッド検索
- **rag_v1.0**: LLM要約 + ハイブリッド検索
- **RAG_yokin**: シンプルなハイブリッド検索

### 3.5 キーワード抽出
- **RAG**: 基本的な抽出
- **rag_v1.0**: Sudachi + 重要度重み付け
- **RAG_yokin**: Sudachi + TF-IDF + 位置重み付け

### 3.6 設定管理
- **RAG**: config.pyでの定数定義
- **rag_v1.0**: SearchConfigクラス
- **RAG_yokin**: SearchConfigクラス

## 4. 技術的特徴

### 4.1 使用ライブラリ
| 機能 | RAG | rag_v1.0 | RAG_yokin |
|------|-----|----------|-----------|
| 形態素解析 | - | Sudachi | Sudachi |
| UI | Console | Streamlit | Streamlit |
| 進捗表示 | - | tqdm | - |
| ベクトル化 | sentence-transformers | sentence-transformers | sentence-transformers |
| LLM | Anthropic/OpenAI/Gemini | Anthropic/OpenAI | Anthropic/OpenAI |

### 4.2 キャッシュ機能
- **RAG**: pickle形式でのベクトルキャッシュ
- **rag_v1.0**: JSON形式でのベクトルキャッシュ
- **RAG_yokin**: JSON形式でのベクトルキャッシュ

### 4.3 出力形式
- **RAG**: コンソール出力 + ファイルリンク
- **rag_v1.0**: Excel出力 + Streamlit表示
- **RAG_yokin**: Excel出力 + Streamlit表示

## 5. 進化の流れ

1. **RAG**: 基本的なRAGシステム、融資業務特化
2. **rag_v1.0**: モジュール化、Streamlit UI追加、汎用化
3. **RAG_yokin**: 預金業務特化、最適化されたUI、キャッシュ改善

## 6. 推奨用途

### RAG
- 融資業務の専門知識が必要な場合
- PDFドキュメントの階層構造が重要な場合
- コンソールベースの運用が適している場合

### rag_v1.0
- 汎用的な質問応答システムが必要な場合
- バッチ処理とインタラクティブ処理の両方が必要な場合
- 進捗の可視化が重要な場合

### RAG_yokin
- 預金業務特化の質問応答が必要な場合
- Excel形式のデータ処理が中心の場合
- シンプルで高速な処理が必要な場合

## 7. 詳細なアーキテクチャ比較

### 7.1 設定管理の違い

#### RAG/config.py
```python
# 定数ベースの設定
MODEL_PROVIDER = "anthropic"
MODEL_NAME = "claude-3-5-sonnet-20240620"
EMBEDDING_PROVIDER = "sentence_transformers"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
RERANKER_MODEL = "hotchpotch/japanese-reranker-cross-encoder-large-v1"
```

#### rag_v1.0/config.py
```python
@dataclass
class SearchConfig:
    # 外部YAMLファイル対応
    def _load_external_config(self):
        # config.yamlから設定を読み込み
    
    # 入力/出力タイプの設定
    input_type: str = "excel"
    output_type: str = "excel"
    input_config: Dict[str, Any] = field(default_factory=dict)
    output_config: Dict[str, Any] = field(default_factory=dict)
```

#### RAG_yokin/config.py
```python
@dataclass
class SearchConfig:
    # シンプルな設定管理
    is_interactive: bool = False
    
    @property
    def keyword_weight(self) -> float:
        # プロパティとして計算
        return 1.0 - self.vector_weight
```

### 7.2 データハンドリングの違い

#### RAG: 直接処理
```python
def load_documents_from_directory(self) -> None:
    # 直接ファイルを読み込み処理
    documents = find_documents_recursively(DOCS_DIRECTORY)
    for file_path in documents['pdf']:
        self._load_single_pdf(file_path)
    for file_path in documents['excel']:
        self._load_single_excel(file_path)
```

#### rag_v1.0: ファクトリーパターン
```python
class InputHandlerFactory:
    @staticmethod
    def create(input_type: str, config: SearchConfig) -> InputHandler:
        if input_type == "excel":
            return ExcelInputHandler(config)

class OutputHandlerFactory:
    @staticmethod
    def create(output_type: str, config: SearchConfig) -> OutputHandler:
        if output_type == "excel":
            return ExcelOutputHandler(config)
```

#### RAG_yokin: Mixinパターン
```python
class ExcelVectorProcessor(HybridSearchMixin):
    def __init__(self, config: SearchConfig):
        super().__init__(config=config)
        # 継承による機能の統合
```

### 7.3 UI実装の違い

#### RAG: コンソールベース
```python
def main():
    while True:
        command = input(f"\n{Colors.BOLD}入力{Colors.END}: ").strip().lower()
        if command == 'quit':
            break
        elif command == 'params':
            bot.show_search_params()
        elif command == 'set':
            # パラメータ設定処理
        else:
            answer = bot.ask(command)
            print(f"\n{Colors.BOLD}回答:{Colors.END} {answer}")
```

#### rag_v1.0 & RAG_yokin: Streamlit
```python
def run_streamlit_ui():
    st.set_page_config(page_title="類似回答検索ボット", layout="wide")
    
    with st.sidebar:
        st.session_state.config.vector_weight = st.slider("ベクトルの重み", 0.0, 1.0, ...)
        st.session_state.config.top_k = st.number_input("表示する候補数", ...)
    
    with st.form(key="chat_form"):
        query = st.text_input("質問を入力してください")
        submit_button = st.form_submit_button("送信")
```

### 7.4 キャッシュ戦略の違い

#### RAG: Pickleベース
```python
def save_vectors(self) -> None:
    save_pickle({
        'chunks': self.chunks,
        'doc_sources': self.doc_sources,
        'file_paths': self.file_paths
    }, CHUNKS_FILE)
    save_pickle(self.vectors, VECTORS_FILE)
```

#### rag_v1.0: JSONベース（シンプル）
```python
def prepare_search(self, reference_data):
    cache_file = os.path.join(cache_dir, "cache.json")
    cache_data = {
        'vectors': self.reference_vectors.tolist(),
        'texts': self.reference_texts,
        'timestamp': datetime.now().isoformat()
    }
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)
```

#### RAG_yokin: JSONベース（ファイル別）
```python
def _cache_vectors(self, vectors: np.ndarray, texts: List[str], reference_file: str):
    cache_file = os.path.join(
        self.cache_dir, 
        f"cache_{os.path.splitext(os.path.basename(reference_file))[0]}.json"
    )
    cache_data = {
        'vectors': vectors.tolist(),
        'texts': texts,
        'timestamp': datetime.now().isoformat(),
        'reference_file': reference_file
    }
```

## 8. 技術的進化の詳細

### 8.1 コード品質の向上

1. **RAG**: 単一ファイルに多機能を集約
2. **rag_v1.0**: 責任分離、ファクトリーパターン導入
3. **RAG_yokin**: Mixinパターン、継承活用

### 8.2 エラーハンドリングの改善

#### RAG
```python
except Exception as e:
    print(f"{Colors.YELLOW}エラー: {e}{Colors.END}")
```

#### rag_v1.0
```python
except Exception as e:
    logger.error(f"Error processing data: {str(e)}", exc_info=True)
    raise
```

#### RAG_yokin
```python
except Exception as e:
    logger.error(f"Error processing files: {str(e)}", exc_info=True)
    raise
```

### 8.3 テストとデバッグ機能

#### RAG
- コンソール出力での確認
- ファイルリンクによるソース確認

#### rag_v1.0
- tqdmによる進捗表示
- 構造化ログ出力
- モジュール化によるテスト容易性

#### RAG_yokin
- 詳細なログ出力
- 処理サマリー表示
- キャッシュ状態の可視化

## 9. パフォーマンス特性

### 9.1 処理速度
- **RAG**: PDF処理のオーバーヘッド、リランキング処理
- **rag_v1.0**: 進捗表示付きバッチ処理
- **RAG_yokin**: キャッシュ最適化、Excel特化

### 9.2 メモリ使用量
- **RAG**: 大量のPDFデータを保持
- **rag_v1.0**: 汎用的な処理による適度な使用量
- **RAG_yokin**: 最小限のメモリ使用

### 9.3 スケーラビリティ
- **RAG**: 大規模PDF処理に適合
- **rag_v1.0**: 柔軟な拡張性
- **RAG_yokin**: 単純な処理での高速化

## 10. 使用シーンと選択基準

### 10.1 RAGを選択する場合
- 複雑な文書構造の解析が必要
- 融資業務の専門知識が重要
- コンソール環境での運用
- リランキング機能が必要

### 10.2 rag_v1.0を選択する場合
- 複数の入力/出力形式に対応したい
- 外部設定ファイルによる柔軟な設定が必要
- バッチ処理とUI処理の両方が必要
- 将来的な機能拡張を考慮

### 10.3 RAG_yokinを選択する場合
- 預金業務特化の処理が必要
- 高速で軽量な処理が求められる
- Excel形式でのデータ処理が中心
- シンプルな構成での運用

## 11. まとめ

3つのシステムは、それぞれ異なる目的と要件に最適化されており、共通の基盤技術を使用しながらも、実装方法と機能に大きな違いがあります。進化の過程で、コード品質の向上、モジュール化、パフォーマンスの最適化が行われており、用途に応じて適切なシステムを選択することが重要です。