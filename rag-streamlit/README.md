# RAG-Streamlit

**Streamlit UI による対話的 RAG 検索システム**

[![Python](https://img.shields.io/badge/Python-3.7+-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1.0+-green)](https://python.langchain.com/)

---

## 目次

- [他プロジェクトとの違い](#他プロジェクトとの違い)
- [概要](#概要)
- [アーキテクチャ](#アーキテクチャ)
- [クイックスタート](#クイックスタート)
- [プロジェクト構成](#プロジェクト構成)
- [使用方法](#使用方法)
- [セッション状態管理](#セッション状態管理)
- [UI コンポーネント詳細](#ui-コンポーネント詳細)
- [検索アルゴリズム](#検索アルゴリズム)
- [設定オプション](#設定オプション)
- [入出力フォーマット](#入出力フォーマット)
- [チャット履歴保存](#チャット履歴保存)
- [Docker デプロイ](#docker-デプロイ)
- [キャッシュシステム](#キャッシュシステム)
- [トラブルシューティング](#トラブルシューティング)
- [セキュリティ](#セキュリティ)
- [依存パッケージ](#依存パッケージ)

---

## 他プロジェクトとの違い

| 特徴 | rag-streamlit | rag-batch | rag-gemini |
|------|---------------|-----------|------------|
| **主な用途** | 対話的検索・デモ | バッチ処理 | 最新技術 |
| **UI** | Streamlit Web UI | CLI + Streamlit | Streamlit Web UI |
| **ベクトルDB** | JSON キャッシュ | JSON キャッシュ | ChromaDB |
| **埋め込みモデル** | multilingual-e5-base | multilingual-e5-base | Gemini Embedding |

**関連プロジェクト:**
- バッチ処理が必要な場合: [rag-batch](../rag-batch/)
- 最新技術を使いたい場合: [rag-gemini](../rag-gemini/)

---

## 概要

RAG-Streamlit は、銀行預金商品の問い合わせ対応を支援する対話型検索システムです。過去の Q&A データベースから類似の質問と回答を検索し、リアルタイムで結果を表示します。

### 主な機能

| 機能 | 説明 |
|------|------|
| **対話型 Web UI** | Streamlit ベースのチャットインターフェース |
| **ハイブリッド検索** | ベクトル類似度 + キーワード検索の組み合わせ |
| **リアルタイム設定** | 検索パラメータをスライダーで即座に調整 |
| **チャット履歴保存** | 対話履歴を Excel にエクスポート |
| **バッチ処理** | Excel ファイルの一括処理も可能 |
| **Docker 対応** | コンテナ化されたデプロイメント |

---

## アーキテクチャ

```mermaid
graph TB
    subgraph UI層
        A[Streamlit UI<br/>ui/ui.py]
        B[セッション状態<br/>チャット履歴など]
        A <--> B
    end

    subgraph 処理層
        C[入力ハンドラ<br/>Excel 読み込み]
        D[検索エンジン<br/>ハイブリッド検索]
        E[出力ハンドラ<br/>Excel 出力]
        C --> D --> E
    end

    subgraph 検索エンジン
        F[ベクトル検索<br/>SentenceTransformers]
        G[キーワード検索<br/>SudachiPy 形態素解析]
        H[スコア統合<br/>combined = vw × vec + kw × keyword]
    end

    A --> C
    D --> F
    D --> G
    F --> H
    G --> H
    H --> E
```

### UI ライフサイクル

```mermaid
sequenceDiagram
    participant User as ユーザー
    participant UI as Streamlit UI
    participant State as セッション状態
    participant Engine as 検索エンジン
    participant Cache as ベクトルキャッシュ

    User->>UI: ページアクセス
    UI->>State: initialize_session_state()
    State-->>UI: 初期状態設定完了

    User->>UI: 質問入力
    UI->>State: processing_query = True
    UI->>Engine: process_query()
    Engine->>Cache: キャッシュ確認
    alt キャッシュあり
        Cache-->>Engine: ベクトルデータ
    else キャッシュなし
        Engine->>Engine: ベクトル生成
        Engine->>Cache: キャッシュ保存
    end
    Engine->>Engine: ハイブリッド検索
    Engine-->>UI: 検索結果
    UI->>State: chat_history に追加
    State-->>UI: processing_query = False
    UI->>UI: st.rerun()
```

---

## クイックスタート

### 1. 環境構築

```bash
# リポジトリのクローン
git clone <repository-url>
cd rag-streamlit

# 仮想環境の作成・有効化
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存パッケージのインストール
pip install -r requirements.txt
```

### 2. 環境変数の設定

```bash
cp .env.example .env
```

`.env` ファイルを編集:

```env
# 必須: いずれか1つ以上
ANTHROPIC_API_KEY=sk-ant-your_anthropic_api_key
OPENAI_API_KEY=sk-your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
```

### 3. データの配置

```bash
# 参照データを配置
cp your_reference_data.xlsx reference/

# 入力データを配置（バッチモードの場合）
cp your_input_data.xlsx input/
```

### 4. 実行

**対話モード（Streamlit UI）:**

```bash
python main.py interactive
# または
streamlit run ui/ui.py
```

ブラウザで http://localhost:8501 にアクセス

**バッチモード:**

```bash
python main.py
```

---

## プロジェクト構成

```text
rag-streamlit/
├── main.py                      # エントリーポイント
├── config.py                    # 設定管理（SearchConfig dataclass）
├── requirements.txt             # Python 依存パッケージ
├── .env.example                 # 環境変数テンプレート
├── Dockerfile                   # Docker コンテナ設定
├── .dockerignore                # Docker 除外ファイル
│
├── src/
│   ├── core/
│   │   ├── processor.py         # データ処理エンジン
│   │   │   ├── ExcelVectorProcessor  # メイン処理クラス
│   │   │   ├── _load_latest_prompt() # プロンプト読み込み
│   │   │   ├── get_latest_files()    # 最新ファイル検出
│   │   │   ├── _cache_vectors()      # ベクトルキャッシュ
│   │   │   └── process_files()       # バッチ処理
│   │   │
│   │   └── search.py            # ハイブリッド検索エンジン
│   │       ├── HybridSearchMixin     # 検索ミックスイン
│   │       ├── _setup_llm()          # LLM 初期化
│   │       ├── _extract_keywords()   # キーワード抽出
│   │       └── _get_hybrid_search_results()  # 検索実行
│   │
│   ├── handlers/
│   │   └── (将来の拡張用)
│   │
│   └── utils/
│       └── utils.py             # ログ設定ユーティリティ
│
├── ui/
│   └── ui.py                    # Streamlit チャット UI
│       ├── initialize_session_state()  # セッション初期化
│       ├── format_message()            # メッセージ表示
│       ├── format_response_card()      # 結果カード表示
│       ├── process_query()             # クエリ処理
│       └── save_chat_history()         # 履歴保存
│
├── prompt/
│   └── summarize_v1.0.txt       # LLM プロンプトテンプレート
│
├── input/                       # 入力ファイル（バッチ処理用）
├── reference/                   # 参照データ
│   └── vector_cache/            # ベクトルキャッシュ（JSON）
├── output/                      # 出力ファイル
├── app.log                        # アプリケーションログ
│
└── docs/                        # ドキュメント
```

---

## 使用方法

### 対話モード（Streamlit UI）

```bash
python main.py interactive
```

**UI 構成:**

| コンポーネント | 説明 |
|---------------|------|
| **サイドバー** | 検索パラメータ設定（vector_weight, top_k） |
| **チャットエリア** | 質問入力と結果表示 |
| **保存ボタン** | チャット履歴を Excel に保存 |

初期値: vector_weight=0.7、top_k=3（`ui/ui.py` で設定）

**操作手順:**

1. サイドバーで検索パラメータを調整
2. テキストボックスに質問を入力
3. 送信ボタンをクリック
4. 類似質問と回答が表示される
5. 必要に応じて「チャット履歴を保存」

**検索結果の表示:**

```text
┌─────────────────────────────────────────────┐
│ 候補 1 (類似度: 0.9234)                     │
├─────────────────────────────────────────────┤
│ 類似質問内容:                               │
│ ATMでの現金処理時に操作を誤った場合は？     │
├─────────────────────────────────────────────┤
│ 回答:                                       │
│ 現金処理で操作を誤った場合は、以下の手順... │
└─────────────────────────────────────────────┘
```

### バッチモード

```bash
python main.py
```

**処理フロー:**

1. `input/` フォルダの最新 Excel ファイルを読み込み
2. 各質問に対してハイブリッド検索を実行
3. 結果を `output/` フォルダに Excel 形式で保存

**出力ファイル名:**

```text
{入力ファイル名}_result_batch_v{vector_weight}_k{keyword_weight}_{タイムスタンプ}.xlsx
例: input_data_result_batch_v0.9_k0.1_20250101_120000.xlsx
```

---

## セッション状態管理

Streamlit の `st.session_state` を使用して、ページリロード間でデータを保持します。

### セッション変数一覧

| 変数名 | 型 | 初期値 | 説明 |
|--------|-----|--------|------|
| `chat_history` | `list[dict]` | `[]` | チャット履歴（user/bot メッセージ） |
| `summaries` | `dict` | `{}` | LLM による要約結果のキャッシュ |
| `processing_query` | `bool` | `False` | クエリ処理中フラグ |
| `config` | `SearchConfig` | UI用デフォルト | 検索設定オブジェクト |

### 初期化コード

```python
def initialize_session_state():
    """セッションステートの初期化"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "summaries" not in st.session_state:
        st.session_state.summaries = {}
    if "processing_query" not in st.session_state:
        st.session_state.processing_query = False
    if "config" not in st.session_state:
        st.session_state.config = SearchConfig(
            top_k=3,
            model_name="intfloat/multilingual-e5-base",
            llm_provider="anthropic",
            llm_model="claude-3-sonnet-20240229",
            vector_weight=0.7,
            base_dir="."
        )
```

### chat_history の構造

```python
# ユーザーメッセージ
{
    "type": "user",
    "text": "ATMの操作方法を教えて"
}

# ボットレスポンス（検索結果）
{
    "type": "bot",
    "text": [
        {
            "Summarized_Query": "ATM操作方法",
            "Search_Result_Q": "ATMでの現金処理時に...",
            "Search_Result_A": "以下の手順で...",
            "Similarity": 0.9234,
            "Vector_Weight": 0.7,
            "Top_K": 3
        },
        # ... top_k 件分
    ]
}

# ボットレスポンス（エラー時）
{
    "type": "bot",
    "text": "エラーが発生しました: ..."
}
```

### セッションライフサイクル

```mermaid
stateDiagram-v2
    [*] --> 初期化: ページアクセス
    初期化 --> 待機: initialize_session_state()

    待機 --> 処理中: 質問送信
    処理中 --> 待機: 結果表示

    待機 --> 保存中: 履歴保存
    保存中 --> 待機: Excel 出力完了

    待機 --> [*]: ブラウザ閉じる

    note right of 処理中
        processing_query = True
        送信ボタン無効化
    end note

    note right of 保存中
        chat_history → Excel
        output/ に保存
    end note
```

---

## UI コンポーネント詳細

### コンポーネント階層

```mermaid
graph TB
    subgraph Page[ページ全体]
        subgraph Sidebar[サイドバー]
            S1[タイトル: 設定]
            S2[パラメータ調整 Expander]
            S3[ベクトルの重み Slider]
            S4[表示候補数 NumberInput]
            S5[履歴保存ボタン]
        end

        subgraph Main[メインエリア]
            M1[タイトル: 質問応答チャットボット]
            M2[チャットコンテナ]
            M3[ユーザーメッセージ]
            M4[ボットレスポンスカード]
            M5[入力フォーム]
            M6[送信ボタン]
        end
    end

    S2 --> S3
    S2 --> S4
    M2 --> M3
    M2 --> M4
    M5 --> M6
```

### format_message() - メッセージ表示

ユーザーとボットのメッセージを異なるスタイルで表示します。

```python
def format_message(message, is_user=False):
    """メッセージのスタイルを定義"""
    style = f"""
        <div style="
            display: flex;
            justify-content: {'flex-end' if is_user else 'flex-start'};
            margin: 5px 0;
        ">
            <div style="
                background-color: {'#e6f3ff' if is_user else '#f5f5f5'};
                padding: 10px 15px;
                border-radius: 15px;
                max-width: 80%;
                {'margin-left: auto;' if is_user else ''}
            ">
                {message}
            </div>
        </div>
    """
    return style
```

**スタイル仕様:**

| 要素 | ユーザー | ボット |
|------|---------|--------|
| 配置 | 右寄せ | 左寄せ |
| 背景色 | `#e6f3ff`（水色） | `#f5f5f5`（灰色） |
| 最大幅 | 80% | 80% |
| 角丸 | 15px | 15px |

### format_response_card() - 検索結果カード

検索結果を視認しやすいカード形式で表示します。

```python
def format_response_card(number, similarity, query, answer):
    """応答カードのスタイルを定義"""
    return f"""
        <div class="response-card" style="
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            <div style="...">
                候補 {number} (類似度: {similarity:.4f})
            </div>
            <div style="...">
                <div>類似質問内容:</div>
                <div>{query}</div>
            </div>
            <div style="...">
                <div>回答:</div>
                <div>{answer}</div>
            </div>
        </div>
    """
```

**カード構造:**

```text
┌─────────────────────────────────────────────┐
│ ヘッダー: 候補番号 + 類似度スコア          │
├─────────────────────────────────────────────┤
│ 類似質問セクション                          │
│   背景: #f8f9fa                            │
│   角丸: 8px                                │
│   パディング: 12px                          │
├─────────────────────────────────────────────┤
│ 回答セクション                              │
│   同上のスタイル                            │
└─────────────────────────────────────────────┘
```

### サイドバーコンポーネント

```python
with st.sidebar:
    st.title("設定")

    with st.expander("パラメータ調整", expanded=True):
        # ベクトル重みスライダー
        st.session_state.config.vector_weight = st.slider(
            "ベクトルの重み",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.config.vector_weight,
            step=0.1
        )

        # 候補数入力
        st.session_state.config.top_k = st.number_input(
            "表示する候補数",
            min_value=1,
            max_value=10,
            value=st.session_state.config.top_k,
            step=1
        )

    # 履歴保存ボタン
    if st.button("チャット履歴を保存", use_container_width=True):
        save_chat_history()
```

### カスタム CSS

```python
st.markdown("""
    <style>
    /* ボタンのホバー時の色 */
    div.stButton > button:hover {
        background-color: #007bff;
        color: white;
    }
    div.stButton > button:focus {
        background-color: #007bff;
        color: white;
    }
    /* チャット履歴保存ボタン */
    [data-baseweb="button"]:has(#save_chat_history_button) {
        background-color: #28a745 !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)
```

---

## 検索アルゴリズム

### ハイブリッド検索

```mermaid
flowchart TB
    A[質問文] --> B[LLM 要約]
    B --> C[検索クエリ生成]

    A --> D[ベクトル化]
    D --> E[コサイン類似度計算]
    E --> F[Vector Similarity]

    A --> G[形態素解析]
    G --> H[キーワード抽出]
    H --> I[Keyword Similarity]

    F --> J[Score Integration<br/>combined = vw × vec + kw × keyword]
    I --> J
    J --> K[Top-K 結果]
```

### ベクトル検索

```python
# SentenceTransformer による埋め込み
model = SentenceTransformer("intfloat/multilingual-e5-base")
query_embedding = model.encode(query, normalize_embeddings=True)

# コサイン類似度
similarity = cosine_similarity(query_embedding, doc_embeddings)
```

### キーワード検索

```python
# SudachiPy による形態素解析
tokenizer = sudachipy.Dictionary().create()
tokens = tokenizer.tokenize(text, sudachipy.SplitMode.C)

# 名詞のみ抽出（重要度重み付け）
keywords = []
for token in tokens:
    pos = token.part_of_speech()[0]
    if pos == "名詞":
        weight = 2 if "固有名詞" in token.part_of_speech() else 1
        keywords.append((token.surface(), weight))
```

### スコア統合

```python
# 重み付け統合
combined_score = (
    vector_weight * vector_similarity +
    keyword_weight * keyword_similarity
)
# デフォルト: vector_weight = 0.9, keyword_weight = 0.1
```

---

## 設定オプション

### SearchConfig (config.py)

```python
@dataclass
class SearchConfig:
    """検索設定を管理するデータクラス"""
    # デフォルト設定
    DEFAULT_TOP_K: int = 4
    DEFAULT_MODEL_NAME: str = "intfloat/multilingual-e5-base"
    DEFAULT_LLM_PROVIDER: str = "anthropic"
    DEFAULT_LLM_MODEL: str = "claude-3-5-sonnet-20241022"
    DEFAULT_VECTOR_WEIGHT: float = 0.9  # バッチ処理用
    DEFAULT_UI_VECTOR_WEIGHT: float = 0.9  # UI用

    top_k: int = DEFAULT_TOP_K
    model_name: str = DEFAULT_MODEL_NAME
    llm_provider: str = DEFAULT_LLM_PROVIDER
    llm_model: str = DEFAULT_LLM_MODEL
    vector_weight: float = DEFAULT_VECTOR_WEIGHT
    base_dir: str = "."
    is_interactive: bool = False

    @property
    def keyword_weight(self) -> float:
        """キーワード検索の重みを自動計算"""
        return 1.0 - self.vector_weight
```

### パラメータ一覧

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `top_k` | int | 4 | 返却する類似文書数 |
| `model_name` | str | intfloat/multilingual-e5-base | 埋め込みモデル |
| `vector_weight` | float | 0.9 (batch) / 0.7 (UI) | ベクトル検索の重み |
| `keyword_weight` | float | 自動計算 | 1.0 - vector_weight |
| `llm_provider` | str | anthropic | LLM プロバイダー |
| `llm_model` | str | claude-3-5-sonnet-20241022 | LLM モデル |
| `base_dir` | str | "." | 基準ディレクトリ |
| `is_interactive` | bool | False | インタラクティブモードフラグ |

### UI モード vs バッチモードの設定差異

| 設定項目 | UI モード | バッチモード | 理由 |
|---------|----------|-------------|------|
| `vector_weight` | 0.7 | 0.9 | UI では対話的に調整可能 |
| `top_k` | 3 | 4 | UI では表示量を抑える |
| `is_interactive` | True | False | モード識別 |
| LLM モデル | claude-3-sonnet | claude-3-5-sonnet | UI は速度重視 |

### 設定変更方法

**1. config.py を直接編集:**

```python
@dataclass
class SearchConfig:
    top_k: int = 4
    vector_weight: float = 0.9
    llm_provider: str = "anthropic"
    # ...
```

**2. UI からリアルタイム変更:**

サイドバーのスライダーで `vector_weight` と `top_k` を調整

**3. コード内で動的に設定:**

```python
config = SearchConfig(
    top_k=5,
    vector_weight=0.8,
    llm_provider="openai"
)
processor = ExcelVectorProcessor(config)
```

---

## 入出力フォーマット

### 入力ファイル（バッチモード）

**場所:** `input/` ディレクトリ
**形式:** Excel (.xlsx)

| 列 | 必須 | 説明 |
|----|------|------|
| 1列目 | はい | 番号/ID |
| 2列目 | はい | 質問内容 |
| 3列目 | いいえ | オリジナル回答 |

### 参照データ

**場所:** `reference/` ディレクトリ
**形式:** Excel (.xlsx)

| 列名 | 必須 | 説明 |
|------|------|------|
| 問合せ内容 | はい | 参照質問文 |
| 回答 | はい | 参照回答文 |

**重要:** 列名は完全一致が必要です（「問合せ内容」「回答」）

### 出力ファイル

**場所:** `output/` ディレクトリ
**形式:** Excel (.xlsx)

| 列名 | 説明 |
|------|------|
| # | 入力番号 |
| ユーザーの質問 | 元の質問文 |
| オリジナルの回答 | 入力ファイルの回答（あれば） |
| 検索クエリ(AI処理) | LLM で生成した検索クエリ |
| 類似質問 | 検索結果の質問 |
| 類似回答 | 検索結果の回答 |
| 類似度 | 統合類似度スコア |
| ベクトルの重み | 使用した vector_weight |
| 候補数 | top_k 値 |

---

## チャット履歴保存

### 保存処理フロー

```mermaid
flowchart TB
    A[保存ボタンクリック] --> B[save_chat_history 呼び出し]
    B --> C[chat_history をペアで処理]
    C --> D{ボットレスポンスがリストか?}
    D -->|はい| E[各検索結果を行として追加]
    D -->|いいえ| F[エラーメッセージをスキップ]
    E --> G[DataFrame 作成]
    G --> H[タイムスタンプ付きファイル名生成]
    H --> I[Excel 出力]
    I --> J[書式設定適用]
    J --> K[成功メッセージ表示]
```

### 出力 Excel 構造

```python
result_columns = [
    'Input_Number',       # 質問番号（最初の行のみ）
    'Original_Query',     # 元の質問（最初の行のみ）
    'Summarized_Query',   # LLM による要約（最初の行のみ）
    'Search_Result_Q',    # 検索結果の質問
    'Search_Result_A',    # 検索結果の回答
    'Similarity',         # 類似度スコア
    'Vector_Weight',      # ベクトルの重み
    'Top_K'               # 候補数
]
```

### ファイル名形式

```text
chat_log_result_{param_summary}_{timestamp}.xlsx

パラメータ:
  param_summary: w{vector_weight}_k{top_k}
  timestamp: %Y%m%d_%H%M%S

例: chat_log_result_w0.7_k3_20250101_143052.xlsx
```

### 保存コード例

```python
def save_chat_history():
    """チャット履歴をExcelファイルに保存"""
    result_rows = []

    # チャット履歴を2つずつ処理（user + bot のペア）
    for i in range(0, len(st.session_state.chat_history), 2):
        if i + 1 < len(st.session_state.chat_history):
            query = st.session_state.chat_history[i]["text"]
            query_number = i//2 + 1
            bot_response = st.session_state.chat_history[i + 1]["text"]

            if isinstance(bot_response, list):
                first_row = True
                for response in bot_response:
                    row = {
                        'Input_Number': query_number if first_row else '',
                        'Original_Query': query if first_row else '',
                        'Summarized_Query': response["Summarized_Query"] if first_row else '',
                        'Search_Result_Q': response["Search_Result_Q"],
                        'Search_Result_A': response["Search_Result_A"],
                        'Similarity': f"{response['Similarity']:.4f}",
                        'Vector_Weight': response['Vector_Weight'],
                        'Top_K': response['Top_K']
                    }
                    result_rows.append(row)
                    first_row = False

    # DataFrame 作成と Excel 出力
    result_df = pd.DataFrame(result_rows, columns=result_columns)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    param_summary = f"w{st.session_state.config.vector_weight:.1f}_k{st.session_state.config.top_k}"
    output_file = f"output/chat_log_result_{param_summary}_{timestamp}.xlsx"

    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        result_df.to_excel(writer, index=False)
```

---

## Docker デプロイ

### Docker ビルド

```bash
docker build -t rag-streamlit:latest .
```

### Docker 実行

**対話モード:**

```bash
docker run -p 8501:8501 \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/reference:/app/reference \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/prompt:/app/prompt \
  --env-file .env \
  rag-streamlit:latest main.py interactive
```

**バッチモード:**

```bash
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/reference:/app/reference \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/prompt:/app/prompt \
  --env-file .env \
  rag-streamlit:latest main.py
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  rag-streamlit:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8501:8501"
    volumes:
      - ./input:/app/input
      - ./reference:/app/reference
      - ./output:/app/output
      - ./prompt:/app/prompt
    env_file:
      - .env
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
```

**使用方法:**

```bash
# 起動
docker-compose up -d

# ログ確認
docker-compose logs -f

# 停止
docker-compose down
```

### Dockerfile の概要

```dockerfile
FROM python:3.11-slim
# ビルドツールのインストール
RUN apt-get update && apt-get install -y build-essential gcc curl
# 依存パッケージのインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# アプリケーションのコピー
COPY . /app
WORKDIR /app
# ポート公開とヘルスチェック
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
```

### ボリュームマウント一覧

| ボリューム | 用途 | 必須 |
|-----------|------|------|
| `/app/input` | 入力 Excel ファイル | ○（バッチ） |
| `/app/reference` | 参照データ + ベクトルキャッシュ | ○ |
| `/app/output` | 出力ファイル | ○ |
| `/app/prompt` | LLM プロンプトテンプレート | △ |

---

## キャッシュシステム

### ベクトルキャッシュ

**場所:** `reference/vector_cache/cache_{reference_file_stem}.json`

**構造:**

```json
{
  "vectors": [[0.1, 0.2, ...], ...],
  "texts": ["参照テキスト1", ...],
  "timestamp": "2025-01-01T00:00:00",
  "reference_file": "reference/example.xlsx"
}
```

### キャッシュの動作

1. **初回実行:** 参照データをベクトル化し、キャッシュに保存（30-60秒）
2. **以降の実行:** キャッシュを使用（1秒未満）
3. **参照データ変更時:** 自動的にキャッシュを再生成

### キャッシュのクリア

```bash
rm reference/vector_cache/cache_*.json
```

---

## トラブルシューティング

共通の問題については [docs/TROUBLESHOOTING.md](../docs/TROUBLESHOOTING.md) を参照してください。

### エラー種別と対処

| エラー | 原因 | 対処法 |
|--------|------|--------|
| `No prompt files found` | prompt/ が空 | `summarize_v1.0.txt` を配置 |
| `Required columns not found` | 参照ファイルの列名不一致 | 「問合せ内容」「回答」列を確認 |
| `Streamlit port already in use` | 別インスタンスが起動中 | プロセス終了または別ポート使用 |
| `ANTHROPIC_API_KEY not set` | 環境変数未設定 | `.env` ファイルを確認 |
| `Model not found` | モデル名の誤り | `intfloat/multilingual-e5-base` を確認 |
| メモリ不足 | 大量データ処理 | データ分割またはメモリ増設 |
| 初回クエリが遅い | モデル初期化中 | 初回のみ。2回目以降は高速 |

### エラーハンドリングフロー

```mermaid
flowchart TB
    A[クエリ処理開始] --> B{例外発生?}
    B -->|いいえ| C[検索結果を chat_history に追加]
    B -->|はい| D[エラーメッセージ生成]
    D --> E[st.error で表示]
    E --> F[logger.error でログ記録]
    F --> G[エラーを chat_history に追加]
    C --> H[processing_query = False]
    G --> H
    H --> I[st.rerun]
```

### フォールバック処理

```python
def process_query(query: str):
    st.session_state.processing_query = True

    try:
        # 正常処理
        results = processor._process_query(...)

        if results:
            st.session_state.chat_history.append({
                "type": "bot",
                "text": results
            })
        else:
            # 結果が空の場合
            st.session_state.chat_history.append({
                "type": "bot",
                "text": "該当する結果が見つかりませんでした。"
            })

    except Exception as e:
        # エラー発生時
        error_message = f"エラーが発生しました: {str(e)}"
        st.error(error_message)
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        st.session_state.chat_history.append({
            "type": "bot",
            "text": error_message
        })

    finally:
        st.session_state.processing_query = False
```

### ログの確認

```bash
# リアルタイムでログを確認
tail -f app.log

# エラーのみ表示
grep ERROR app.log

# 特定の日付のログ
grep "2025-01-01" app.log
```

### デバッグモード

```python
# utils/utils.py でログレベルを変更
logger.setLevel(logging.DEBUG)
```

---

## セキュリティ

### 重要な注意事項

- `.env` ファイルは絶対に Git にコミットしない
- API キーは定期的にローテーション
- 本番環境では適切なアクセス制御を実装

詳細は [docs/SECURITY.md](../docs/SECURITY.md) を参照してください。

### .gitignore 設定

```text
.env
*.env
input/
output/
reference/
*.log
__pycache__/
.venv/
```

---

## 依存パッケージ

### 主要パッケージ

```text
# Web UI
streamlit>=1.30.0
streamlit_elements==0.1.0

# LangChain エコシステム
langchain>=0.1.0
langchain-anthropic>=0.0.1
langchain-openai>=0.0.1

# 埋め込み・検索
sentence-transformers>=2.2.0
numpy>=1.24.0

# 日本語 NLP
sudachipy>=0.6.8
sudachidict-core>=20230927

# データ処理
pandas>=2.0.0
openpyxl>=3.1.2
xlsxwriter>=3.1.0

# ユーティリティ
python-dotenv>=1.0.0
tqdm
```

---

## ライセンス

MIT License

---

## 関連プロジェクト

| プロジェクト | 説明 | 状態 |
|-------------|------|------|
| [rag-reranker](../rag-reranker/) | Cross-Encoder Reranking 版 | Deprecated |
| [rag-batch](../rag-batch/) | バッチ処理特化版 | Active |
| [rag-gemini](../rag-gemini/) | Gemini 統合版 | Active |
