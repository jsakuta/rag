# RAG-Reranker

**Cross-Encoder Reranking を用いた高精度 RAG システム**

> **Status: DEPRECATED**
> このプロジェクトは非推奨です。新規プロジェクトには [rag-batch](../rag-batch/)、[rag-gemini](../rag-gemini/)、または [rag-streamlit](../rag-streamlit/) をご使用ください。

---

## 目次

- [概要](#概要)
- [アーキテクチャ](#アーキテクチャ)
- [クイックスタート](#クイックスタート)
- [プロジェクト構成](#プロジェクト構成)
- [使用方法](#使用方法)
- [検索アルゴリズム詳細](#検索アルゴリズム詳細)
- [設定オプション](#設定オプション)
- [使用モデル一覧](#使用モデル一覧)
- [入出力フォーマット](#入出力フォーマット)
- [キャッシュシステム](#キャッシュシステム)
- [トラブルシューティング](#トラブルシューティング)
- [依存パッケージ](#依存パッケージ)
- [技術的制限事項](#技術的制限事項)
- [非推奨理由と移行ガイド](#非推奨理由と移行ガイド)

---

## 概要

RAG-Reranker は、融資・ローン業務に特化した質問応答システムです。ハイブリッド検索（ベクトル類似度 + キーワードマッチング）と Cross-Encoder Reranking を組み合わせることで、高精度な回答生成を実現します。

### 主な特徴

| 機能 | 説明 |
|------|------|
| **ハイブリッド検索** | セマンティック検索とキーワード検索を重み付けで組み合わせ |
| **Cross-Encoder Reranking** | 日本語に最適化された Reranker モデルによる精度向上 |
| **マルチプロバイダー LLM** | Anthropic Claude、OpenAI、Google Gemini に対応 |
| **自動キャッシュ** | ベクトル・チャンク・メタデータの自動キャッシュ |
| **バッチ処理** | Excel ファイルの一括処理に対応 |

---

## アーキテクチャ

```mermaid
graph LR
    A[入力<br/>PDF/Excel] --> B[チャンキング<br/>階層構造]
    B --> C[ベクトル化<br/>SentenceTransformer]
    C --> D[ハイブリッド検索<br/>ベクトル + キーワード]
    D --> E[リランキング<br/>Cross-Encoder]
    E --> F[回答生成<br/>LLM]
```

### 検索パイプライン

```mermaid
flowchart TB
    A[ユーザー質問] --> B[ハイブリッド検索]

    subgraph ハイブリッド検索
        C[セマンティック検索<br/>Embedding + Cosine]
        D[キーワード検索<br/>形態素解析 + Overlap]
        C --> E[スコア正規化 & 重み付け統合]
        D --> E
        E -.- E1[combined = 0.7 × semantic + 0.3 × keyword]
    end

    B --> C
    B --> D
    E -->|Top-K 候補| F[Cross-Encoder Reranking<br/>japanese-reranker-cross-encoder-large-v1]
    F -->|Final Top-K| G[LLM 回答生成<br/>Claude / GPT-4 / Gemini]
    G --> H[回答 + 参照情報]
```

---

## クイックスタート

### 1. 環境構築

```bash
# リポジトリのクローン
git clone <repository-url>
cd rag-reranker

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

`.env` ファイルを編集：

```env
# 必須: いずれか1つ以上
OPENAI_API_KEY=sk-your_openai_api_key
ANTHROPIC_API_KEY=sk-ant-your_anthropic_api_key
GOOGLE_API_KEY=your_google_api_key

# オプション: Azure 統合
AZURE_STORAGE_CONNECTION_STRING=your_connection_string
AZURE_TABLE_NAME=your_table_name
AZURE_FORM_RECOGNIZER_ENDPOINT=your_endpoint
AZURE_FORM_RECOGNIZER_KEY=your_key
```

### 3. ドキュメントの配置

```bash
# PDF/Excel ファイルを配置
cp your_documents/*.pdf manuals/
```

### 4. 実行

```bash
python main.py
```

---

## プロジェクト構成

```
rag-reranker/
├── main.py                          # エントリーポイント（対話モード）
├── config.py                        # グローバル設定
├── requirements.txt                 # Python 依存パッケージ
├── .env.example                     # 環境変数テンプレート
├── Test.py                          # PDF 処理・チャンキングテスト
│
├── src/
│   ├── core/
│   │   ├── LoanAssistantBot.py      # メイン Bot（フル機能）
│   │   │   ├── RerankerModel        # Cross-Encoder リランキング
│   │   │   ├── EmbeddingModel       # 埋め込みモデル抽象化
│   │   │   ├── SearchParameters     # 検索パラメータ管理
│   │   │   └── DocumentMetadata     # ドキュメントメタデータ
│   │   │
│   │   ├── LoanAssistantBot_Batch.py    # バッチ処理 v1
│   │   ├── LoanAssistantBot_Batch_v2.py # バッチ処理 v2（改良版）
│   │   ├── bot.py                       # 代替 Bot 実装
│   │   └── azure_rag.py                 # Azure 統合（レガシー）
│   │
│   └── utils/
│       ├── PDF.py                   # PDF 抽出ユーティリティ
│       │   └── pymupdf4llm による Markdown 変換
│       └── text_analyzer.py         # 日本語形態素解析
│           └── Janome/Sudachi トークナイザー
│
├── manuals/                         # 入力ドキュメント（PDF/Excel）
├── input/                           # バッチ処理入力（Excel）
├── prompts/                         # LLM プロンプトテンプレート
│   ├── v1.txt                       # プロンプト v1
│   └── v2.txt                       # プロンプト v2
│
├── cache/                           # キャッシュデータ
│   ├── vectors.pkl                  # ベクトル埋め込み
│   ├── chunks.pkl                   # ドキュメントチャンク
│   ├── metadata.json                # 変更検知用メタデータ
│   └── search_params.json           # 検索パラメータ設定
│
├── docs/                            # ドキュメント
└── tests/                           # テストコード
```

---

## 使用方法

### 対話モード

```bash
python main.py
```

**コマンド一覧:**

| コマンド | 説明 |
|----------|------|
| `<質問文>` | 質問を入力して回答を取得 |
| `params` | 現在の検索パラメータを表示 |
| `set` | 検索パラメータを対話的に設定 |
| `quit` | アプリケーションを終了 |

**出力例:**

```
質問を入力してください: 融資の申し込み条件は？

回答: 融資の申し込みには、以下の条件を満たす必要があります...

=== 参照情報の詳細 ===

[参照チャンク 1]
ファイル: loan_manual.pdf
パス: C:/path/to/loan_manual.pdf
リランキングスコア: 0.9234
初期類似度: 85.2%
意味的類似度: 82.1%
キーワード類似度: 71.3%
内容: ローンの申し込みには...

[参照チャンク 2]
...
```

### バッチ処理

```python
from src.core.LoanAssistantBot_Batch import ExcelQAProcessor, LoanAssistantBot

# Bot の初期化
bot = LoanAssistantBot()
bot.initialize()

# Excel ファイルの一括処理
processor = ExcelQAProcessor(
    bot=bot,
    input_dir="input",
    max_rows=100  # 処理する最大行数
)
processor.process_files()

# 結果: input/ に {入力ファイル名}_processed_{timestamp}.xlsx が生成される
```

### 検索パラメータの調整

```python
# プログラムから設定
bot.update_search_params(
    initial_top_k=15,     # Reranking 前の候補数
    final_top_k=6,        # 最終結果数
    semantic_weight=0.8   # セマンティック重み
)

# または対話モードで 'set' コマンドを使用
```

---

## 検索アルゴリズム詳細

### 1. ハイブリッド検索

#### セマンティック検索（ベクトル類似度）

```python
# クエリをベクトル化
query_embedding = embedding_model.encode(question)

# コサイン類似度を計算
similarities = cosine_similarity(query_embedding, doc_embeddings)

# スコア範囲: 0.0 〜 1.0
```

**使用モデル:**
- デフォルト: `intfloat/multilingual-e5-base` (768次元)
- 代替: `text-embedding-3-large` (OpenAI, 3072次元)

#### キーワード検索

```python
# 日本語形態素解析でキーワード抽出
query_keywords = extract_keywords(question)  # Sudachi/Janome

# キーワード重複率を計算
keyword_score = len(matched_keywords) / len(total_keywords)

# スコア範囲: 0.0 〜 1.0
```

#### スコア統合

```python
# 正規化（0-1範囲に調整）
semantic_norm = normalize(semantic_scores)
keyword_norm = normalize(keyword_scores)

# 重み付け統合
combined = semantic_weight * semantic_norm + (1 - semantic_weight) * keyword_norm

# デフォルト: semantic_weight = 0.7
```

### 2. Cross-Encoder Reranking

```python
# 上位候補を Cross-Encoder で再評価
reranker = CrossEncoder("hotchpotch/japanese-reranker-cross-encoder-large-v1")

# (質問, 文書) ペアのスコアリング
scores = reranker.predict([(question, doc) for doc in top_k_docs])

# 最終スコアでソート
final_results = sorted(zip(scores, docs), reverse=True)[:final_top_k]
```

**Cross-Encoder の特徴:**
- 日本語に最適化されたモデル
- 質問と文書の相互作用を直接モデリング
- 高精度だが計算コストが高い（候補数を制限して使用）

---

## 設定オプション

### 検索パラメータ

| パラメータ | デフォルト | 範囲 | 説明 |
|-----------|-----------|------|------|
| `initial_top_k` | 10 | ≥1 | Reranking 前の候補数 |
| `final_top_k` | 4 | 1〜initial_top_k | 最終結果数 |
| `semantic_weight` | 0.7 | 0.0〜1.0 | セマンティック vs キーワードのバランス |

### モデル設定（config.py）

```python
# LLM プロバイダー
MODEL_PROVIDER = "anthropic"  # "openai", "gemini" も選択可
MODEL_NAME = "claude-3-5-sonnet-20240620"

# 埋め込みモデル
EMBEDDING_PROVIDER = "sentence_transformers"  # "openai" も選択可
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

# Reranker モデル
RERANKER_MODEL = "hotchpotch/japanese-reranker-cross-encoder-large-v1"

# ディレクトリ設定
DOCS_DIRECTORY = "manuals/"
CACHE_DIRECTORY = "cache/"
PROMPT_DIRECTORY = "prompts/"
```

---

## 使用モデル一覧

### 言語モデル（LLM）

| プロバイダー | モデル | 用途 |
|-------------|--------|------|
| Anthropic | claude-3-5-sonnet-20240620 | 回答生成（デフォルト） |
| OpenAI | gpt-4o, gpt-4-turbo | 回答生成（代替） |
| Google | gemini-pro | 回答生成（代替） |

### 埋め込みモデル

| モデル | 次元数 | 特徴 |
|--------|--------|------|
| intfloat/multilingual-e5-base | 768 | 多言語対応（デフォルト） |
| text-embedding-3-large | 3072 | OpenAI 高精度モデル |

### Reranker モデル

| モデル | 特徴 |
|--------|------|
| hotchpotch/japanese-reranker-cross-encoder-large-v1 | 日本語特化、高精度 |

---

## 入出力フォーマット

### 入力フォーマット

#### PDF ファイル
- 標準的なテキスト PDF に対応
- `pymupdf4llm` で Markdown に変換
- 階層的ヘッダー（1．、1-1、[1] など）を認識してチャンキング

#### Excel ファイル（階層構造）
```
| Lv1   | Lv2      | Lv3       | 内容                |
|-------|----------|-----------|---------------------|
| 融資  | 申込条件 | 個人      | 年齢20歳以上...     |
| 融資  | 申込条件 | 法人      | 設立3年以上...      |
```

### 出力フォーマット（バッチ処理）

```
| 質問 | 回答 | 参照ファイル | リランキングスコア | 類似度 |
|------|------|--------------|------------------|--------|
| ...  | ...  | manual.pdf   | 0.9234           | 85.2%  |
```

---

## キャッシュシステム

### キャッシュファイル

| ファイル | 内容 | 形式 |
|----------|------|------|
| `cache/vectors.pkl` | ベクトル埋め込み | Pickle |
| `cache/chunks.pkl` | ドキュメントチャンク | Pickle |
| `cache/metadata.json` | 変更検知用メタデータ | JSON |
| `cache/search_params.json` | ユーザー設定 | JSON |

### キャッシュ無効化条件

以下の場合、自動的に再計算されます：
- ドキュメントファイルの追加・変更・削除
- 埋め込みモデルの変更
- キャッシュファイルの破損

### キャッシュのクリア

```bash
# 全キャッシュをクリア
rm -rf cache/

# または特定のキャッシュのみ
rm cache/vectors.pkl  # ベクトルのみ再計算
```

---

## トラブルシューティング

### よくある問題

| 問題 | 原因 | 解決策 |
|------|------|--------|
| 初回起動が遅い | モデルダウンロード中 | 初回のみ。約5-10分待機 |
| GPU が使用されない | PyTorch の CUDA 版未インストール | `pip install torch --index-url https://download.pytorch.org/whl/cu118` |
| メモリ不足エラー | 大量ドキュメント処理 | `initial_top_k` を減らす、またはドキュメント分割 |
| API エラー | レート制限 or キー無効 | APIキー確認、レート制限待機 |
| 文字化け | エンコーディング問題 | UTF-8 でファイルを保存し直す |

### デバッグ方法

```python
# ログレベルを DEBUG に設定
import logging
logging.basicConfig(level=logging.DEBUG)

# または config.py で設定
LOG_LEVEL = "DEBUG"
```

### キーワード抽出の確認

```python
from src.utils.text_analyzer import extract_keywords

keywords = extract_keywords("融資の申し込み条件は？")
print(keywords)  # ['融資', '申し込み', '条件']
```

---

## 依存パッケージ

### 主要パッケージ

```
# LangChain エコシステム
langchain==0.1.12
langchain-openai==0.0.8
langchain-anthropic==0.1.4
langchain-google-genai==0.0.11

# 埋め込み・検索
sentence-transformers
torch
numpy==1.26.4
scikit-learn==1.4.1.post1

# ドキュメント処理
pymupdf==1.24.0
pandas
openpyxl

# 日本語NLP
sudachipy
sudachidict-core
# または
janome

# ユーティリティ
python-dotenv==1.0.1
```

### GPU サポート（オプション）

```bash
# CUDA 11.8 の場合
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1 の場合
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## 技術的制限事項

1. **日本語特化**: Reranker と形態素解析は日本語に最適化されています。他言語では精度が低下する可能性があります。

2. **メモリ使用量**: 全ベクトルをメモリに保持するため、大規模コーパス（10万文書以上）では注意が必要です。

3. **API レート制限**: 外部 API（LLM、埋め込み）のレート制限を考慮してください。バッチ処理では適切な待機時間を設定することを推奨します。

4. **Cross-Encoder 計算コスト**: Reranking は計算コストが高いため、`initial_top_k` を適切に設定してください（推奨: 10-20）。

---

## 非推奨理由と移行ガイド

### 非推奨理由

1. **古いコード構造**: メンテナンスが困難
2. **機能の改善**: 新プロジェクトで機能が大幅に改善
3. **Azure 依存**: Azure Table Storage への依存が強く、柔軟性に欠ける
4. **スケーラビリティ**: 大規模データに対応していない

### 移行先

| 用途 | 推奨プロジェクト |
|------|-----------------|
| バッチ処理 | [rag-batch](../rag-batch/) |
| 最新技術（Gemini） | [rag-gemini](../rag-gemini/) |
| Web UI | [rag-streamlit](../rag-streamlit/) |

---

## ライセンス

MIT License

---

## 関連プロジェクト

| プロジェクト | 説明 | 状態 |
|-------------|------|------|
| [rag-batch](../rag-batch/) | バッチ処理特化版 | Active |
| [rag-gemini](../rag-gemini/) | Gemini 統合版 | Active |
| [rag-streamlit](../rag-streamlit/) | Web UI 版 | Active |
