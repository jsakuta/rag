# RAG-Batch

**Excel 一括処理に特化したハイブリッド検索システム**

[![Python](https://img.shields.io/badge/Python-3.7+-blue)](https://www.python.org/)
[![Sentence Transformers](https://img.shields.io/badge/SentenceTransformers-2.2.0+-blue)](https://www.sbert.net/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1.0+-green)](https://python.langchain.com/)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## 目次

- [他プロジェクトとの違い](#他プロジェクトとの違い)
- [概要](#概要)
- [アーキテクチャ](#アーキテクチャ)
- [クイックスタート](#クイックスタート)
- [プロジェクト構成](#プロジェクト構成)
- [使用方法](#使用方法)
- [実行結果の具体例](#実行結果の具体例)
- [検索アルゴリズム](#検索アルゴリズム)
- [設定オプション](#設定オプション)
- [入出力フォーマット](#入出力フォーマット)
- [キャッシュシステム](#キャッシュシステム)
- [パフォーマンス](#パフォーマンス)
- [Docker デプロイ](#docker-デプロイ)
- [トラブルシューティング](#トラブルシューティング)
- [拡張性（Factory Pattern）](#拡張性factory-pattern)
- [依存パッケージ](#依存パッケージ)
- [セキュリティ](#セキュリティ)

---

## 他プロジェクトとの違い

| 特徴 | rag-batch | rag-streamlit | rag-gemini |
|------|-----------|---------------|------------|
| **主な用途** | Excel バッチ処理 | 対話的検索・デモ | 最新技術 |
| **UI** | CLI + Streamlit | Streamlit | Streamlit |
| **ベクトルDB** | JSON キャッシュ | JSON キャッシュ | ChromaDB |
| **埋め込みモデル** | multilingual-e5-base | multilingual-e5-base | Gemini Embedding |
| **設計パターン** | Factory Pattern | 単一実装 | Factory Pattern |

**関連プロジェクト:**

- 対話的 UI が必要な場合: [rag-streamlit](../rag-streamlit/)
- 最新技術を使いたい場合: [rag-gemini](../rag-gemini/)

---

## 概要

RAG-Batch は、大規模言語モデル（LLM）とベクトル検索、キーワード検索を組み合わせたハイブリッド検索システムです。銀行預金業務における問い合わせデータの一括処理に最適化されています。

### 主な機能

| 機能 | 説明 |
|------|------|
| **ハイブリッド検索** | ベクトル類似度 + キーワード類似度の重み付け統合 |
| **バッチ処理** | Excel ファイルの一括処理と結果出力 |
| **インタラクティブモード** | Streamlit ベースの Web UI |
| **ベクトルキャッシング** | JSON 形式でのキャッシュ保存 |
| **詳細ロギング** | 処理状況の完全トレース |
| **Factory Pattern** | 入出力形式の拡張性 |

---

## アーキテクチャ

```mermaid
graph TB
    subgraph ファクトリ層
        A1[入力ハンドラ<br/>Excel / CSV / JSON]
        A2[出力ハンドラ<br/>Excel / CSV / JSON]
    end

    subgraph 処理層
        B[Processor]
        B1[load_data - 入力データ読み込み]
        B2[load_reference - 参照データ読み込み]
        B3[prepare_search - ベクトル準備]
        B4[process_data - バッチ処理ループ]
    end

    subgraph 検索層
        C[Searcher]
        C1[ベクトル検索<br/>SentenceTransformers]
        C2[キーワード検索<br/>SudachiPy 形態素解析]
        C3[スコア統合<br/>combined = vw × vec + kw × keyword]
    end

    A1 --> B
    B --> B1 --> B2 --> B3 --> B4
    B4 --> C
    C --> C1
    C --> C2
    C1 --> C3
    C2 --> C3
    C3 --> A2
```

### 処理シーケンス

```mermaid
sequenceDiagram
    participant Main as main.py
    participant Proc as Processor
    participant Input as InputHandler
    participant Search as Searcher
    participant Output as OutputHandler

    Main->>Proc: process_data(mode="batch")
    Proc->>Input: load_data()
    Input-->>Proc: input_data

    Proc->>Input: load_reference_data()
    Input-->>Proc: reference_data

    Proc->>Search: prepare_search(reference_data)
    Search->>Search: キャッシュ確認
    alt キャッシュあり
        Search->>Search: キャッシュ読み込み
    else キャッシュなし
        Search->>Search: ベクトル生成 + 保存
    end

    loop 各入力データ
        Proc->>Search: search(query)
        Search->>Search: LLM 要約
        Search->>Search: ベクトル検索
        Search->>Search: キーワード検索
        Search->>Search: スコア統合
        Search-->>Proc: results[]
    end

    Proc->>Output: save_data(all_results)
    Output-->>Proc: output_file
```

---

## クイックスタート

### 1. 環境構築

```bash
# リポジトリのクローン
git clone <repository-url>
cd rag-batch

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
```

### 3. データの配置

```bash
# 参照データを配置
cp your_reference_data.xlsx reference/

# 入力データを配置
cp your_input_data.xlsx input/
```

### 4. 実行

**バッチモード:**

```bash
python main.py
```

**インタラクティブモード:**

```bash
python main.py interactive
# または
streamlit run ui/chat.py
```

---

## プロジェクト構成

```text
rag-batch/
├── main.py                       # エントリーポイント
├── config.py                     # 設定管理（SearchConfig dataclass）
├── requirements.txt              # Python 依存パッケージ
├── .env.example                  # 環境変数テンプレート
├── Dockerfile                    # Docker コンテナ設定
├── .dockerignore                 # Docker 除外ファイル
│
├── src/
│   ├── core/
│   │   ├── processor.py          # データ処理エンジン
│   │   │   └── Processor
│   │   │       └── process_data()
│   │   │
│   │   └── searcher.py           # ハイブリッド検索エンジン
│   │       └── Searcher
│   │           ├── _setup_llm()
│   │           ├── _extract_keywords()
│   │           ├── _calculate_keyword_similarity()
│   │           ├── prepare_search()
│   │           ├── summarize_text()
│   │           └── search()
│   │
│   ├── handlers/
│   │   ├── input_handler.py      # 入力処理（Factory Pattern）
│   │   │   └── ExcelInputHandler
│   │   │       ├── load_data()
│   │   │       └── load_reference_data()
│   │   │
│   │   └── output_handler.py     # 出力処理（Factory Pattern）
│   │       └── ExcelOutputHandler
│   │           └── save_data()
│   │
│   └── utils/
│       ├── logger.py             # ログ設定
│       └── utils.py              # ユーティリティ関数
│
├── ui/
│   └── chat.py                   # Streamlit チャット UI
│
├── prompt/
│   └── summarize_v1.0.txt        # LLM プロンプトテンプレート
│
├── input/                        # 入力ファイル
├── reference/                    # 参照データ
│   └── vector_cache/             # ベクトルキャッシュ（JSON）
│       └── cache.json
├── output/                       # 出力ファイル
├── logs/                         # アプリケーションログ
│   └── app.log
│
└── tests/                        # テストコード
```

---

## 使用方法

### バッチモード

```bash
python main.py
```

**処理フロー:**

1. `input/` フォルダの最新 Excel ファイルを読み込み
2. `reference/` フォルダの参照データをベクトル化（キャッシュ対応）
3. 各質問に対してハイブリッド検索を実行
4. 結果を `output/` フォルダに Excel 形式で保存

**出力ファイル名:**

```text
output_batch_v{vector_weight}_k{keyword_weight}_{タイムスタンプ}.xlsx
例: output_batch_v0.9_k0.1_20250101_120000.xlsx
```

### インタラクティブモード

```bash
python main.py interactive
```

**UI 構成:**

| コンポーネント | 説明 |
|---------------|------|
| **サイドバー** | vector_weight スライダー（0.0-1.0）、top_k 入力（1-10） |
| **チャットエリア** | 質問入力と検索結果表示 |
| **保存ボタン** | チャット履歴を Excel にエクスポート |

**検索結果カード:**

```text
┌─────────────────────────────────────────────┐
│ 候補 1 (類似度: 0.9234)                     │
├─────────────────────────────────────────────┤
│ 類似質問:                                   │
│ ATMでの現金処理時に操作を誤った場合は？     │
├─────────────────────────────────────────────┤
│ 回答:                                       │
│ 現金処理で操作を誤った場合は、以下の手順... │
└─────────────────────────────────────────────┘
```

---

## 実行結果の具体例

### 入力データ

**input/questions_20250101.xlsx:**

| No | ユーザーの質問 | オリジナル回答 |
|----|---------------|----------------|
| 1 | ATM現金処理の操作方法は？ | ATMで現金を... |
| 2 | 口座開設に必要な書類は？ | 本人確認書類... |

### ログ出力

```text
2025-01-01 12:00:00 - INFO - processor - Starting batch processing...
2025-01-01 12:00:01 - INFO - input_handler - Loaded 2 input rows from input/questions_20250101.xlsx
2025-01-01 12:00:02 - INFO - input_handler - Loaded 500 reference rows from reference/qa_database.xlsx
2025-01-01 12:00:03 - INFO - searcher - No cache file found. Creating new cache.
2025-01-01 12:00:35 - INFO - searcher - Cached reference vectors to reference/vector_cache/cache.json

Processing data: 100%|██████████| 2/2 [00:15<00:00,  7.50s/it]

2025-01-01 12:00:50 - INFO - searcher - Row (No.1):
2025-01-01 12:00:50 - INFO - searcher -   Original query: ATM現金処理の操作方法は？
2025-01-01 12:00:50 - INFO - searcher -   Summarized query: ATM現金処理操作
2025-01-01 12:00:50 - INFO - searcher -   Extracted keywords: ['ATM', '現金', '処理', '操作']
2025-01-01 12:00:51 - INFO - searcher -   Added result with similarity: 0.9234
2025-01-01 12:00:51 - INFO - searcher -   Added result with similarity: 0.8756
2025-01-01 12:00:51 - INFO - searcher -   Added result with similarity: 0.8421
2025-01-01 12:00:51 - INFO - searcher -   Added result with similarity: 0.8102

2025-01-01 12:00:58 - INFO - searcher - Row (No.2):
2025-01-01 12:00:58 - INFO - searcher -   Original query: 口座開設に必要な書類は？
2025-01-01 12:00:58 - INFO - searcher -   Summarized query: 口座開設必要書類
2025-01-01 12:00:58 - INFO - searcher -   Extracted keywords: ['口座', '開設', '書類', '必要']
2025-01-01 12:00:59 - INFO - searcher -   Added result with similarity: 0.9512
2025-01-01 12:00:59 - INFO - searcher -   Added result with similarity: 0.8934
2025-01-01 12:00:59 - INFO - searcher -   Added result with similarity: 0.8567
2025-01-01 12:00:59 - INFO - searcher -   Added result with similarity: 0.8234

2025-01-01 12:01:00 - INFO - output_handler - Generated 8 result rows
2025-01-01 12:01:01 - INFO - output_handler - Saved to output/output_batch_v0.9_k0.1_20250101_120000.xlsx
```

### 出力ファイル

**output/output_batch_v0.9_k0.1_20250101_120000.xlsx:**

| # | ユーザーの質問 | 検索クエリ(AI処理) | 類似質問 | 類似回答 | 類似度 |
|---|---------------|-------------------|---------|---------|--------|
| 1 | ATM現金処理の操作方法は？ | ATM現金処理操作 | ATMでの現金処理について | 現金処理の手順は... | 0.9234 |
|   |               |                   | ATM入金手続き | 入金する場合は... | 0.8756 |
|   |               |                   | ATM出金方法 | 出金の際は... | 0.8421 |
|   |               |                   | 現金取扱注意点 | 現金を扱う際の... | 0.8102 |
| 2 | 口座開設に必要な書類は？ | 口座開設必要書類 | 新規口座開設の必要書類 | 本人確認書類と... | 0.9512 |
|   |               |                   | 口座作成手続き | 口座を作る際は... | 0.8934 |

---

## 検索アルゴリズム

### ハイブリッド検索パイプライン

```mermaid
flowchart TB
    A[入力質問] --> B[LLM 要約処理<br/>summarize_v1.0.txt プロンプト使用]
    B --> C[検索クエリ生成]
    C --> D[ベクトル検索]
    C --> E[キーワード検索]

    subgraph ベクトル検索
        D --> D1[SentenceTransformers<br/>multilingual-e5]
        D1 --> D2[コサイン類似度]
    end

    subgraph キーワード検索
        E --> E1[SudachiPy 形態素解析<br/>名詞抽出 + 重み付け]
        E1 --> E2[重み付き Jaccard<br/>+ 位置重み ×1.2]
    end

    D2 --> F[スコア統合]
    E2 --> F
    F --> G[Top-K 結果]

    F -.- F1[combined = vw × vec + kw × keyword<br/>デフォルト: vw=0.9, kw=0.1]
```

### ベクトル検索詳細

```python
# SentenceTransformer による埋め込み
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/multilingual-e5-base")
query_embedding = model.encode(query, normalize_embeddings=True)

# コサイン類似度計算
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
```

### キーワード検索詳細

```python
# SudachiPy による形態素解析
import sudachipy

tokenizer = sudachipy.Dictionary().create()
tokens = tokenizer.tokenize(text, sudachipy.SplitMode.C)

# 名詞抽出（重み付け）
keywords = []
for token in tokens:
    pos = token.part_of_speech()
    if pos[0] == "名詞":
        # 固有名詞は重み2、一般名詞は重み1
        weight = 2 if "固有名詞" in pos else 1
        keywords.append((token.surface(), weight))

# ストップワード除去
STOP_WORDS = {"こと", "もの", "する", "ある", "いる", ...}
keywords = [kw for kw in keywords if kw[0] not in STOP_WORDS]
```

### 重み付き Jaccard 類似度

```python
def calculate_keyword_similarity(query_keywords, doc_keywords):
    # 位置重み: 前半のキーワードは ×1.2
    POSITION_WEIGHT = 1.2

    query_set = set()
    for i, (kw, weight) in enumerate(query_keywords):
        pos_weight = POSITION_WEIGHT if i < len(query_keywords) / 2 else 1.0
        query_set.add((kw, weight * pos_weight))

    # Jaccard 類似度計算
    intersection = query_set & doc_set
    union = query_set | doc_set
    return len(intersection) / len(union)
```

---

## 設定オプション

### SearchConfig (config.py)

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `top_k` | int | 4 | 返却する類似文書数 |
| `model_name` | str | intfloat/multilingual-e5-base | 埋め込みモデル |
| `vector_weight` | float | 0.9 (batch) / 0.7 (UI) | ベクトル検索の重み |
| `keyword_weight` | float | 自動計算 | 1.0 - vector_weight |
| `llm_provider` | str | anthropic | LLM プロバイダー |
| `llm_model` | str | claude-3-5-sonnet-20241022 | LLM モデル |
| `base_dir` | str | "." | 基準ディレクトリ |
| `input_type` | str | excel | 入力形式 |
| `output_type` | str | excel | 出力形式 |

### 設定読み込み優先順位

1. コード内デフォルト値（最低優先度）
2. `config.yaml`（存在する場合）
3. 環境変数（API キーのみ）

---

## 入出力フォーマット

### 入力ファイル

**場所:** `input/` ディレクトリ（最新ファイルを自動選択）

**形式:** Excel (.xlsx)

| 列 | 必須 | 説明 |
|----|------|------|
| 1列目 | はい | 番号/ID |
| 2列目 | はい | 質問内容 |
| 3列目 | いいえ | オリジナル回答 |

**例:**

```text
| No | ユーザーの質問                       | オリジナル回答 |
|----|--------------------------------------|----------------|
| 1  | ATM現金処理の操作方法は？            | ATMで現金...   |
| 2  | 口座開設に必要な書類は？             | 本人確認...    |
```

### 参照データ

**場所:** `reference/` ディレクトリ（最新ファイルを自動選択）

**形式:** Excel (.xlsx)

| 列名 | 必須 | 説明 |
|------|------|------|
| 問合せ内容 | はい | 参照質問文 |
| 回答 | はい | 参照回答文 |

**重要:** 列名は完全一致が必要です

### 出力ファイル

**場所:** `output/` ディレクトリ

**形式:** Excel (.xlsx)

| 列名（日本語） | 説明 |
|----------------|------|
| # | 入力番号 |
| ユーザーの質問 | 元の質問文 |
| オリジナルの回答 | 入力ファイルの回答 |
| 検索クエリ（AI処理） | LLM 生成検索クエリ |
| 類似質問 | 検索結果の質問 |
| 類似回答 | 検索結果の回答 |
| 類似度 | 統合類似度スコア（0.0-1.0） |
| ベクトルの重み | 使用した vector_weight |
| 候補数 | top_k 値 |

**Excel フォーマット:**

- フォント: Meiryo 10pt
- ヘッダー: 太字、グレー背景 (#D9D9D9)
- 罫線: 全セル
- テキスト: 折り返し有効

---

## キャッシュシステム

### ベクトルキャッシュ

**場所:** `reference/vector_cache/cache.json`

### キャッシュファイル構造

```json
{
  "vectors": [
    [0.0234, -0.0156, 0.0312, ...],  // 768次元のベクトル
    [0.0189, -0.0234, 0.0156, ...],
    // ... 参照データ件数分
  ],
  "texts": [
    "ATMでの現金処理について教えてください",
    "口座開設に必要な書類は何ですか",
    // ... 参照データ件数分
  ],
  "timestamp": "2025-01-01T12:00:00.000000"
}
```

### キャッシュサイズ目安

| 参照データ件数 | ベクトル次元 | 概算ファイルサイズ |
|---------------|-------------|-------------------|
| 100件 | 768 | 約 2 MB |
| 500件 | 768 | 約 10 MB |
| 1,000件 | 768 | 約 20 MB |
| 5,000件 | 768 | 約 100 MB |
| 10,000件 | 768 | 約 200 MB |

### キャッシュ動作フロー

```mermaid
flowchart TB
    A[prepare_search 開始] --> B{cache.json 存在?}
    B -->|いいえ| C[ベクトル生成]
    B -->|はい| D[キャッシュ読み込み]
    D --> E{テキスト一致?}
    E -->|はい| F[キャッシュ使用]
    E -->|いいえ| C
    C --> G[ベクトルキャッシュ保存]
    G --> H[検索準備完了]
    F --> H
```

### キャッシュ動作一覧

| 状態 | 動作 | 所要時間（500件） |
|------|------|------------------|
| キャッシュなし | 参照データをベクトル化、キャッシュ保存 | 30-60秒 |
| キャッシュあり + データ変更なし | キャッシュ使用（高速） | 1秒未満 |
| キャッシュあり + データ変更あり | キャッシュ再生成 | 30-60秒 |

### キャッシュクリア

```bash
rm reference/vector_cache/cache.json
```

---

## パフォーマンス

### 処理時間内訳

```mermaid
pie title 処理時間内訳（100件処理時）
    "LLM 要約" : 45
    "ベクトル検索" : 25
    "キーワード検索" : 15
    "ファイル I/O" : 10
    "その他" : 5
```

### 処理時間の目安

| 処理フェーズ | 100件 | 500件 | 1,000件 |
|-------------|-------|-------|---------|
| 初回キャッシュ生成 | 15秒 | 60秒 | 120秒 |
| キャッシュ読み込み | 0.5秒 | 1秒 | 2秒 |
| 1件あたりの検索 | 3-5秒 | 3-5秒 | 3-5秒 |
| **合計（初回）** | 5-8分 | 25-40分 | 50-80分 |
| **合計（2回目以降）** | 5-8分 | 25-40分 | 50-80分 |

### ベンチマーク結果

**テスト環境:**
- CPU: Intel Core i7-12700 (12コア)
- RAM: 32GB
- GPU: なし（CPU のみ）
- 参照データ: 500件
- 入力データ: 50件

| 項目 | 値 |
|------|-----|
| 初回実行（キャッシュ生成込み） | 4分32秒 |
| 2回目以降 | 3分45秒 |
| 1件あたり平均処理時間 | 4.5秒 |
| LLM API 呼び出し回数 | 50回 |
| メモリ使用量（ピーク） | 2.3GB |

### パフォーマンス最適化のヒント

| 最適化項目 | 効果 | 方法 |
|-----------|------|------|
| キャッシュ活用 | 初回起動時間を短縮 | 参照データ変更時のみ再生成 |
| バッチサイズ調整 | メモリ使用量削減 | 大量データは分割処理 |
| LLM モデル変更 | API 呼び出し高速化 | haiku など軽量モデル使用 |
| top_k 削減 | 出力処理高速化 | 必要最小限の候補数に設定 |

---

## Docker デプロイ

### Docker ビルド

```bash
docker build -t rag-batch:latest .
```

### Docker 実行

**バッチモード:**

```bash
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/reference:/app/reference \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/prompt:/app/prompt \
  -e ANTHROPIC_API_KEY=your_key \
  rag-batch:latest main.py
```

**インタラクティブモード:**

```bash
docker run -p 8501:8501 \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/reference:/app/reference \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/prompt:/app/prompt \
  -e ANTHROPIC_API_KEY=your_key \
  rag-batch:latest main.py interactive
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  rag-batch:
    build:
      context: .
      dockerfile: Dockerfile
    volumes:
      - ./input:/app/input
      - ./reference:/app/reference
      - ./output:/app/output
      - ./prompt:/app/prompt
      - ./logs:/app/logs
    env_file:
      - .env
    profiles:
      - batch

  rag-batch-ui:
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
    command: ["python", "main.py", "interactive"]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

**使用方法:**

```bash
# バッチ処理実行
docker-compose --profile batch run rag-batch

# UI モード起動
docker-compose up rag-batch-ui -d
```

### Dockerfile の概要

```dockerfile
FROM python:3.11-slim

# ビルドツール
RUN apt-get update && apt-get install -y build-essential gcc curl

# 依存パッケージ
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーション
WORKDIR /app
COPY . .

# 環境設定
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
EXPOSE 8501

# ヘルスチェック
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# ボリューム
VOLUME ["/app/input", "/app/reference", "/app/output", "/app/prompt"]

ENTRYPOINT ["python"]
CMD ["main.py"]
```

---

## トラブルシューティング

共通の問題については [docs/TROUBLESHOOTING.md](../docs/TROUBLESHOOTING.md) を参照してください。

### エラー種別と対処

| エラー | 原因 | 解決策 |
|--------|------|--------|
| `FileNotFoundError: cache.json` | vector_cache/ ディレクトリがない | `mkdir -p reference/vector_cache` |
| `MemoryError` | 大量データ処理 | データ分割またはメモリ増設 |
| `KeyError: '問合せ内容'` | 参照ファイルの列名不一致 | 「問合せ内容」「回答」列を確認 |
| `ANTHROPIC_API_KEY not set` | API キー未設定 | `.env` ファイルを確認 |
| `Connection timeout` | LLM API 接続エラー | ネットワーク確認、リトライ |
| モデルダウンロード遅延 | 初回実行時 | 初回のみ。約5-10分待機 |

### エラーハンドリングフロー

```mermaid
flowchart TB
    A[process_data 開始] --> B{入力読み込み}
    B -->|成功| C{参照データ読み込み}
    B -->|失敗| E1[FileNotFoundError<br/>→ 入力ファイル確認]
    C -->|成功| D{キャッシュ準備}
    C -->|失敗| E2[KeyError<br/>→ 列名確認]
    D -->|成功| F{検索ループ}
    D -->|失敗| E3[IOError<br/>→ ディスク容量確認]
    F -->|成功| G[結果保存]
    F -->|失敗| E4[API Error<br/>→ キー・ネットワーク確認]
    G -->|成功| H[完了]
    G -->|失敗| E5[PermissionError<br/>→ 出力ディレクトリ確認]

    E1 --> I[ログ記録]
    E2 --> I
    E3 --> I
    E4 --> I
    E5 --> I
    I --> J[例外再送出]
```

### ログの確認

```bash
# リアルタイムログ
tail -f logs/app.log

# エラーのみ
grep ERROR logs/app.log

# 特定の日付
grep "2025-01-01" logs/app.log

# デバッグレベル有効化
# utils/logger.py で logging.DEBUG に変更
```

---

## 拡張性（Factory Pattern）

### 新しい入力形式の追加

```python
# src/handlers/input_handler.py

class CsvInputHandler:
    def load_data(self):
        # CSV 読み込みロジック
        pass

    def load_reference_data(self):
        # CSV 参照データ読み込み
        pass

# ファクトリーに登録
INPUT_HANDLERS = {
    "excel": ExcelInputHandler,
    "csv": CsvInputHandler,  # 追加
}
```

### 新しい出力形式の追加

```python
# src/handlers/output_handler.py

class JsonOutputHandler:
    def save_data(self, results, filepath):
        # JSON 出力ロジック
        pass

# ファクトリーに登録
OUTPUT_HANDLERS = {
    "excel": ExcelOutputHandler,
    "json": JsonOutputHandler,  # 追加
}
```

### Factory Pattern の利点

| 利点 | 説明 |
|------|------|
| 疎結合 | 入出力ロジックとコアロジックの分離 |
| 拡張性 | 新形式追加時にコア変更不要 |
| テスト容易性 | ハンドラ単体でのテスト可能 |
| 設定駆動 | config で形式を切り替え可能 |

---

## 依存パッケージ

### 主要パッケージ

```text
# LangChain エコシステム
langchain>=0.1.0
langchain-anthropic>=0.0.1
langchain-openai>=0.0.1

# 埋め込み・検索
sentence-transformers>=2.2.0
numpy>=1.24.0
scikit-learn>=1.3.0

# 日本語 NLP
sudachipy>=0.6.8
sudachidict-core>=20230927

# データ処理
pandas>=2.0.0
openpyxl>=3.1.2
xlsxwriter>=3.1.0

# Web UI
streamlit>=1.30.0

# ユーティリティ
python-dotenv>=1.0.0
tqdm
```

---

## セキュリティ

### 重要な注意事項

- `.env` ファイルは絶対に Git にコミットしない
- API キーは定期的にローテーション
- 入力データの検証を実装（本番環境）
- 機密データを含む Excel ファイルの適切な管理

詳細は [docs/SECURITY.md](../docs/SECURITY.md) を参照してください。

### .gitignore 設定

```text
.env
*.env
input/
output/
reference/
logs/
__pycache__/
.venv/
```

---

## ライセンス

MIT License

---

## バグ報告・機能要望

GitHub Issues にて受け付けています。

**バグ報告時に含める情報:**

- エラーメッセージ
- `logs/app.log` の内容
- 再現手順
- 利用環境（OS、Python バージョン）

---

## 関連プロジェクト

| プロジェクト | 説明 | 状態 |
|-------------|------|------|
| [rag-reranker](../rag-reranker/) | Cross-Encoder Reranking 版 | Deprecated |
| [rag-streamlit](../rag-streamlit/) | 対話的 UI 版 | Active |
| [rag-gemini](../rag-gemini/) | Gemini 統合版 | Active |
