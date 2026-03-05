# システムアーキテクチャ

このドキュメントでは、RAG-Localシステムのアーキテクチャとモジュール間の依存関係を説明します。

## 目次

- [全体構成](#全体構成)
- [レイヤー構造](#レイヤー構造)
- [主要クラス一覧](#主要クラス一覧)
- [データフロー](#データフロー)
- [依存関係](#依存関係)
- [拡張性](#拡張性)
- [パフォーマンス最適化](#パフォーマンス最適化)
- [セキュリティ考慮事項](#セキュリティ考慮事項)
- [テスト](#テスト)

---

## 全体構成

```
┌────────────────────────────────────────────────────────────┐
│                      Entry Points                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   main.py    │  │  ui/chat.py  │  │   scripts/   │    │
│  │  (Batch)     │  │ (Streamlit)  │  │ (Evaluation) │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
└────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────┐
│                   Configuration Layer                      │
│  config/ - 設定ファイル                                    │
│    ├─ settings.yaml（common/ui/batch/evaluation）          │
│    └─ business_areas.yaml（業務分野マッピング）              │
│  config.py - SearchConfig データクラス                      │
└────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────┐
│                      Core Layer                            │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ processor.py - データ処理エンジン                    │ │
│  │ judgment_support.py - LLM判断支援                    │ │
│  └──────────────────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ search/ - 多段階検索サブシステム                     │ │
│  │   ├─ search_strategy.py - 検索戦略切替（4戦略）    │ │
│  │   ├─ multi_stage_orchestrator.py - オーケストレータ │ │
│  │   ├─ query_enhancer.py - クエリ拡張                 │ │
│  │   ├─ vector_search_engine.py - ベクトル検索         │ │
│  │   ├─ keyword_search_engine.py - キーワード検索      │ │
│  │   ├─ chromadb_keyword_search.py - ChromaDBキーワード│ │
│  │   └─ text_combiner.py - テキスト結合                │ │
│  └──────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────┐
│                     Types Layer                            │
│  types/ - 型定義・定数                                     │
│    └─ search_types.py（TypedDict / Dataclass / 定数）     │
└────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────┐
│                    Handler Layer                           │
│  ┌──────────────────┐  ┌──────────────────┐               │
│  │ input_handler.py │  │ output_handler.py│               │
│  │ (Excel読み込み)  │  │ (Excel出力)      │               │
│  └──────────────────┘  └──────────────────┘               │
└────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────┐
│                     Utils Layer                            │
│  ┌────────────────────────────────────────────────────┐   │
│  │ データベース管理                                    │   │
│  │   ├─ dynamic_db_manager.py - 動的DB管理           │   │
│  │   ├─ vector_db.py - ChromaDB 操作の共通インターフェース             │   │
│  │   └─ business_area_translator.py - 業務領域変換   │   │
│  └────────────────────────────────────────────────────┘   │
│  ┌────────────────────────────────────────────────────┐   │
│  │ 埋め込みモデル                                      │   │
│  │   ├─ base_embedding.py - 抽象基底クラス           │   │
│  │   ├─ gemini_embedding.py - VertexAI埋め込み       │   │
│  │   └─ azure_embedding.py - Azure OpenAI埋め込み    │   │
│  └────────────────────────────────────────────────────┘   │
│  ┌────────────────────────────────────────────────────┐   │
│  │ その他                                              │   │
│  │   ├─ auth.py - Google Cloud認証                   │   │
│  │   └─ logger.py - ログ設定                         │   │
│  └────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────┐
│                  External Services                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│  │ Vertex   │  │  Azure   │  │  Azure   │               │
│  │   AI     │  │  OpenAI  │  │Key Vault │               │
│  └──────────┘  └──────────┘  └──────────┘               │
└────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────┐
│                    Data Storage                            │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐          │
│  │ ChromaDB   │  │  Excel     │  │   Logs     │          │
│  │(data/      │  │(data/      │  │  (logs/)   │          │
│  │ vector_db/)│  │ input/out) │  │            │          │
│  └────────────┘  └────────────┘  └────────────┘          │
└────────────────────────────────────────────────────────────┘
```

---

## レイヤー構造

### 1. Entry Points Layer

| モジュール | 説明 | 用途 |
|-----------|------|------|
| `main.py` | バッチ処理エントリーポイント | 一括検索実行 |
| `ui/chat.py` | Streamlit UI | インタラクティブ検索 |
| `ui/ops_ui.py` | 運用保守効率化AI Streamlit UI | 評価モード（正解ID付き精度検証）/ 影響調査モード（正解IDなし影響範囲調査）の2モード構成。詳細は [REVISION_OPS.md](./REVISION_OPS.md) |
| `scripts/*` | ユーティリティスクリプト | DB再構築、評価実行 |

### 2. Core Layer

**processor.py** - 入力データ読み込みと検証、Searcher + SearchStrategy による検索実行の管理、結果の集約と整形（通常モード / 多段階検索モード）を統合管理する。

**searcher.py** - 動的ベクトルDB選択（業務分野・入力ファイルに基づく）、埋め込みモデル・LLMの初期化管理、キーワード抽出・キャッシング、SearchStrategy への検索委譲を行う。

**search/multi_stage_orchestrator.py** - 多段階ハイブリッド検索のオーケストレーター。検索ステージの管理、ベクトル・キーワード検索の統合、スコア計算とランキングを担う。

**judgment_support.py** - LLMによる検索結果の関連性判定と判定根拠の生成。

#### SearchStrategy - 戦略パターン

統一インターフェース: `execute(input_number, query_text, original_answer) -> List[Dict]`

| 戦略クラス | search_mode | 処理 | 用途 |
|-----------|-------------|------|------|
| `OriginalSearchStrategy` | original（**デフォルト**） | 原文でベクトル+キーワード検索 | 回答支援AI。固有名詞の欠落なく最も安定 |
| `LLMEnhancedSearchStrategy` | llm_enhanced | LLMクエリ生成後にベクトル+キーワード検索 | 表現の揺れが大きい検索語の場合 |
| `MultiStageSearchStrategy` | multi_stage | 原文+LLMクエリの両検索→OR結合・3分類（運用保守効率化AI専用） | 改定影響調査。漏れを減らすために両方の結果を統合 |
| `KeywordFilterSearchStrategy` | keyword_filter | キーワードマッチのみ（ベクトル検索なし） | 用語の単純置換（AML→GPLEX等）の検出 |

デフォルト値は `config/settings.yaml` の `search_mode: original`。UI ではサイドバーで動的に切替可能。

### 3. Handler Layer

**input_handler.py** - 入力処理。Factoryパターンで入力形式に応じたハンドラーを生成する。

| サブクラス | 用途 | 入力形式 |
|-----------|------|---------|
| `ExcelInputHandler` | 問い合わせ履歴データ（FAQ）読み込み（回答支援AI） | 標準Excel（番号, 質問, 回答） |
| `HierarchicalExcelInputHandler` | マージ版シナリオ読み込み | Excel複数シート（階層+質問+回答） |
| `MultiFolderInputHandler` | シナリオ+FAQ統合（回答支援AI） | 複数フォルダ |
| `TextInputHandler` | 改定内容入力（改定影響調査） | Excel + 正解ID対応 |

列名の動的解決: settings.yaml の `columns` セクションに候補列を列挙し、Excelに存在する最初の列を採用。query/answer は必須（ValueError）、tag は任意（警告続行）。

**output_handler.py** - 出力処理。Factoryパターンで出力形式に応じたハンドラーを生成する。`app_prefix` で出力サブディレクトリを指定（例: "answer", "rev"）。

### 4. Utils Layer

**データベース管理** - `dynamic_db_manager.py`（業務領域別のベクトルDB管理、タイムスタンプ検証、差分ベクトル化）、`vector_db.py`（ChromaDB 操作の共通インターフェース、LRUCache(max_size=10) でクライアントキャッシュ）、`business_area_translator.py`（日本語業務名→英語コレクション名変換）。

**埋め込みモデル** - `base_embedding.py`（抽象基底クラス）、`gemini_embedding.py`（VertexAI Gemini）、`azure_embedding.py`（Azure OpenAI）。各プロバイダーの実装はバッチ処理（`EMBEDDING_BATCH_SIZE = 250`）、リトライロジック対応。対応プロバイダー: `config.py` の `VALID_EMBEDDING_PROVIDERS = ("vertex_ai", "azure_openai")`。

**その他** - `auth.py`（Google Cloud 認証）、`logger.py`（ログ設定）。

---

## 主要クラス一覧

各クラスの詳細な API（メソッドシグネチャ・引数・戻り値）はソースコードの docstring を参照してください。

### コアモジュール

| クラス | モジュール | 責務 |
|--------|----------|------|
| `Processor` | `src/core/processor.py` | 入力読込→検索→出力の統合管理。InputHandlerFactory / OutputHandlerFactory で入出力を切替 |
| `Searcher` | `src/core/searcher.py` | 動的DB選択・キーワードキャッシュ・SearchStrategy への検索委譲 |
| `JudgmentSupport` | `src/core/judgment_support.py` | LLMによる検索結果の関連性判定（関連あり/要確認/無関係の3段階） |

### 検索エンジン

| クラス | モジュール | 責務 |
|--------|----------|------|
| `SearchStrategy` | `src/core/search/search_strategy.py` | 4戦略パターン（Original / LLMEnhanced / MultiStage / KeywordFilter） |
| `MultiStageOrchestrator` | `src/core/search/multi_stage_orchestrator.py` | 多段階ハイブリッド検索（Stage 1: 原文 → Stage 2: LLM拡張 → Stage 3: OR結合・3分類） |
| `QueryEnhancer` | `src/core/search/query_enhancer.py` | LLMによるクエリ拡張（プロンプト: `prompt/summarize_v1.0.txt`） |
| `VectorSearchEngine` | `src/core/search/vector_search_engine.py` | ベクトル検索（コサイン類似度） |
| `KeywordSearchEngine` | `src/core/search/keyword_search_engine.py` | キーワード検索（Jaccard 類似度、Sudachi で名詞抽出） |
| `ChromaDBKeywordSearcher` | `src/core/search/chromadb_keyword_search.py` | ChromaDB ベースのキーワード検索 |
| `TextCombiner` | `src/core/search/text_combiner.py` | テキスト結合・パース |

### データベース管理

| クラス | モジュール | 責務 |
|--------|----------|------|
| `DynamicDBManager` | `src/utils/dynamic_db_manager.py` | 業務領域別ベクトルDB管理。参照ファイル（Excel）の更新日時を `data/vector_db/update_timestamps.json` に記録し、ファイルが更新されていなければDB再構築をスキップする（APIコスト削減） |
| `MetadataVectorDB` | `src/utils/vector_db.py` | ChromaDB 操作の共通インターフェース。LRUCache(max_size=10) でクライアントキャッシュ |
| `BusinessAreaTranslator` | `src/utils/business_area_translator.py` | 日本語業務名→英語コレクション名変換（YAML マッピング） |

### 埋め込みモデル

| クラス | モジュール | 責務 |
|--------|----------|------|
| `BaseEmbeddingModel` | `src/utils/base_embedding.py` | 抽象基底クラス。`encode()`, `embedding_dimension`, `provider_name` を定義 |
| `GeminiEmbeddingModel` | `src/utils/gemini_embedding.py` | VertexAI Gemini 埋め込み（gemini-embedding-001、3072次元） |
| `AzureEmbeddingModel` | `src/utils/azure_embedding.py` | Azure OpenAI 埋め込み（text-embedding-3-large、3072次元） |

### ハンドラー

| クラス | モジュール | 責務 |
|--------|----------|------|
| `InputHandlerFactory` | `src/handlers/input_handler.py` | 入力形式に応じたハンドラー生成（Excel / Hierarchical / MultiFolder / Text） |
| `OutputHandlerFactory` | `src/handlers/output_handler.py` | 出力形式に応じたハンドラー生成。`app_prefix` で出力サブディレクトリを指定 |

### 型定義

| 定義 | モジュール | 用途 |
|------|----------|------|
| `SearchResultDict` / `MultiStageSearchResultDict` | `src/types/search_types.py` | 検索結果の TypedDict |
| `SearchResultKeys` / `MetadataKeys` | `src/types/search_types.py` | Excel出力列名・ChromaDBメタデータキーの定数 |

### ユーティリティ

| モジュール | 責務 |
|----------|------|
| `src/utils/auth.py` | Google Cloud 認証（local / key_vault の2方式） |
| `src/utils/logger.py` | ログ設定（LOG_LEVEL 環境変数、ファイル + コンソール出力） |

---

## データフロー

### 通常バッチ処理フロー（回答支援AI）

```
1. 入力ファイル読み込み
   ↓ input_handler.py (load_data)
2. 参照データ読み込み + ベクトル化
   ↓ dynamic_db_manager.py → embedding → vector_db.py
3. 各行を検索実行
   ↓ processor.py → searcher.py → search_strategy.py
   ├─ OriginalSearchStrategy: 原文ハイブリッド検索（search_mode=original）
   │   ├─ vector_search_engine.py（ベクトル検索）
   │   └─ keyword_search_engine.py（キーワード類似度計算）
   └─ LLMEnhancedSearchStrategy: LLM拡張検索（search_mode=llm_enhanced）
       ├─ searcher.summarize_text()（LLMクエリ生成）
       ├─ vector_search_engine.py
       └─ keyword_search_engine.py
4. 結果出力
   ↓ output_handler.py (save_data)
5. Excelファイル保存
```

### 多段階検索バッチフロー（運用保守効率化AI）

```
1. 入力ファイル読み込み
   ↓ input_handler.py (load_data)
2. 参照データ読み込み + ベクトル化
   ↓ dynamic_db_manager.py → embedding → vector_db.py
3. 各行を検索実行（search_mode=multi_stage）
   ↓ multi_stage_orchestrator.py (execute)
   ├─ Stage 1: 原文でハイブリッド検索
   │   ├─ vector_search_engine.py
   │   └─ keyword_search_engine.py
   ├─ Stage 2: LLM強化クエリで検索
   │   ├─ query_enhancer.py
   │   ├─ vector_search_engine.py
   │   └─ keyword_search_engine.py
   └─ Stage 3: OR結合と3分類（Both / Original_Only / LLM_Enhanced_Only）
4. LLM判断支援（オプション、ThreadPoolExecutor で並列実行）
   ↓ judgment_support.py
5. 結果出力（3シート形式）
   ↓ output_handler.py (save_data_multi_stage)
6. Excelファイル保存
```

### 運用保守効率化AI 実行フロー

```
1. 変更前シナリオExcel配置
   └─ data/source/scenarios/revXXボット_シナリオデータ_YYYYMMDD.xlsx
2. DB再構築
   ↓ scripts/build_db.py --revisions-only
   ├─ Azure OpenAI でベクトル化
   │   └─ data/vector_db/revXX_{bot}/azure_openai/
   └─ VertexAI でベクトル化
       └─ data/vector_db/revXX_{bot}/vertex_ai/
3. 評価実行
   ↓ apps/revision-ops/run_eval.py
   ├─ 各改定内容をクエリとして検索
   ├─ 正解IDとの照合
   └─ Azure / VertexAI 横並び比較
4. Excel出力
   └─ data/output/latest/rev/rev_eval_batch_YYYYMMDD_HHMMSS.xlsx
```

---

## 依存関係

### モジュール間依存

```
main.py
  ├─ config.py (SearchConfig)
  ├─ src/core/processor.py
  │   ├─ src/handlers/input_handler.py
  │   ├─ src/handlers/output_handler.py
  │   ├─ src/core/searcher.py (検索統合・スコア計算)
  │   │   ├─ src/core/search/search_strategy.py (4戦略切替)
  │   │   ├─ src/core/search/multi_stage_orchestrator.py
  │   │   │   ├─ src/core/search/query_enhancer.py
  │   │   │   ├─ src/core/search/vector_search_engine.py
  │   │   │   │   └─ src/utils/vector_db.py
  │   │   │   ├─ src/core/search/keyword_search_engine.py
  │   │   │   ├─ src/core/search/chromadb_keyword_search.py
  │   │   │   └─ src/core/search/text_combiner.py
  │   │   ├─ src/core/search/vector_search_engine.py
  │   │   ├─ src/core/search/keyword_search_engine.py
  │   │   └─ src/core/search/text_combiner.py
  │   ├─ src/core/judgment_support.py
  │   └─ src/utils/dynamic_db_manager.py
  │       ├─ src/utils/vector_db.py
  │       ├─ src/utils/base_embedding.py
  │       │   ├─ src/utils/gemini_embedding.py
  │       │   │   └─ src/utils/auth.py
  │       │   └─ src/utils/azure_embedding.py
  │       └─ src/utils/business_area_translator.py
  └─ src/utils/logger.py
```

### 外部ライブラリ依存

| カテゴリ | ライブラリ | 用途 |
|---------|-----------|------|
| **Google Cloud** | google-genai | Vertex AI Embedding API |
| | google-auth | Google Cloud 認証 |
| **Azure** | openai | Azure OpenAI Embedding API |
| | azure-identity | Azure 認証 |
| | azure-keyvault-secrets | Key Vault 認証（オプション） |
| **ベクトルDB** | chromadb | ベクトルストレージ |
| **LangChain** | langchain-google-genai | Gemini LLM API |
| **NLP** | sudachipy | 日本語形態素解析 |
| **リトライ制御** | tenacity | API呼び出しの自動リトライ・指数バックオフ（5モジュールで使用） |
| **データ処理** | pandas | データフレーム操作 |
| | openpyxl | Excel読み書き |
| **UI** | streamlit | Web UI |

---

## 拡張性

### 新しい埋め込みモデルの追加

`BaseEmbeddingModel` 抽象基底クラス（`src/utils/base_embedding.py`）+ Factory パターン（`auth.py` の `create_embedding_model()`）で拡張する設計。新プロバイダー追加時の変更ファイル:

1. `src/utils/{provider}_embedding.py` — `BaseEmbeddingModel` を継承し `encode()`, `embedding_dimension`, `provider_name` を実装
2. `src/utils/auth.py` — `create_embedding_model()` にプロバイダー分岐を追加
3. `config.py` — `VALID_EMBEDDING_PROVIDERS` タプルにプロバイダー名を追加

### 新しい業務分野の追加

YAML マッピング（`config/business_areas.yaml`）+ ファイル自動検出（`DynamicDBManager.analyze_reference_files()`）により、**設定ファイルとデータファイルの配置だけで完結し、コード変更は不要**。

- `analyze_reference_files()` が `data/source/` を走査してファイル名から業務分野を自動検出する
- `BusinessAreaTranslator` が YAML マッピングで日本語→英語を変換する
- `extract_business_area_from_input()` が入力ファイル名から業務分野を抽出する
- DB パスは `data/vector_db/{英語名}/{provider}/` に自動生成される

操作手順は [ANSWER_SUPPORT.md の「業務分野の追加」](./ANSWER_SUPPORT.md#業務分野の追加) を参照。

### 新しい検索エンジンの追加

DI（コンストラクタ注入）パターンで拡張する設計。`MultiStageOrchestrator` は `vector_engine`, `keyword_engine`, `query_enhancer`, `text_combiner` を個別引数で受け取るため、新しいエンジンクラスを作成し、呼び出し元（`Searcher`）でコンストラクタに注入する。

### 認証方式

[CONFIGURATION.md の「GCP認証方式」](./CONFIGURATION.md#gcp認証方式) を参照。

### 外部API変更時の変更箇所

> 行番号は 2026-03-06 時点のもの。コード変更により行番号はずれる可能性があるため、クラス名・メソッド名を主キーとして参照すること。

| 変更対象 | 影響ファイル | 変更箇所の詳細 |
|---------|------------|---------------|
| Gemini LLM API | src/core/search/query_enhancer.py | QueryEnhancer._invoke_llm_with_retry()（L92）の LangChain 呼び出しを修正する。プロンプト形式が変わる場合は prompt/summarize_v1.0.txt も修正する |
| | src/core/judgment_support.py | JudgmentSupport._invoke_llm_with_retry()（L52）の LangChain 呼び出しを修正する。プロンプト形式が変わる場合は prompt/judgment_support.txt も修正する |
| Gemini Embedding API | src/utils/gemini_embedding.py | GeminiEmbeddingModel._get_embeddings_with_retry()（L131）のAPI呼び出しパラメータを修正する。次元数が変わる場合は embedding_dimension プロパティ（L185）も修正する |
| Azure OpenAI Embedding API | src/utils/azure_embedding.py | AzureOpenAIEmbeddingModel._get_embeddings_with_retry()（L146）のAPI呼び出しパラメータを修正する。認証方式が変わる場合は _setup_model()（L95）のクライアント初期化も修正する |
| ChromaDB API | src/utils/vector_db.py | MetadataVectorDB の全メソッド（add_documents L178、search L242、delete_collection L275 等）を修正する。クライアント初期化（L79）のパラメータも確認する |
| Google Cloud 認証 | src/utils/auth.py | _get_credentials_local()（L45）または _get_credentials_key_vault()（L70）を修正する。認証フローが変わる場合は get_google_credentials()（L103）のハンドラー選択ロジックも修正する |
| Sudachi（形態素解析） | src/core/search/keyword_search_engine.py | KeywordSearchEngine.extract_keywords()（L80）の形態素解析呼び出しを修正する。品詞体系が変わる場合はフィルタ条件も修正する |

---

## パフォーマンス最適化

### キャッシング戦略

1. **ベクトル化結果のキャッシュ**
   - タイムスタンプベースの検証
   - ChromaDB永続化

### 並列処理

1. **バッチベクトル化**
   - バッチサイズ: `EMBEDDING_BATCH_SIZE = 250`（Vertex AI / Azure 共通、`config.py`）
   - 並列API呼び出し

| 箇所 | max_workers | 用途 |
|------|-------------|------|
| `processor.py` | 10 | LLM判断支援の並列評価 |
| `ops_ui.py` | 2 | Azure/VertexAI プロバイダー並列検索 |
| `run_eval.py` | 5 | 複数エリアのDB検索並列実行 |

---

## セキュリティ考慮事項

### 認証フロー

```
1. 環境変数読み込み (.env)
   ↓
2. 認証情報取得
   ├─ Google Cloud: サービスアカウントキー
   └─ Azure: API キー
   ↓
3. API クライアント初期化
   ├─ auth.py (Google Cloud)
   └─ azure_embedding.py (Azure)
   ↓
4. API呼び出し
```

---

## テスト

`pytest` で実行（`tests/`, `pytest.ini`, `requirements-dev.txt`）。

### ユニットテスト

```
tests/unit/
├── test_business_area_mapping.py    # 業務分野マッピング・DB選択
├── test_chromadb_keyword_search.py  # ChromaDBキーワード検索
├── test_keyword_search_engine.py    # キーワード抽出・類似度計算
├── test_keyword_similarity_sync.py  # Searcher/Orchestrator間のキーワード同期
├── test_logger_dashboard.py         # ダッシュボードログ出力
├── test_logger_noise.py             # サードパーティログ抑制
├── test_run_eval_cache.py           # 評価キャッシュ動作
├── test_search_strategy.py          # 検索戦略パターン（4戦略）
├── test_text_combiner.py           # テキスト結合・パース
├── test_timestamp_migration.py      # タイムスタンプ移行
└── test_ui_shared.py               # UI共通ユーティリティ
```

### テスト概要

| 項目 | 値 |
|------|-----|
| テストフレームワーク | pytest |
| テストファイル数 | 11 |
| カバレッジ対象 | コアロジック + ハンドラー + ユーティリティ |

実行: `pytest tests/unit/ -v`
カバレッジ: `pytest tests/unit/ --cov=src --cov-report=term-missing`
