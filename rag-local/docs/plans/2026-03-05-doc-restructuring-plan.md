# ドキュメント再構成 実装計画

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Diataxis + SSOT 原則に基づき、rag-local ドキュメント6ファイルの重複を排除し、各ファイルの責務を明確化する。

**Architecture:** 各情報の SSOT を1箇所に定め、他の箇所はリンク参照に置換。ファイル数は現行6ファイルを維持。ARCHITECTURE.md の API リファレンス（~700行）を主要クラス一覧表（~80行）に圧縮するのが最大の変更。

**Tech Stack:** Markdown のみ。コード変更なし。

**設計書:** `docs/plans/2026-03-05-doc-restructuring-design.md`

**注意:** ANSWER_SUPPORT.md の末尾（設定パラメータ → CONFIGURATION.md リンク、トラブルシューティング → TROUBLESHOOTING.md リンク）はユーザーが既に変更済み。この変更を維持すること。

---

## Task 1: ARCHITECTURE.md — API リファレンス圧縮と重複削除

最大の変更（1,518行 → ~600行）を最初に実施。

**Files:**
- Modify: `docs/ARCHITECTURE.md`

**Step 1: API リファレンスセクション（700行超）を主要クラス一覧表に置換**

`## API リファレンス` セクション（行700〜行1498付近）を以下の表に置換する:

```markdown
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
| `DynamicDBManager` | `src/utils/dynamic_db_manager.py` | 業務領域別ベクトルDB管理。タイムスタンプ検証による差分更新 |
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
```

**Step 2: プロンプトセクション削除**

`## プロンプト` セクション（AI使用箇所マップ + プロンプトファイル詳細）を削除。理由: README.md の「AI使用箇所マップ」表が SSOT。プロンプトファイルの説明は2ファイルしかなく、ファイル自体を読めば分かる。

**Step 3: 「規約と注意事項」セクション削除**

5項目の箇条書きだが、各項目は他ドキュメントに既出の情報。削除。

**Step 4: 「関連ドキュメント」セクション削除**

末尾の5行リンクリスト削除。README.md のドキュメント一覧表が SSOT。

**Step 5: レイヤー構造セクションの圧縮**

`## レイヤー構造` の各レイヤー説明にあるコード例（class 定義の抜粋）を削除。主要クラス一覧表で責務を示しているため冗長。各レイヤーは表形式の概要のみ残す。

ただし `### 2. Core Layer` 内の以下は維持:
- SearchStrategy の4戦略テーブル（`search_strategy.py` の戦略パターンが一覧で分かる）

**Step 6: 拡張性セクションの圧縮**

- 「新しい埋め込みモデルの追加」: コード例を最小限に（3ステップの手順 + 1つのコード例のみ）
- 「新しい業務分野の追加」: Step 1-5 の手順は維持するが、各ステップの説明を圧縮
- 「新しい検索エンジンの追加」: MultiStageOrchestrator の注入説明を2行に圧縮

**Step 7: 確認**

- 目次を再構成後のセクションに合わせて更新
- 内部リンク（`#セクション名`）が正しいか確認
- 目標: ~600行

**Step 8: コミット**

```bash
git add docs/ARCHITECTURE.md
git commit -m "docs: restructure ARCHITECTURE.md — compress API reference to class table, remove duplicates"
```

---

## Task 2: ANSWER_SUPPORT.md — 重複削除とリンク化

**Files:**
- Modify: `docs/ANSWER_SUPPORT.md`

**注意:** 末尾の「設定パラメータ」「トラブルシューティング」はユーザーが既にリンク形式に変更済み。この変更を維持すること。

**Step 1: DB構築セクションの圧縮**

`## DB構築` セクション（行242〜行312付近）を以下に置換:

```markdown
## DB構築

統合スクリプト `scripts/build_db.py` で回答支援AI用 DB を構築する。セットアップ手順・全コマンドは [README.md の Step 5](../README.md#step-5-db構築) を参照。

回答支援AI固有のオプション:

```bash
# 回答支援AI用DBのみ構築（改定別を除外）
python scripts/build_db.py --no-revisions

# 差分のみ構築（未構築 or 参照ファイル更新時のみ）
python scripts/build_db.py --no-revisions
```

> **注意**: DB構築中は Streamlit UI を停止してください（ChromaDB のファイルロック競合を防止）。
```

これにより以下を削除:
- build_db.py 引数の完全な表（README.md が SSOT）
- スキップロジックの説明（README.md が SSOT）
- 参照データの配置ディレクトリツリー（README.md Step 4 が SSOT）
- ファイル命名規則の重複（README.md Step 4 が SSOT）
- 前提条件3項目（README.md + CONFIGURATION.md が SSOT）

**Step 2: 埋め込みプロバイダー表のリンク化**

`### 埋め込みプロバイダー（文章を数値に変換するサービス）` の4行テーブル + 説明文を以下に置換:

```markdown
### 埋め込みプロバイダー

2つのプロバイダー（Azure OpenAI / VertexAI）に対応。モデル・次元数の詳細は [ARCHITECTURE.md](./ARCHITECTURE.md#埋め込みモデル) を参照。

`build_db.py` は両プロバイダーの DB を構築する。実行時は `DEFAULT_EMBEDDING_PROVIDER` 環境変数に一致するプロバイダーの DB が使用される。
```

**Step 3: 確認**

- 内部リンクが正しいか確認
- ユーザーの既存変更（末尾の設定パラメータ・トラブルシューティング）が維持されているか確認
- 目標: ~250行

**Step 4: コミット**

```bash
git add docs/ANSWER_SUPPORT.md
git commit -m "docs: ANSWER_SUPPORT.md — replace duplicates with SSOT links"
```

---

## Task 3: REVISION_OPS.md — 重複削除とリンク化

**Files:**
- Modify: `docs/REVISION_OPS.md`

**Step 1: スコア計算セクションのリンク化**

`## 多段階ハイブリッド検索` 内の `### スコア計算` を以下に置換:

```markdown
### スコア計算

スコア計算式は回答支援AIと共通。詳細は [ANSWER_SUPPORT.md のスコア計算式](./ANSWER_SUPPORT.md#スコア計算式) を参照。

改定影響調査でのデフォルト設定値:
```

（設定値テーブルはそのまま維持）

**Step 2: プロバイダー表のリンク化**

`### プロバイダー別DBの理由` 内のプロバイダー表（Azure/VertexAI の埋め込みモデル・次元数）を以下に置換:

```markdown
### プロバイダー別DBの理由
- 文章を数値に変換した結果（ベクトル）の特性がプロバイダーによって異なる
- 同じデータベースに異なるモデルで変換した数値は混在できない
- 検索時は、DB構築時と同じモデルで質問文を変換する必要がある

対応プロバイダーの詳細は [ANSWER_SUPPORT.md](./ANSWER_SUPPORT.md#埋め込みプロバイダー) を参照。
```

**Step 3: 参照データ管理セクションの圧縮**

`## 参照データ管理` のフォルダ構造を圧縮。`reference/` の詳細ツリー（マージ版シナリオ/問い合わせ履歴 等）は改定シナリオのツリーのみに絞る:

```markdown
## 参照データ管理

改定評価に必要な参照データは `reference/` ディレクトリに格納（git管理外、別途提供）。

```
reference/
├── 改定内容/                       # 改定内容の説明 (revXX_*.md)
├── 改定シナリオ/
│   ├── rev01_スマイル機能変更/
│   │   ├── 差分.md                 # 統一フォーマットの差分ファイル
│   │   ├── 修正前/
│   │   └── 修正後/
│   ├── rev02_相続少額払い/
│   └── ...（rev03〜rev06 同様）
└── シナリオボットメンテナンス管理台帳.xlsx
```
```

削除対象: `マージ版シナリオ/`、`問い合わせ履歴/`、`参考資料/` の詳細ツリー。

**Step 4: 「関連ファイル」セクションと末尾の関連ドキュメント削除**

`## 関連ファイル` テーブル（12行）を削除。理由: ARCHITECTURE.md の主要クラス一覧表と重複。

**Step 5: 確認**

- 内部リンクが正しいか確認
- 目標: ~480行

**Step 6: コミット**

```bash
git add docs/REVISION_OPS.md
git commit -m "docs: REVISION_OPS.md — replace duplicates with SSOT links, compress reference section"
```

---

## Task 4: CONFIGURATION.md — 末尾の重複削除

**Files:**
- Modify: `docs/CONFIGURATION.md`

**Step 1: 末尾トラブルシューティングの削除**

`## トラブルシューティング` セクション（行591〜行618付近、「環境変数が読み込まれない」「APIキーエラー」の2項目）を以下に置換:

```markdown
## トラブルシューティング

設定に関する問題は [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) を参照。
```

**Step 2: 末尾「関連ドキュメント」セクション削除**

行622〜行630 の「関連ドキュメント」リンクリストを削除。

**Step 3: コミット**

```bash
git add docs/CONFIGURATION.md
git commit -m "docs: CONFIGURATION.md — remove duplicate troubleshooting and related docs"
```

---

## Task 5: README.md — 環境変数表の圧縮と引き継ぎパッケージの整理

**Files:**
- Modify: `README.md`

**Step 1: 環境変数テーブルの圧縮**

Step 3 の環境変数テーブル（6行テーブル）を主要3変数に絞り、リンクを追加:

```markdown
`.env` を編集して以下の主要変数を設定:

| 変数 | 説明 | 例 |
|------|------|-----|
| `DEFAULT_LLM_PROVIDER` | LLMプロバイダー（`gemini` のみ） | `gemini` |
| `DEFAULT_LLM_MODEL` | LLMモデル名 | `gemini-2.5-flash-lite` |
| `DEFAULT_EMBEDDING_PROVIDER` | 埋め込みプロバイダー | `azure_openai` |

認証情報（Azure OpenAI / Google Cloud）を含む全変数の詳細は [docs/CONFIGURATION.md](./docs/CONFIGURATION.md) を参照。
```

**Step 2: 引き継ぎパッケージセクションの圧縮**

`### 同梱するもの` の後にあるフラグ表（`--include-data` 等の4行テーブル）と Note を圧縮:

```markdown
引き継ぎパッケージの作成には `scripts/create_handover_package.py` を使用（許可リスト方式で秘密情報の混入を防止）:

```bash
# 事前確認（コピーせず対象一覧とサイズを表示）
python scripts/create_handover_package.py ./handover_package --dry-run

# パッケージ作成（コードのみ）
python scripts/create_handover_package.py ./handover_package

# パッケージ作成（data/source/ の実データも含む）
python scripts/create_handover_package.py ./handover_package --include-data
```

全オプションは `--help` を参照。出力先ディレクトリが既に存在する場合はエラーになります。
```

**Step 3: コミット**

```bash
git add README.md
git commit -m "docs: README.md — compress env var table and handover package section"
```

---

## Task 6: TROUBLESHOOTING.md — 末尾の関連ドキュメント削除

**Files:**
- Modify: `docs/TROUBLESHOOTING.md`

**Step 1: 末尾「関連ドキュメント」セクション削除**

行640〜行649 の「関連ドキュメント」リンクリストを削除。

**Step 2: コミット**

```bash
git add docs/TROUBLESHOOTING.md
git commit -m "docs: TROUBLESHOOTING.md — remove duplicate related docs section"
```

---

## Task 7: 最終確認とスカッシュコミット

**Step 1: 全ファイルの行数確認**

```bash
wc -l README.md docs/ANSWER_SUPPORT.md docs/REVISION_OPS.md docs/CONFIGURATION.md docs/ARCHITECTURE.md docs/TROUBLESHOOTING.md
```

目標:
- README.md: ~300行
- ANSWER_SUPPORT.md: ~250行
- REVISION_OPS.md: ~480行
- CONFIGURATION.md: ~530行
- ARCHITECTURE.md: ~600行
- TROUBLESHOOTING.md: ~600行
- 合計: ~2,760行（現在4,142行から33%削減）

**Step 2: リンク整合性チェック**

全ドキュメント内のリンク先（`#セクション名`）が実在するか確認:

```bash
grep -n '\](#' docs/*.md README.md
```

**Step 3: git diff --stat で変更サマリを確認**

```bash
git diff --stat HEAD~6
```
