# ARCHITECTURE.md レビューレポート

**レビュー日**: 2026-03-03
**対象ファイル**: `rag-local/docs/ARCHITECTURE.md` (1428行)
**レビュー観点**: 正確性、APIシグネチャ、データフロー、定数値、プロンプト、設計書鮮度、引き継ぎ適性

---

## Critical (事実誤認 — コードと不一致)

### C-1: JudgmentSupport の `analyze_relevance` メソッドは存在しない

**箇所**: L195-204 (レイヤー構造セクション)

ドキュメントは以下のように記載:
```python
def analyze_relevance(self, query: str, result: SearchResult) -> JudgmentResult:
    """関連性を分析"""
    # LLMで判定: 関連あり / 要確認 / 関連なし
```

**実際のコード** (`src/core/judgment_support.py:54`):
```python
def evaluate(
    self, revision_content: str, search_result_q: str, search_result_a: str
) -> Dict[str, str]:
```

- メソッド名が `analyze_relevance` ではなく `evaluate`
- 引数が `(query, result)` ではなく `(revision_content, search_result_q, search_result_a)`
- 戻り値型が `JudgmentResult` ではなく `Dict[str, str]`

なお、APIリファレンスセクション(L901-924)では `evaluate` メソッドが正しく記載されている。**レイヤー構造セクションのみ古い情報が残存**。

### C-2: OutputHandler.__init__ の `app_prefix` 引数が欠落

**箇所**: L1210-1218 (APIリファレンス, OutputHandler)

ドキュメントは:
```python
def __init__(self, config: SearchConfig):
```

**実際のコード** (`src/handlers/output_handler.py:11`):
```python
def __init__(self, config: SearchConfig, app_prefix: str = ""):
```

`app_prefix` は出力ディレクトリのサブフォルダ指定に使用される重要な引数。OutputHandlerFactory.create() も同様に `app_prefix` を受け取る。

### C-3: OutputHandlerFactory.create() の `app_prefix` 引数が欠落

**箇所**: L1225-1230

ドキュメントは:
```python
def create(output_type: str, config: SearchConfig) -> OutputHandler:
```

**実際のコード** (`src/handlers/output_handler.py:443`):
```python
def create(output_type: str, config: SearchConfig, app_prefix: str = "") -> OutputHandler:
```

### C-4: DynamicDBManager.__init__ の `self.translator` は `self._translator`

**箇所**: L253-258 (レイヤー構造セクション)

ドキュメントは:
```python
self.translator = BusinessAreaTranslator()
```

**実際のコード** (`src/utils/dynamic_db_manager.py:32`):
```python
self._translator = BusinessAreaTranslator()
```

private 属性であり、外部から直接アクセスすべきでない。ドキュメントが public と誤解させる可能性がある。

### C-5: レイヤー図の `utils.py` は存在しない

**箇所**: L74 (全体構成図)

```
│   │   └─ utils.py - ユーティリティ関数                │
```

`src/utils/utils.py` は存在しない。`src/utils/` の実際の構成:
- `auth.py` — Google Cloud認証
- `azure_embedding.py` — Azure OpenAI埋め込み
- `base_embedding.py` — 埋め込みモデル基底
- `business_area_translator.py` — 業務領域変換
- `dynamic_db_manager.py` — 動的DB管理
- `gemini_embedding.py` — Gemini埋め込み
- `logger.py` — ログ設定
- `vector_db.py` — ChromaDB ラッパー

### C-6: Utils Layer のレイヤー図でデータベース管理の分類が不正確

**箇所**: L57-76 (全体構成図)

レイヤー図では `dynamic_db_manager.py` と `vector_db.py` を「Utils Layer > データベース管理」の下に、`business_area_translator.py` を同列に配置しているが、CLAUDE.md のディレクトリ構造（およびコード配置の実態）では `src/utils/` 直下にフラットに存在する。レイヤー図のサブカテゴリ分類自体は概念的に正しいが、「データベース管理」の中に `business_area_translator.py` が含まれている記述はやや誤解を招く。

---

## Important (情報不足 — 文書化されていない機能・設定)

### I-1: `src/types/` ディレクトリが文書化されていない

**箇所**: レイヤー構造セクション全般

`src/types/search_types.py` が存在し、検索関連の型定義（TypedDict 等）を提供しているが、ARCHITECTURE.md のレイヤー図・APIリファレンスのいずれにも記載がない。CLAUDE.md には記載あり。

### I-2: InputHandler のサブクラス群が文書化されていない

**箇所**: L211-231, L1169-1202 (Handler Layer / APIリファレンス)

ドキュメントでは `InputHandler` 基底クラスと `InputHandlerFactory` のみ記載。実際には以下のサブクラスが存在:
- `ExcelInputHandler` — Excel入力（回答支援AI用）
- `HierarchicalExcelInputHandler` — 階層構造Excel（シナリオ直接入力）
- `MultiFolderInputHandler` — 複数フォルダ参照（回答支援AI標準）
- `TextInputHandler` — テキスト/Excel入力（改定影響調査用）

ExcelOutputHandler の `save_data_multi_stage()`, `save_data_dual_provider()` メソッドも未記載。

### I-3: run_eval.py の max_workers 記載が不完全

**箇所**: L605

ドキュメントは:
```
run_eval.py: ThreadPoolExecutor(max_workers=5) で複数エリアの検索を並列実行
```

実際には run_eval.py には3箇所の ThreadPoolExecutor がある:
- L567: `max_workers=5` (エリア並列)
- L582: `max_workers=5` (エリア並列)
- L691: `max_workers=2` (プロバイダー並列)

### I-4: `config/` ディレクトリが文書化されていない

`config/business_areas.yaml` と `config/settings.yaml` がレイヤー図に含まれていない。`business_areas.yaml` は拡張性セクション(L475)で言及されているが、`settings.yaml` はSearchConfigの説明内で間接的に参照されているのみ。

### I-5: MetadataVectorDB の LRUCache 機構が文書化されていない

**箇所**: L1044-1068

`MetadataVectorDB` は `LRUCache(max_size=10)` でクライアントキャッシュを行っているが、APIリファレンスでは言及されていない。並行アクセス時の `_cache_lock` の存在も未記載。

### I-6: SearchStrategy の4戦略の概要は記載されているが、各戦略の execute() シグネチャが統一的に文書化されていない

**箇所**: L36 (レイヤー図) vs 実コード

4戦略は記載されているが、各クラスの用途説明が以下の場所に分散:
- レイヤー図(L36): 列挙のみ
- データフロー(L319-326): Original, LLMEnhanced のみ
- APIリファレンス: SearchStrategy の項目なし

`KeywordFilterSearchStrategy` のデータフロー記載が不足。

### I-7: `_get_column_names()` が `settings.yaml` のカラム設定を使用していることが未記載

`InputHandler` の説明で、入力ファイルのカラム名が `settings.yaml` から動的に解決されることが記載されていない。コミット `ea62203` で追加された機能。

---

## Minor (文体・明瞭性・体裁の改善提案)

### M-1: レイヤー構造セクションとAPIリファレンスセクションの重複

L100-306（レイヤー構造）とL652-1302（APIリファレンス）で同じクラスの情報が二重に記載されている。レイヤー構造の簡易コード例はAPIリファレンスと矛盾するリスクがある（C-1 が好例）。

**提案**: レイヤー構造セクションはコードスニペットを削除し、責務の説明とAPIリファレンスへのリンクのみにする。

### M-2: Entry Points の `scripts/` 説明が曖昧

**箇所**: L108

```
| `scripts/*` | ユーティリティスクリプト | DB再構築、評価実行 |
```

実際の scripts/ には5本のスクリプトがある:
- `build_db.py` — DB構築
- `generate_correct_ids.py` — 正解ID生成
- `prepare_before_scenario.py` — データ前処理
- `check_db_content.py` — DB内容確認
- `create_handover_package.py` — 引き継ぎパッケージ作成

「評価実行」は scripts/ ではなく `apps/revision-ops/run_eval.py` の責務。

### M-3: テストファイル一覧が正確

**箇所**: L636-648

テストファイル一覧は実際のファイルと一致しており正確。ただし、テストの合計件数やカバレッジ情報があると引き継ぎ時に有用。

### M-4: 拡張性セクションの `VALID_EMBEDDING_PROVIDERS` の例が架空

**箇所**: L457

```python
VALID_EMBEDDING_PROVIDERS: Tuple[str, ...] = ("vertex_ai", "azure_openai", "custom")
```

これは「追加する場合の例」として書かれているが、現在の値が `("vertex_ai", "azure_openai")` であることの明記がない。

### M-5: 外部ライブラリ依存テーブルに `tenacity` (リトライ) が欠落

**箇所**: L407-421

`judgment_support.py` で `tenacity` ライブラリの `@retry` デコレータが使用されているが、外部ライブラリ依存テーブルに記載がない。

---

## docs/plans/ 設計書の鮮度チェック

| 設計書 | 状態 | 提案 |
|--------|------|------|
| `2026-03-02-code-simplification-design.md` | **未実装** — 設計レビュー完了、実装は未着手 | 維持（アクティブ計画） |
| `2026-03-02-code-simplification-plan.md` | **未実装** — 14件の指摘に対する5 Phase実装計画 | 維持（アクティブ計画） |
| `2026-03-02-terminal-log-redesign.md` | **部分実装** — logger.py のノイズ抑制は実装済み（`test_logger_noise.py` 存在）、ダッシュボード出力は `test_logger_dashboard.py` 存在で進行中 | 維持（アクティブ計画、進行中と明記推奨） |

**注**: ARCHITECTURE.md では `docs/plans/` について「performance-improvement」設計書に言及していないが、実際にも存在しない。設計書のインデックスを ARCHITECTURE.md に追加すると引き継ぎに有用。

---

## 引き継ぎ適性評価

### 良い点
1. **全体構成図が明快** — Entry Points / Core / Handler / Utils / External Services / Data Storage のレイヤー図は初見でシステムの概要を掴みやすい
2. **データフロー図が3パターン網羅** — 通常バッチ/多段階/改定影響調査の3フローが段階的に記載
3. **拡張ガイドが実用的** — 新規業務分野追加の Step 1-5 が「コード変更不要」の理由まで含めて丁寧
4. **プロンプトファイルの用途マップ** — AI使用箇所マップが呼び出し元まで記載
5. **APIリファレンスが充実** — 主要クラスのシグネチャ・引数・使用例が揃っている

### 改善が必要な点
1. **レイヤー構造とAPIリファレンスの二重記載** — 同じ情報が2箇所にあり、片方が古くなりやすい（C-1 が実例）
2. **utils.py ゴースト参照** — 存在しないファイルへの参照は混乱を招く
3. **OutputHandler の `app_prefix`** — 出力先のサブフォルダ制御は運用上重要だが未記載
4. **型定義(types/)の欠落** — 検索結果の型が定義されているが文書化なし

### 総合評価

初見のエンジニアがシステムの全体像と各モジュールの責務を理解するには**十分な品質**。特に拡張性セクションの実用性が高い。ただし、Critical 指摘6件の修正が必要。特に C-1 (JudgmentSupport の誤ったメソッド名) と C-5 (存在しない utils.py) は引き継ぎ時の混乱要因となる。

---

## 修正優先度まとめ

| 優先度 | 件数 | 主な内容 |
|--------|------|----------|
| **Critical** | 6件 | C-1〜C-6: メソッド名不一致、引数欠落、存在しないファイル参照 |
| **Important** | 7件 | I-1〜I-7: 未文書化の型定義、サブクラス、設定ファイル |
| **Minor** | 5件 | M-1〜M-5: 二重記載、ライブラリ欠落、説明の曖昧さ |
