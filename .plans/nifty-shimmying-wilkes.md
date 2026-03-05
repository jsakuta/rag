# ドキュメント残余修正計画 v2（コード検証済み）

## Context

ドキュメント総合レビューで Critical 20件は全て修正済み。
残り Important + Minor のうち、コード調査で **修正不要** と確認された項目を除いた **40件** を修正する。

## 修正不要と確認された項目（16件）

| ID | 理由 |
|----|------|
| Important #1 (X-1) | Critical修正で対応済み（search_mode × search_type 2軸） |
| Important #3 (X-5) | Critical修正で対応済み（Excel出力列構成） |
| Important #4 (I-1 入力Excel) | Critical修正で対応済み |
| Important #6 (I-4 hybrid固定) | Critical修正で対応済み |
| Important #10 (I-04 テスト実行) | Critical修正で対応済み |
| Important #35 (I-01 search_source) | b8f0d81 コミットで対応済み（L319-332に記載） |
| I-R07 (サマリー改定内容) | ドキュメント L227 に「全文」と記載済み、実装と一致 |
| I-T02 (--limit/--business) | main.py:50 で argparse 実在確認済み |
| M-05 (README Step参照) | Step 1-6 の参照は正確、不整合なし |
| M-04 (CONFIG dotenv) | L520 に `load_dotenv()` 含有済み |
| M-T02 (ポート番号) | 問題なし |
| M-T03 (Vertex AI関数名) | SDK移行完了済み、記載は正確 |
| M-1 (キーワードTop-5) | デフォルト5でパラメータ化済み、正確 |
| X-6 (読み順) | 概ね妥当 |
| M-3 (処理フロー図) | 調査で重大な問題なし |
| M-R03 (コマンド順序) | I-R04/M-R02 のパス修正に含まれる |

## 方針

- 6ドキュメントを並列エージェント（3並列 × 2バッチ）で修正
- 各エージェントに **検証済みの正確な修正指示** を渡す
- 行番号は現在のドキュメント状態に基づく（2026-03-04 時点）

---

## Batch 1: 3並列

### Agent A: ANSWER_SUPPORT.md（5件）

#### A-1: I-5 バッチ実行時のDB自動更新動作が未記載

**コード根拠:** `main.py:192-194` で `run_db_update(config, business_filter)` を呼び出し。
`run_db_update()` (L92-119): DynamicDBManager → analyze_reference_files(include_revisions=False) → 業務分野ごとに update_business_db()。参照ファイル未更新ならスキップ。

**操作:** L148（Note の直後、L149 の前）に以下を挿入:

```markdown
> **Note:** バッチ処理実行時、DB は自動更新されます。実行フロー：
> 1. `run_db_update()` で参照ファイル（FAQ / シナリオ）の更新を検出
> 2. 参照ファイル未更新 + DB既存 → スキップ（API コスト発生なし）
> 3. 参照ファイル更新あり or DB未存在 → 構築/更新実行
> 4. DB 更新完了後、バッチ処理を開始
>
> UI（インタラクティブ）モードでは DB更新を実行しません。
```

---

#### A-2: I-3 UIセッション内設定変更がYAMLに保存されない

**コード根拠:** `chat.py:264-295` で `st.session_state.config.vector_weight = weight` 等、session_state のみに保持。YAML への書き戻しは行われない。

**操作:** L196（「チャット履歴保存」行の直後、「### プレフライト検証」セクションの前）に以下を挿入:

```markdown
> **Note:** UI 内で変更したパラメータ（ベクトル重み、検索モード等）は、セッション内のメモリのみに保持されます。
> `config/settings.yaml` には保存されないため、UI を閉じると変更は失われます。永続化が必要な場合は、YAML を直接編集してください。
```

---

#### A-3: I-2 keyword_weight プロパティの説明不足

**コード根拠:** `config.py:301-304`
```python
@property
def keyword_weight(self) -> float:
    """vector_weight から自動計算（常に 1.0 - vector_weight）"""
    return 1.0 - self.vector_weight
```

**操作:** L67（`keyword_similarity` の説明行の直後）に以下を挿入:

```markdown
- `keyword_weight`: 1.0 - `vector_weight` で自動計算（手動設定不可）。デフォルト 0.1
```

---

#### A-4: M-2 multi_stage Note の簡潔化

**現在:** L57 に multi_stage Note がある。内容は有用だが冗長。

**操作:** L57 の既存 Note を以下に置換:

```markdown
> **Note:** `multi_stage` モードは改定影響調査専用です。このシステムでは使用しないでください。
```

---

#### A-5: M-4 チャット履歴とバッチ出力の列差異

**コード根拠:**
- チャット履歴（chat.py:206-227）: 8列（Input_Number, Original_Query, Search_Query, Search_Result_Q, Search_Result_A, Similarity, Vector_Weight, Top_K）
- バッチ出力（output_handler.py:74-81）: 12列（上記 + Original_Answer, Scenario_ID, Sheet_Name, Row_Index）

**操作:** L196 付近（チャット履歴保存の説明直後）に以下を挿入:

```markdown
> **Note:** チャット履歴とバッチ出力で列構成が異なります:
> - チャット履歴: 8列（`Original_Answer`, `Scenario_ID`, `Sheet_Name`, `Row_Index` を省略）
> - バッチ出力: 12列（全列を含む詳細分析用）
```

---

### Agent B: REVISION_OPS.md（8件）

#### B-1: I-R01 business_areas.yaml と settings.yaml の二重登録の理由

**コード根拠:**
- `business_area_translator.py:77-82`: `revision_mappings` = ChromaDB コレクション名の物理マッピング
- `settings.yaml:158-189`: `evaluation.revision_areas` = 検索パラメータ（search_type, vector_weight）
- `run_eval.py:85-99`: 両設定を独立して読み込み

**操作:** L488-491 の Step 2 冒頭を以下に置換:

```markdown
### Step 2: マッピング登録（2箇所）

**2つの設定ファイルの役割:**
- `business_areas.yaml` の `revision_mappings`: エリア名と ChromaDB コレクション名の**物理マッピング**（DB検索時の名前解決に使用）
- `settings.yaml` の `evaluation.revision_areas`: 改定番号と検索設定の**ロジックマッピング**（search_type, vector_weight 等のパラメータ）

**config/business_areas.yaml** — `revision_mappings` に追加:
```

---

#### B-2: I-R02 keyword_filter で多段階検索が適用されない

**コード根拠:**
- `search_strategy.py:232-316`: KeywordFilterSearchStrategy.execute() はベクトル検索を使わず ChromaDB キーワードキャッシュのみ使用
- `run_eval.py:291-332`: `_execute_keyword_filter_search()` で ChromaDBKeywordSearcher を使用、Stage 1/2/3 は実行しない

**操作:** L57（フィルタリング行の後）に以下を挿入:

```markdown

### 補足: search_type による動作の違い

- **hybrid**: 多段階検索フロー（Stage 1/2/3）を実行。ベクトル+キーワードのハイブリッド検索
- **keyword_filter**: ベクトル検索をスキップ。ChromaDB キーワードキャッシュのみ使用（多段階検索は実行しない）。用語の単純置換（AML→GPLEX等）の検出に適する
```

---

#### B-3: I-R04 + M-R02 `reference/変更前シナリオ/` パス修正

**実態:** `reference/変更前シナリオ/` は存在しない。正しくは `reference/改定シナリオ/revXX_*/修正前/`。

**操作:** L411 のフロー図内のパスを修正:

```
2. 変更前シナリオ (reference/改定シナリオ/revXX_*/修正前/revXX_変更前シナリオ_XXX-bot.xlsx)
```

同一パスが他に出現する箇所も同様に修正。

---

#### B-4: I-R05 rev07 が文書化されていない

**実態:** `reference/改定内容/rev07_積立定期預金_未評価.md` が存在。

**操作:** L105-112 の改定番号テーブルに以下を行追加:

```markdown
| ⑦ | - | 積立定期預金 | smile-bot | rev07_smile（未評価） |
```

---

#### B-5: I-R06 入力ファイルの「変更内容」列が文書化されていない

**コード根拠:**
- `run_eval.py:128-136`: `if "変更内容" not in df.columns: df["変更内容"] = ""`
- `run_eval.py:780-784`: 未発見シナリオの詳細情報に使用
- `run_eval.py:1171`: 出力キー `["シナリオID", "変更内容", ...]`

**操作:** L142（正解ID列の後）に以下を追記:

```markdown
- `変更内容`（オプション）: 各正解IDごとの具体的な変更内容。未検出シナリオ（「未発見」セクション）に表示される。省略時は自動的に空で初期化
```

---

#### B-6: I-R03 keyword_filter 時の vector_weight 出力値

**コード根拠:** `search_strategy.py:303` で `'Vector_Weight': 0.0` を出力。
**現在:** L251 に「keyword_filter時は `-`」と記載 → 実装は `0.0`。

**操作:** L251 を以下に修正:

```markdown
| 7 | ベクトル重み | vector_weight（hybrid時のみ有効、keyword_filter時は `0.0`）（1行目のみ） |
```

---

#### B-7: M-R01 FILTER_MODE デフォルト値のコード/YAML区別

**コード根拠:**
- `settings.yaml:134`: `filter_mode: top_k`
- `multi_stage_orchestrator.py:53`: `filter_mode: str = "threshold"` (コードデフォルト)
- `run_eval.py:80`: `FILTER_MODE = _settings.get("filter_mode", "threshold")` (フォールバック)

**操作:** L62 の FILTER_MODE 行を以下に修正:

```markdown
| FILTER_MODE | top_k | フィルタリング方式（`top_k` / `threshold`）。settings.yaml で指定、コード上のフォールバックは `threshold` |
```

---

### Agent C: CONFIGURATION.md（8件）

#### C-1: I-02 SearchConfig の未記載フィールド

**コード根拠:** `config.py:135-188` の SearchConfig に 30+ フィールド。CONFIGURATION.md には主要なもののみ記載。

**操作:** L182（コード例の直後）に以下のテーブルを挿入:

```markdown
### SearchConfig の主要フィールド一覧

| フィールド | 型 | デフォルト値 | 説明 |
|-----------|-----|-----------|------|
| `top_k` | int | batch=4, ui=3 | 返却する結果数 |
| `vector_weight` | float | 0.9 | ベクトル検索の重み（0.0〜1.0） |
| `search_type` | str | `hybrid` | 検索アルゴリズム（`hybrid` / `keyword_filter`） |
| `search_mode` | str | `original` | クエリ処理（`original` / `llm_enhanced` / `multi_stage`） |
| `search_source` | str | `history_data` | 検索対象（`scenario` / `history_data`） |
| `reference_type` | str | `multi_folder` | 参照データ形式 |
| `multi_stage_threshold` | float | 0.45 | 多段階検索の統合スコア閾値 |
| `multi_stage_max_results` | int | 100 | 多段階検索の各検索結果最大数 |
| `multi_stage_enable_judgment_support` | bool | True | LLM判断支援の有効化 |
| `include_hierarchy_in_vector` | bool | True | 階層情報をベクトル化に含める |
| `force_db_update` | bool | False | 強制DB更新フラグ |
| `embedding_provider` | str | 環境変数 | 埋め込みプロバイダー |
| `llm_provider` | str | 環境変数 | LLMプロバイダー（`gemini` のみ） |
| `credential_source` | str | `local` | GCP認証方式（`local` / `key_vault`） |
```

---

#### C-2: I-06 evaluation セクションの詳細

**コード根拠:** `settings.yaml:116-244` に全パラメータ。

**操作:** L307 付近（evaluation の簡潔な説明の後）に以下を挿入:

```markdown
### 改定影響調査（evaluation）の詳細パラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `max_results` | int | 100 | 最大検索結果数 |
| `filter_mode` | str | `top_k` | フィルタリング（`threshold` / `top_k`） |
| `top_k` | int | 130 | `filter_mode: top_k` 時の上位K件数 |
| `thresholds.azure_openai` | float | 0.40 | Azure OpenAI 類似度閾値 |
| `thresholds.vertex_ai` | float | 0.50 | VertexAI 類似度閾値 |
| `enable_judgment_support` | bool | true | LLM判断支援の有効化 |

**revision_areas:**
改定番号ごとに `areas`（検索対象DBエリア）、`search_type`、`vector_weight` を指定。

**area_to_bot / area_to_category:**
エリア名からボット名・日本語カテゴリ名への変換マッピング。
```

---

#### C-3: I-07 smile_tablet マッピングの説明

**コード根拠:** `business_area_translator.py:102-150` で `translate()` メソッドが使用。完全一致→部分一致の線形検索。

**操作:** L376（business_areas.yaml 説明セクション）の後に以下を追加:

```markdown
> **Note:** `smile_tablet`（スマイルタブレット）は `smile`（スマイル）と別の業務分野マッピングです。現在の回答支援AIでは `smile` に統合されていますが、将来的なタブレット専用DB拡張時に活用されます。
```

---

#### C-4: X-3 top_k 統合テーブル

**コード根拠:** `settings.yaml` の ui.top_k=3, batch.top_k=4, evaluation.top_k=130

**操作:** L249（重み調整セクションの後、セパレータの前）に以下を挿入:

```markdown
### top_k 設定一覧

| 用途 | 設定場所 | 値 | 説明 |
|------|---------|-----|------|
| 回答支援AI（UI） | `ui.top_k` | 3 | 画面表示用 |
| 回答支援AI（バッチ） | `batch.top_k` | 4 | Excel出力用 |
| 改定影響調査（評価） | `evaluation.top_k` | 130 | 網羅性重視（`filter_mode: top_k` 時） |
```

---

#### C-5: X-2 GOOGLE_APPLICATION_CREDENTIALS

**操作:** L77（環境変数テーブルの後）に以下を追加:

```markdown
> **Note:** `GOOGLE_APPLICATION_CREDENTIALS` は Google Cloud SDK の標準環境変数で、認証情報JSONファイルの絶対パスを指定します。`.env` ではなく OS 環境変数として設定してください（`config.py` では直接参照しません）。
```

---

#### C-6: M-01 keyword_weight 重複説明の削除

**操作:** L171 のコメント部分を短縮:

変更前: `vector_weight=0.9,          # ベクトル検索の重み（keyword_weight は 1.0 - vector_weight で自動計算）`
変更後: `vector_weight=0.9,          # ベクトル検索の重み`

---

#### C-7: M-02 未実装ログファイルの記載

**実態:** `logs/` には `app.log` と `archive/` のみ存在。`error.log`, `access.log` は存在しない。

**操作:** L427-434 を以下に置換:

```markdown
### ログファイル

```
logs/
├── app.log          # メインログ（全レベル統合出力）
└── archive/         # アーカイブ
```
```

---

#### C-8: M-06 環境変数の条件付き必須の分類

**操作:** 環境変数テーブル（L22-77）のヘッダーに「必須度」列を追加し、「常に必須」「プロバイダー別必須」「オプション」を明記。現テーブルの Note を整理して3階層に再構成。

---

## Batch 2: 3並列

### Agent D: ARCHITECTURE.md（11件）

#### D-1: I-1 src/types/ ディレクトリ

**コード根拠:** `src/types/search_types.py` に TypedDict 6個、Dataclass 1個、定数クラス 4個。

**操作:**
1. レイヤー図（L29）に `types/` 行を追加
2. API リファレンスセクション（L650 付近）に「型定義（Types Layer）」セクションを新設

挿入テキスト（セクション用）:

```markdown
## 型定義（Types Layer）

**モジュール:** `src/types/search_types.py`

| 分類 | クラス名 | 用途 |
|------|---------|------|
| TypedDict | `SearchResultDict` | 検索結果の基本型 |
| TypedDict | `MultiStageSearchResultDict` | 多段階検索結果（Search_Category付き） |
| TypedDict | `VectorSearchResultDict` | ベクトルDB検索結果 |
| Dataclass | `ParsedCombinedText` | 結合テキスト解析結果（frozen=True） |
| 定数 | `SearchResultKeys` | Excel出力列名の一元管理 |
| 定数 | `MetadataKeys` | ChromaDBメタデータキー定数 |
| 定数 | `SourceValues` | ソース値定数（SCENARIO, HISTORY_DATA） |
| 定数 | `SearchCategoryValues` | カテゴリ値定数（BOTH, ORIGINAL_ONLY, LLM_ENHANCED_ONLY） |
```

---

#### D-2: I-2 InputHandler の4サブクラス

**コード根拠:** `input_handler.py` L103-547 に4サブクラス:
1. ExcelInputHandler (L103-206): FAQ履歴データ
2. HierarchicalExcelInputHandler (L208-379): マージ版シナリオ（階層構造）
3. MultiFolderInputHandler (L381-459): シナリオ+FAQ統合
4. TextInputHandler (L464-547): 改定内容入力（多段階検索用）

**操作:** L209-228（Handler Layer セクション内）に以下のテーブルを追加:

```markdown
**サブクラス一覧:**

| サブクラス | 用途 | 入力形式 |
|-----------|------|---------|
| `ExcelInputHandler` | FAQ履歴データ読み込み（回答支援AI） | 標準Excel（番号, 質問, 回答） |
| `HierarchicalExcelInputHandler` | マージ版シナリオ読み込み | Excel複数シート（階層+質問+回答） |
| `MultiFolderInputHandler` | シナリオ+FAQ統合（回答支援AI） | 複数フォルダ |
| `TextInputHandler` | 改定内容入力（改定影響調査） | Excel + 正解ID対応 |
```

---

#### D-3: I-4 config/ ディレクトリ

**操作:** レイヤー図（Entry Points と Core の間）に Configuration Layer を追加:

```markdown
│                   Configuration Layer                      │
│  config/ - 設定ファイル                                    │
│    ├─ settings.yaml（common/ui/batch/evaluation）          │
│    └─ business_areas.yaml（業務分野マッピング）              │
│  config.py - SearchConfig データクラス                      │
```

---

#### D-4: I-5 MetadataVectorDB の LRUCache

**コード根拠:** `vector_db.py:16-58` に LRUCache クラス（OrderedDict + threading.Lock）。`MetadataVectorDB._client_cache = LRUCache(max_size=10)` (L58)。

**操作:** L267-280（MetadataVectorDB セクション）に以下を追加:

```markdown
**パフォーマンス最適化:**
- `LRUCache(max_size=10)`: ChromaDB クライアントを DB パス別にキャッシュ
- スレッドセーフ（`threading.Lock`）、最大10エントリで古いものを自動削除
```

---

#### D-5: I-6 SearchStrategy の execute() 統一文書化

**コード根拠:** `search_strategy.py` に抽象基底クラス + 4戦略。全て同一シグネチャ:
```python
def execute(self, input_number: str, query_text: str, original_answer: str) -> List[Dict[str, Any]]
```

**操作:** API リファレンスの検索エンジンセクション（L946 付近）に以下を追加:

```markdown
#### SearchStrategy - 戦略パターン

統一インターフェース: `execute(input_number, query_text, original_answer) -> List[Dict]`

| 戦略クラス | search_mode | 処理 |
|-----------|-------------|------|
| `OriginalSearchStrategy` | original | 原文でベクトル+キーワード検索 |
| `LLMEnhancedSearchStrategy` | llm_enhanced | LLMクエリ生成後にベクトル+キーワード検索 |
| `MultiStageSearchStrategy` | multi_stage | 原文+LLMクエリの両検索→OR結合・3分類 |
| `KeywordFilterSearchStrategy` | keyword_filter | キーワードマッチのみ（ベクトル検索なし） |
```

---

#### D-6: I-7 InputHandler の settings.yaml カラム動的解決

**コード根拠:** `input_handler.py:140-158` で `self.config.QUERY_COLUMN_CANDIDATES` から候補を順にマッチ。`config.py` から動的読み込み。

**操作:** D-2 の InputHandler セクション内に以下を追加:

```markdown
**列名の動的解決:** settings.yaml の `columns` セクションに候補列を列挙し、Excelに存在する最初の列を採用。query/answer は必須（ValueError）、tag は任意（警告続行）。
```

---

#### D-7: I-3 (ARCH) run_eval.py の max_workers

**コード根拠:**
- `processor.py`: `ThreadPoolExecutor(max_workers=10)` - LLM判断支援の並列評価
- `ops_ui.py`: `ThreadPoolExecutor(max_workers=2)` - Azure/VertexAI並列検索
- `run_eval.py`: ThreadPoolExecutor で複数エリア並列検索

**操作:** L595-607（パフォーマンス最適化セクション）を更新:

```markdown
### 並列処理

| 箇所 | max_workers | 用途 |
|------|-------------|------|
| `processor.py` | 10 | LLM判断支援の並列評価 |
| `ops_ui.py` | 2 | Azure/VertexAI プロバイダー並列検索 |
| `run_eval.py` | 5 | 複数エリアのDB検索並列実行 |
```

---

#### D-8: M-1 レイヤー構造とAPIリファレンスの二重記載

**操作:** レイヤー構造セクション冒頭（L99-100）に以下を追記し、詳細をAPIリファレンスに委譲:

```markdown
> **注:** 各モジュールの詳細 API 仕様は [API リファレンス](#api-リファレンス) を参照してください。
```

レイヤー構造内のコード例を簡潔化（シグネチャのみ残し、説明をAPIリファレンスへのリンクに置換）。

---

#### D-9: M-2 (ARCH) scripts/ の説明

**コード根拠:** 5スクリプト: build_db.py, check_db_content.py, generate_correct_ids.py, prepare_before_scenario.py, create_handover_package.py

**操作:** L654 付近に以下のテーブルを追加:

```markdown
### scripts/ ディレクトリ

| スクリプト | 用途 | 主要オプション |
|-----------|------|---------------|
| `build_db.py` | ベクトルDB構築 | `--force`, `--business {name}`, `--revisions-only`, `--no-revisions` |
| `check_db_content.py` | DB内容確認 | — |
| `generate_correct_ids.py` | 正解ID対応表生成 | — |
| `prepare_before_scenario.py` | 改定前データ前処理 | — |
| `create_handover_package.py` | 引き継ぎパッケージ作成 | `--include-vectordb`, `--include-examples` |
```

---

#### D-10: M-3 (ARCH) テストカバレッジ情報

**実態:** `tests/unit/` に 11ファイル、40+ テストケース。

**操作:** L630-649（テストセクション）に概要テーブルを追加:

```markdown
### テスト概要

| 項目 | 値 |
|------|-----|
| テストフレームワーク | pytest |
| テストファイル数 | 11 |
| カバレッジ対象 | コアロジック + ハンドラー + ユーティリティ |

実行: `pytest tests/unit/ -v`
カバレッジ: `pytest tests/unit/ --cov=src --cov-report=term-missing`
```

---

#### D-11: M-5 tenacity 依存の欠落

**コード根拠:** 5ファイルで使用: judgment_support.py, query_enhancer.py, searcher.py, azure_embedding.py, gemini_embedding.py

**操作:** L408-423（依存関係テーブル）に以下を追加:

```markdown
| **リトライ制御** | tenacity | API呼び出しの自動リトライ・指数バックオフ（5モジュールで使用） |
```

---

### Agent E: README.md（3件）

#### E-1: I-08 ディレクトリツリーに .streamlit/config.toml が欠落

**実態:** `rag-local/.streamlit/config.toml` が存在（[server], [logger] セクション）。

**操作:** L197（`.env.example` の直後）に以下を挿入:

```
├── .streamlit/
│   └── config.toml              # Streamlit 設定
```

---

#### E-2: I-05 requirements-dev.txt の依存関係説明

**実態:** requirements-dev.txt に pytest, pytest-cov, pytest-mock, pytest-asyncio, faker。

**操作:** L171（`pip install -r requirements-dev.txt` の直後）に以下を挿入:

```markdown
**開発用パッケージ** (`requirements-dev.txt`):
- `pytest` — テストフレームワーク
- `pytest-cov` — カバレッジ計測
- `pytest-mock` — モックライブラリ
- `pytest-asyncio` — 非同期テスト対応
- `faker` — テストデータ生成
```

---

#### E-3: M-03 CLAUDE.md が引き継ぎ対象外の理由

**コード根拠:** `create_handover_package.py:24` で許可リスト方式。CLAUDE.md は INCLUDE リストに含まれない。

**操作:** L304（CLAUDE.md のテーブル行）の後に以下の Note を追加:

```markdown
> **注記:** 引き継ぎパッケージは許可リスト方式で生成されます（`create_handover_package.py:INCLUDE`）。`CLAUDE.md` は開発者用のプロジェクトメモであり、運用には不要なため許可リストから除外されています。
```

---

### Agent F: TROUBLESHOOTING.md（5件）

#### F-1: I-T01 ChromaDB 再構築の用途別区別

**コード根拠:** `build_db.py:42-50` に `--revisions-only` と `--no-revisions` が相互排他的引数として実装。

**操作:** L122 の後に以下を挿入:

```markdown
### 用途別の再構築コマンド

| 用途 | コマンド | 対象DB |
|------|---------|--------|
| 回答支援AI用のみ | `python scripts/build_db.py --no-revisions` | naibujimu, smile 等 |
| 改定影響調査用のみ | `python scripts/build_db.py --revisions-only` | rev01_smile, rev02_souzoku 等 |
| 両方 | `python scripts/build_db.py` | 全DB |
```

---

#### F-2: I-T03 Streamlit 再起動の詳細

**操作:** L287（Streamlit キャッシュの項目の後）に以下を追加:

```markdown
  Streamlit は `@st.cache_resource` でDBクライアントやLLMをキャッシュしているため、DB再構築後は必ず再起動が必要です（`session_state` のキャッシュ値も古いまま残ります）。
```

---

#### F-3: I-T04 ENABLE_LLM_ANALYSIS のトラブルシューティング

**コード根拠:**
- `run_eval.py:110-121`: `enable_llm_analysis=True` で JudgmentSupport を初期化
- `judgment_support.py:22-27`: `multi_stage_enable_judgment_support` で LLM 初期化を制御
- `config.py:263-269`: `llm_provider` 未設定時に ValueError

**操作:** L383（API関連セクションの後）に以下を挿入:

```markdown
### LLM分析が失敗する

**症状:**
```
ValueError: DEFAULT_LLM_PROVIDER環境変数が設定されていません
```

**原因:** `run_eval.py` はデフォルトで `enable_llm_analysis=True`（JudgmentSupport による関連性判定）。LLM 環境変数が未設定だと初期化エラー。

**解決策:**
1. LLM分析を無効化: `ENABLE_LLM_ANALYSIS=false python apps/revision-ops/run_eval.py`
2. LLM環境変数を設定: `DEFAULT_LLM_PROVIDER=gemini`, `DEFAULT_LLM_MODEL=gemini-2.5-flash-lite`, `GEMINI_PROJECT_ID`
```

---

#### F-4: M-T01 Windows/Linux コマンド混在

**実態:** L35 に `ls -l C:\VSCode\rag\...`（Linux構文でWindowsパス）が存在。他の箇所は適切に Windows/Linux を分けている。

**操作:** L32-36 を以下に修正:

```bash
# ファイルが存在するか確認
# Windows
dir rag-local\gemini_credentials.json

# Linux/Mac
ls -l rag-local/gemini_credentials.json
```

---

#### F-5: M-4 (ARCH) VALID_EMBEDDING_PROVIDERS の明記

**コード根拠:** `config.py:98` に `VALID_EMBEDDING_PROVIDERS = ("vertex_ai", "azure_openai")`

**操作:** ARCHITECTURE.md の埋め込みモデルセクション（L297 付近）に以下を追加:

```markdown
**対応プロバイダー:** `config.py` の `VALID_EMBEDDING_PROVIDERS = ("vertex_ai", "azure_openai")`
```

---

## 実行計画

### Step 1: Batch 1（3並列）
- Agent A: ANSWER_SUPPORT.md（5件: A-1〜A-5）
- Agent B: REVISION_OPS.md（8件: B-1〜B-7 + M-R02はB-3に統合）
- Agent C: CONFIGURATION.md（8件: C-1〜C-8）

### Step 2: Batch 2（3並列）
- Agent D: ARCHITECTURE.md（11件: D-1〜D-11）
- Agent E: README.md（3件: E-1〜E-3）
- Agent F: TROUBLESHOOTING.md（5件: F-1〜F-5、うちF-5はARCHITECTURE.mdの修正）

### Step 3: 検証
- git diff で全変更を確認
- Markdown 構文チェック
- コミット
