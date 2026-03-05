# REVISION_OPS.md + TROUBLESHOOTING.md レビューレポート

**レビュー日**: 2026-03-03
**対象ファイル**: `docs/REVISION_OPS.md` (550行), `docs/TROUBLESHOOTING.md` (620行)
**突き合わせコード**: `run_eval.py`, `multi_stage_orchestrator.py`, `judgment_support.py`, `build_db.py`, `ops_ui.py`, `config/settings.yaml`

---

## REVISION_OPS.md

### Critical（事実誤認）

#### C-R01: サマリーシートの列構成がコードと不一致

**ドキュメント (L217-227)**:
```
改定番号 / 改定内容（先頭50文字）/ 正解ID数 / Azure_候補数 / Azure_正解一致数 / VertexAI_候補数 / VertexAI_正解一致数
```

**実装 (`run_eval.py:885-900`)**:
```
改定番号 / エリア / 改定内容 / 正解数 / Azure_候補数 / Azure_正解発見数 / Azure_正解発見率 / Azure_必要確認件数 / VertexAI_候補数 / VertexAI_正解発見数 / VertexAI_正解発見率 / VertexAI_必要確認件数 / 未発見数 / 未発見ID
```

**差分**:
- 「エリア」列が欠落（改定③のように複数エリアがある場合はエリアごとに行が分かれる）
- 「改定内容（先頭50文字）」→ 実装は全文（先頭50文字トリミングは実装されていない）
- 「正解ID数」→ 実装は「正解数」
- 「Azure_正解一致数」→ 実装は「Azure_正解発見数」
- 「Azure_正解発見率」「Azure_必要確認件数」が文書化されていない
- 「VertexAI_正解発見率」「VertexAI_必要確認件数」が文書化されていない
- 「未発見数」「未発見ID」列が文書化されていない

#### C-R02: 詳細シートの列構成がコードと不一致

**ドキュメント (L229-267)**:
共通列 + Azure側10列 + VertexAI側10列（「Azure_修正案」「VertexAI_修正案」含む）

**実装 (`run_eval.py:1076-1092`)**:
```python
common_headers = ["検出フラグ", "改定内容", "正解ID一覧", "LLM強化クエリ", "抽出キーワード", "検索タイプ", "ベクトル重み"]
result_headers = ["順位", "シナリオID", "類似度", "マッチ種別", "正解フラグ", "質問", "回答", "関連性判定", "判定根拠", "ソースファイル"]
unfound_headers = ["未発見ID", "変更内容", "ソースファイル", "質問", "回答"]
```

**差分**:
- **「修正案」列は実装に存在しない** — `judgment_support.py` は `relevance_judgment` と `judgment_reason` の2フィールドのみ返す。「修正案」は架空の列
- **「Azure_ソース」→ 実装は「Azure_ソースファイル」** — 改定番号+ボット名+Lv1カテゴリから `revision_source_files` マッピングで特定した具体的なExcelファイル名
- **共通列「検出フラグ」「検索タイプ」が文書化されていない** — 「検出フラグ」はAzureとVertexAIのどちらかで正解なら TRUE を表示するOR判定列
- **「Azure_カテゴリ」→ 実装は「Azure_マッチ種別」**
- **未発見セクション（未発見ID, 変更内容, ソースファイル, 質問, 回答）が文書化されていない**
- ドキュメントの「Azure_関連性判定」値: 「関連あり / 要確認 / 関連なし」だが、実装では `judgment_support.py:_parse_response` が「関連性:」プレフィックスの値をそのまま返すため、プロンプトファイル次第（ドキュメントはプロンプトの仕様を記載すべき）

#### C-R03: `run_eval.py` のコメント内のパスが間違っている

**ドキュメント (L12-13 のコード docstring)**:
```
reference/vector_db/rev*/{azure_openai,vertex_ai}/ が構築済みであること
```

**実装**: 実際のDB構築先は `data/vector_db/rev*/` であり、`reference/vector_db/` ではない。これはドキュメントではなく `run_eval.py` のdocstring自体の問題だが、ドキュメントとコードの信頼性に影響する。

#### C-R04: LLM分析のトグルが不正確

**ドキュメント (L155-160, L198-203)**:
```
ENABLE_LLM_ANALYSIS=true   # LLM分析の有効化（オプション、デフォルト: false）
```

**実装 (`run_eval.py:1219`)**:
```python
enable_llm = os.getenv("ENABLE_LLM_ANALYSIS", "false").lower() == "true"
```

ここまでは正しいが、ドキュメントは `ENABLE_LLM_ANALYSIS` が **JudgmentSupport（関連性判定）** を制御することを明確にしていない。QueryEnhancer（Stage 2のLLMクエリ拡張）は常に有効であり、この環境変数では制御されない。「LLM分析」と「LLMクエリ拡張」を混同する読者が出る可能性がある。

### Important（情報不足）

#### I-R01: `business_areas.yaml` の `revision_mappings` セクションが参照されているが実在確認が必要

**ドキュメント (L472-479)**: 新しい改定の追加手順 Step 2 で `config/business_areas.yaml` の `revision_mappings` に追加するよう指示。

**問題**: `settings.yaml` の `evaluation.revision_areas` は確認済みだが、`business_areas.yaml` の `revision_mappings` は別の役割（DynamicDBManager がシナリオパスを解決するため）。この二重登録の必要性を明確に説明すべき。

#### I-R02: `keyword_filter` 検索タイプの説明が不十分

**ドキュメント (L492-494)**: `search_type` の選択基準として `hybrid` / `keyword_filter` を簡潔に説明。

**実装**: `keyword_filter` の場合、run_eval.py は `ChromaDBKeywordSearcher` を使い、ベクトルDBを介さずにChromaDBのメタデータベースでキーワードマッチを行う。Stage 1/2/3 の多段階検索フローは適用されない。この根本的な違いがドキュメントに記載されていない。

#### I-R03: 改定⑤⑥の `keyword_filter` 設定では `vector_weight` が設定不要であることが暗黙的

**ドキュメント (L185-189)**: settings.yaml の ⑤⑥ には `vector_weight` キーがない。

**実装**: `keyword_filter` の場合、ベクトル検索を行わないため `vector_weight` は不要。これは正しいが、「新しい改定の追加手順」(L488-489) で `vector_weight` が `keyword_filter` の場合は不要と書かれている一方、settings.yaml の実例との対応が分かりにくい。

#### I-R04: reference/ ディレクトリに `変更前シナリオ/` が文書化されているが実在しない

**ドキュメント (L396)**: データ準備の詳細フロー内で `reference/変更前シナリオ/` を参照。

**実態**: `reference/` 直下に `変更前シナリオ/` ディレクトリは存在しない。`reference/改定シナリオ/revXX_*/修正前/` に相当するものと思われるが、パスが不整合。

#### I-R05: `reference/改定内容/` に `rev07_積立定期預金_未評価.md` が存在するが文書化されていない

**ドキュメント (L105-113)**: 改定番号とDBの対応表は ①〜⑥ のみ。

**実態**: `reference/改定内容/rev07_積立定期預金_未評価.md` が存在する。settings.yaml の `revision_areas` にも⑦は登録されていないため整合はしているが、将来の改定として存在する参照データについて注記があるとよい。

#### I-R06: 入力ファイルの必要な列に「変更内容」が記載されていない

**ドキュメント (L139-143)**: 入力ファイルの必要列は `番号`, `改定内容`, `正解ID`。

**実装 (`run_eval.py:133-134`)**: `変更内容` 列がない場合は空文字で初期化。ドキュメントには `変更内容` 列の説明がない（未発見シナリオの詳細表示に使用される）。

#### I-R07: サマリーシートの「改定内容（先頭50文字）」が実装では全文出力

ドキュメント L220 では先頭50文字と記載されているが、実装 (`run_eval.py:888`) では `revision_content` を丸ごと書き込んでいる。

### Minor（文体・明瞭性）

#### M-R01: 設定値テーブルのデフォルト値表現

**ドキュメント (L60-66)**: 設定値テーブルの `FILTER_MODE` のデフォルト値が `top_k` と記載されている。

**実装 (`multi_stage_orchestrator.py:53`)**: コンストラクタのデフォルト値は `threshold`。ただし `run_eval.py` は `settings.yaml` から `top_k` を読み込むため、実運用では `top_k` が正しい。コード上のデフォルトとYAML設定のデフォルトの区別を明確にすべき。

#### M-R02: データ準備の詳細フローの図中パス表記が混在

**ドキュメント (L391)**: `reference/マージ版シナリオ/最新/マージ版シナリオ_XXX-bot.xlsx` と記載されているが、L337 のフォルダ構造では `マージ版シナリオ/` 配下に `改定前/` と `最新/` があるのみで、ファイル名の規則が示されていない。

#### M-R03: 「Step 5: 評価実行」で `--provider vertex` が先に提示されている

**ドキュメント (L520-526)**: Azure 未設定の場合に `--provider vertex` を先に記載しているのは実用的だが、デフォルトの `--provider both` を先に書くほうがドキュメントの一貫性がある。

---

## TROUBLESHOOTING.md

### Critical（事実誤認）

#### C-T01: Azure OpenAI 認証エラーのエラーメッセージが古い

**ドキュメント (L53-54)**:
```
openai.error.AuthenticationError: Incorrect API key provided
```

**実態**: `openai` パッケージの v1.x 以降では `openai.AuthenticationError` であり、`openai.error.AuthenticationError` は v0.x の古いAPI。

#### C-T02: LLM 未初期化エラーの解決策が不正確

**ドキュメント (L319-324)**:
```yaml
# config/settings.yaml
common:
  search_mode: original
```
で LLM が不要になると記載。

**実装**: `run_eval.py:114` では `create_llm(config)` が常に呼ばれる（`search_mode` に関係なく）。改定影響調査では QueryEnhancer に LLM が必須。`search_mode: original` に変更しても `run_eval.py` は LLM を初期化する。この解決策は回答支援AI (`main.py`) には有効だが、改定影響調査には無効。

### Important（情報不足）

#### I-T01: ChromaDB 再構築の手順で回答支援AI用と改定DB用の区別がない

**ドキュメント (L110-122)**: `rm -rf data/vector_db/` → `python apps/answer-support/main.py` とあるが、改定DBの再構築は `build_db.py --revisions-only` が必要。一律に `main.py` を実行してもrev*のDBは再構築されない。

#### I-T02: `--limit` オプションが `main.py` に存在するか未確認

**ドキュメント (L231-235)**:
```bash
python apps/answer-support/main.py --limit 50
python apps/answer-support/main.py --business naibujimu --limit 50
```

`main.py` に `--limit` / `--business` オプションが実装されているかを確認する必要がある。バッチ処理のオプションが正しくない場合、初見ユーザーがエラーに遭遇する。

#### I-T03: 検索結果が0件の原因に「Streamlit再起動後のChromaDBインスタンス不整合」を追加すべき

**ドキュメント (L277-288)**: 4つの原因が挙げられているが、DB再構築後にStreamlitを再起動しないとキャッシュ済みの古いChromaDBインスタンスが使われて0件になるケースは L167-169 にのみ記載。L287 に相互参照はあるが、直接的な説明がないため見落としやすい。

#### I-T04: `ENABLE_LLM_ANALYSIS` 環境変数のトラブルシューティングがない

改定影響調査で `ENABLE_LLM_ANALYSIS=true` 設定時にLLM関連エラーが発生した場合の対処が、LLMタイムアウト以外に記載されていない。例えば、`judgment_support.txt` プロンプトファイルの欠落エラーなど。

### Minor（文体・明瞭性）

#### M-T01: Windows/Linux/Mac のコマンド混在

**ドキュメント (L139-149)**: `taskkill /F /IM python.exe` (Windows) と `pkill -f python` (Linux/Mac) が並記されているのは実用的。

ただし FAQ セクション (L571-573) では `pkill -f python` のみ記載されており、Windows コマンドが省略されている。

#### M-T02: ポート番号の不一致の可能性

**ドキュメント (L444)**: ポート 8501 がデフォルトと記載。

**実態**: Streamlit のデフォルトポートは 8501 で正しい。ただし L459 で代替ポート例として `--server.port 8502` を示しているのは適切。

#### M-T03: Vertex AI SDK 移行セクションのファイルパス

**ドキュメント (L409)**: `src/utils/auth.py` の関数名 `initialize_vertex_ai()` → `create_genai_client()` への移行が記載。現在の実装を確認して、実際の関数名と一致しているか要検証。

---

## reference/ ディレクトリ整合性

| ドキュメント記載 | 実ファイルシステム | 判定 |
|---|---|---|
| `reference/改定内容/` (revXX_*.md) | 6ファイル + `rev07_積立定期預金_未評価.md` 存在 | OK（rev07 は文書化外） |
| `reference/改定シナリオ/rev01_スマイル機能変更/` | 存在: `差分.md`, `修正前/`, `修正後/`, `参考資料/` | OK |
| `reference/改定シナリオ/rev02_相続少額払い/` | 存在 | OK |
| `reference/改定シナリオ/rev03_保険証→資格確認証/` | 存在 | OK |
| `reference/改定シナリオ/rev04_0円新規開設可能/` | 存在 | OK |
| `reference/改定シナリオ/rev05_AMLフィルター→GPLEX/` | 存在 | OK |
| `reference/改定シナリオ/rev06_DC→MDC/` | 存在 | OK |
| `reference/マージ版シナリオ/改定前/` | 存在 | OK |
| `reference/マージ版シナリオ/最新/` | 存在 | OK |
| `reference/問い合わせ履歴/` | 存在 | OK |
| `reference/シナリオボットメンテナンス管理台帳.xlsx` | 存在 | OK |
| `reference/変更前シナリオ/` (データ準備フロー内) | **存在しない** | NG (I-R04) |

---

## 引き継ぎ適性評価

### REVISION_OPS.md

**概要・目的**: 明確で分かりやすい。多段階ハイブリッド検索のコンセプトが図式化されており、初見でも理解可能。

**使用手順**: Step 1〜5 の流れは論理的で追いやすい。ただし:
- Excel出力の列構成が実装と大幅に乖離しており（C-R01, C-R02）、出力結果を見た初見者が混乱する
- 「修正案」列が存在しないため、出力にない列を探して時間を浪費する可能性がある
- `keyword_filter` 選択時にStage 1/2/3 が適用されないことが明示されていない（I-R02）

**新しい改定の追加手順**: 最も重要なセクション。手順は網羅的だが、`business_areas.yaml` と `settings.yaml` の二重登録の背景説明が必要（I-R01）。

### TROUBLESHOOTING.md

**網羅性**: 認証、DB、検索、API、UI、パフォーマンスと幅広くカバーしている。改定影響調査固有の問題は REVISION_OPS.md に委譲する設計も適切。

**実用性**: エラーメッセージの具体例と解決策の対が明確。ただし:
- Azure の古い `openai.error.AuthenticationError` は現バージョンと不一致（C-T01）
- LLM必須の解決策が回答支援AI向けで、改定影響調査には適用できない（C-T02）

**自己解決力**: 「デバッグ方法」「DB内容確認」「FAQ」セクションがあり、多くのケースで自力解決可能。

---

## サマリ

| 優先度 | REVISION_OPS.md | TROUBLESHOOTING.md | 合計 |
|--------|:---:|:---:|:---:|
| Critical | 4 | 2 | **6** |
| Important | 7 | 4 | **11** |
| Minor | 3 | 3 | **6** |

**最優先修正**: C-R01, C-R02（Excel出力の列構成を実装に合わせて書き直し）、C-T02（LLM未初期化の解決策を改定影響調査とその他で分離）
