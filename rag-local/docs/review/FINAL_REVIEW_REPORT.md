# rag-local ドキュメント総合レビュー 最終レポート

**レビュー日**: 2026-03-03
**目的**: 引き継ぎ（3/3, 3/10）に向けた全ドキュメント品質検証
**基準**: 引き継ぎ相手がドキュメントだけで理解・運用できるか

---

## Phase 1 サマリ（個別レビュー結果）

| ドキュメント | Critical | Important | Minor | レポート |
|-------------|:---:|:---:|:---:|---|
| CONFIGURATION.md + README.md | 6 | 8 | 6 | [config-readme-review.md](./config-readme-review.md) |
| ANSWER_SUPPORT.md | 2 | 5 | 4 | [answer-support-review.md](./answer-support-review.md) |
| REVISION_OPS.md + TROUBLESHOOTING.md | 6 | 11 | 6 | [revision-ops-troubleshooting-review.md](./revision-ops-troubleshooting-review.md) |
| ARCHITECTURE.md | 6 | 7 | 5 | [architecture-review.md](./architecture-review.md) |
| **合計** | **20** | **31** | **21** | **72件** |

---

## Phase 2: ドキュメント間整合性チェック

### X-1: 検索モードの説明が不統一 [Important]

| ドキュメント | 検索モードの記載 |
|---|---|
| README.md | 言及なし（詳細ドキュメントに委譲） |
| ANSWER_SUPPORT.md | `original`, `llm_enhanced`（`multi_stage` は改定調査専用と注記） |
| CONFIGURATION.md | `original`, `llm_enhanced`, `multi_stage` の3モード |
| ARCHITECTURE.md | 4戦略: Original, LLMEnhanced, MultiStage, KeywordFilter |

**問題点**:
- CONFIGURATION.md は `search_type` (hybrid / keyword_filter) の記載が **完全欠落**（Agent 1 C-04）
- ARCHITECTURE.md の `KeywordFilterSearchStrategy` は `search_type=keyword_filter` に対応するが、他ドキュメントでは `search_type` 自体が説明されていない
- ANSWER_SUPPORT.md は `search_type` に触れずに UI で `hybrid` 固定と暗黙的に扱っている
- **`search_mode`（original/llm_enhanced/multi_stage）と `search_type`（hybrid/keyword_filter）が2つの独立した設定軸であること**が、どのドキュメントでも体系的に説明されていない

**修正提案**: CONFIGURATION.md に「検索設定の2軸」セクションを追加し、`search_mode` × `search_type` の組み合わせマトリクスを記載する。

---

### X-2: 環境変数の網羅性チェック [Important]

各ドキュメントで言及される環境変数が CONFIGURATION.md に漏れなく記載されているかを検証。

| 環境変数 | CONFIGURATION.md | README.md | ANSWER_SUPPORT.md | REVISION_OPS.md | TROUBLESHOOTING.md |
|---|:---:|:---:|:---:|:---:|:---:|
| `DEFAULT_LLM_PROVIDER` | ✅ | ✅ | ✅ | - | ✅ |
| `DEFAULT_LLM_MODEL` | ✅ | ✅ | ✅ | - | ✅ |
| `DEFAULT_EMBEDDING_PROVIDER` | ✅ | ✅ | ✅ | - | ✅ |
| `AZURE_OPENAI_API_KEY` | ✅ | ✅ | - | - | ✅ |
| `AZURE_OPENAI_ENDPOINT` | ✅ | ✅ | - | - | ✅ |
| `GEMINI_PROJECT_ID` | ✅ | ✅ | - | - | ✅ |
| `ENABLE_LLM_ANALYSIS` | ✅ | - | - | ✅ | - |
| `CREDENTIAL_SOURCE` | ✅ | - | - | - | ✅ |
| `LOG_LEVEL` | ✅ | - | - | - | ✅ |
| `GOOGLE_APPLICATION_CREDENTIALS` | - | ✅ | - | - | - |

**問題点**:
- `GOOGLE_APPLICATION_CREDENTIALS` は README.md の Step 2a で言及されるが、CONFIGURATION.md の環境変数一覧に記載なし。ただしこれは OS レベルの設定であり `config.py` では使用しないため、記載しないことは合理的。README.md 側に「OS環境変数であり `.env` には設定しない」旨の注記があると明確。
- `ENABLE_LLM_ANALYSIS` の説明が CONFIGURATION.md (L71) と REVISION_OPS.md で異なる精度 — CONFIGURATION.md は簡潔すぎ、REVISION_OPS.md は JudgmentSupport のみ制御する点が不明確（Agent 3 C-R04）。

---

### X-3: デフォルト値の一貫性チェック [Important]

| パラメータ | ANSWER_SUPPORT.md | REVISION_OPS.md | CONFIGURATION.md | ARCHITECTURE.md | config.py 実態 |
|---|---|---|---|---|---|
| `vector_weight` | 0.9 | 0.9 | 0.9 | 0.9 | settings.yaml から動的ロード（batch.vector_weight=0.9） |
| `top_k`（バッチ） | 4 | - | 4 | - | settings.yaml batch.top_k=4 |
| `top_k`（UI） | - | - | - | - | settings.yaml ui.top_k=3 |
| `top_k`（eval） | - | 130 | - | - | settings.yaml evaluation.*.top_k |
| `FILTER_MODE` | - | top_k | - | - | multi_stage_orchestrator デフォルト=threshold, settings.yaml=top_k |
| `MAX_RESULTS` | - | 100 | - | - | settings.yaml evaluation |
| `EMBEDDING_BATCH_SIZE` | - | - | 250（暗黙） | 250 | config.py EMBEDDING_BATCH_SIZE=250 |

**問題点**:
- **`vector_weight=0.9` は全ドキュメントで一貫** ✅
- **`top_k` の文脈依存** — バッチ=4, UI=3, eval=130 だが、この3つの値が1箇所にまとまっていない。CONFIGURATION.md がリファレンスとしてすべて記載すべき
- **`FILTER_MODE` のデフォルト矛盾** — コードのデフォルトは `threshold` だが、settings.yaml から読み込まれる値は `top_k`。Agent 3 M-R01 で指摘済み。CONFIGURATION.md に「コード上のデフォルト vs settings.yaml の設定値」の区別がない
- **デフォルト値の動的ロードが未説明** — Agent 1 C-06 で指摘。CONFIGURATION.md のコード例はハードコードされた値に見えるが、実際は settings.yaml から読み込まれる

---

### X-4: ファイルパスの一貫性チェック [Minor]

| パス | README.md | ANSWER_SUPPORT.md | REVISION_OPS.md | ARCHITECTURE.md | 実態 |
|---|---|---|---|---|---|
| `data/vector_db/` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `data/source/scenarios/latest/` | ✅ | ✅ | - | - | ✅ |
| `data/source/scenarios/revisions/` | ✅ | - | ✅ | - | ✅ |
| `data/input/` | ✅ | ✅ | ✅ | - | ✅ |
| `data/output/latest/` | ✅ | ✅ | ✅ | - | ✅ |
| `reference/変更前シナリオ/` | - | - | ✅ | - | **存在しない** |
| `reference/vector_db/` | - | - | - | - | **run_eval.py docstringのみ（誤記）** |

**問題点**:
- `reference/変更前シナリオ/` は REVISION_OPS.md で参照されるが存在しない（Agent 3 I-R04）
- `reference/vector_db/` は run_eval.py のdocstringに記載があるが実在しない（Agent 3 C-R03）

---

### X-5: Excel 出力列構成の不一致 [Critical]

3つのドキュメントが出力 Excel の列を説明しているが、すべて実装と不一致：

| ドキュメント | 不一致内容 |
|---|---|
| ANSWER_SUPPORT.md | `Generated_Tags` が廃止済みだが記載。`Scenario_ID`/`Sheet_Name`/`Row_Index` 未記載（Agent 2 C-2） |
| REVISION_OPS.md（サマリーシート） | 列名・列数が実装と大幅乖離。「エリア」「正解発見率」「必要確認件数」「未発見数」「未発見ID」が未記載（Agent 3 C-R01） |
| REVISION_OPS.md（詳細シート） | 「修正案」列は**実装に存在しない架空列**。「検出フラグ」「検索タイプ」「マッチ種別」「未発見セクション」が未記載（Agent 3 C-R02） |

**これは引き継ぎにおいて最大のリスク。** 引き継ぎ者が出力 Excel を見てドキュメントと照合したとき、列構成が一致しないため混乱が確実に発生する。

---

### X-6: 読み順検証（README推奨: 1→6）[Minor]

README.md が推奨する読み順:
1. README.md → 2. ANSWER_SUPPORT.md → 3. REVISION_OPS.md → 4. CONFIGURATION.md → 5. ARCHITECTURE.md → 6. TROUBLESHOOTING.md

**検証結果**:
- ✅ README.md は前提知識なしで読める（セットアップ手順が自己完結）
- ✅ ANSWER_SUPPORT.md は README.md のセットアップ完了を前提としており、順序は適切
- ✅ REVISION_OPS.md は ANSWER_SUPPORT.md の知識を前提としない（独立して読める）
- ⚠️ CONFIGURATION.md を4番目に持ってくるのは遅い — セットアップ時に `settings.yaml` の調整が必要な場合がある。ただし README.md Step 3 で最低限の環境変数は設定するため、実用上は問題なし
- ✅ ARCHITECTURE.md は1-4の知識を前提に技術詳細を提供しており、順序は適切
- ✅ TROUBLESHOOTING.md は最後に配置されており、問題発生時のリファレンスとして適切

**総合判定**: 読み順は概ね妥当。CONFIGURATION.md は2番目（ANSWER_SUPPORT.md の前）に置く選択肢もあるが、現在の順序でも実用上の問題はない。

---

## 全指摘事項の統合優先度分類

### Critical（事実誤認 — 即時修正が必要）: 20件

| # | 出典 | 内容 |
|---|---|---|
| 1 | Agent 3 C-R01 | REVISION_OPS: サマリーシートの列構成が実装と大幅乖離 |
| 2 | Agent 3 C-R02 | REVISION_OPS: 詳細シートに架空の「修正案」列、実在の列が多数未記載 |
| 3 | Agent 2 C-2 | ANSWER_SUPPORT: バッチ出力列に廃止済み `Generated_Tags`、未記載列3つ |
| 4 | Agent 1 C-04 | CONFIGURATION: `search_type` (hybrid/keyword_filter) の設定が完全欠落 |
| 5 | Agent 4 C-1 | ARCHITECTURE: JudgmentSupport の `analyze_relevance` メソッドは存在しない（実際は `evaluate`） |
| 6 | Agent 4 C-5 | ARCHITECTURE: レイヤー図の `utils.py` は存在しないファイル |
| 7 | Agent 1 C-06 | CONFIGURATION: デフォルト値が settings.yaml から動的ロードされることが未説明 |
| 8 | Agent 1 C-01 | CONFIGURATION: settings.yaml 欠損時の `RuntimeError` 発生条件が不正確 |
| 9 | Agent 1 C-02 | CONFIGURATION: LLMプロバイダーが `gemini` のみという制約が欠落 |
| 10 | Agent 1 C-03 | CONFIGURATION: `VECTOR_DB_BATCH_SIZE` 等のバリデーション範囲が未記載 |
| 11 | Agent 1 C-05 | README: `create_handover_package.py` の `--include-examples` 動作説明が曖昧 |
| 12 | Agent 3 C-R03 | REVISION_OPS: run_eval.py docstring 内のパス `reference/vector_db/` が誤り |
| 13 | Agent 3 C-R04 | REVISION_OPS: `ENABLE_LLM_ANALYSIS` が JudgmentSupport のみ制御する点が不明確 |
| 14 | Agent 3 C-T01 | TROUBLESHOOTING: Azure認証エラーメッセージが openai v0.x の古い形式 |
| 15 | Agent 3 C-T02 | TROUBLESHOOTING: `search_mode: original` の解決策が改定影響調査に無効 |
| 16 | Agent 4 C-2 | ARCHITECTURE: OutputHandler の `app_prefix` 引数が欠落 |
| 17 | Agent 4 C-3 | ARCHITECTURE: OutputHandlerFactory.create() の `app_prefix` 引数が欠落 |
| 18 | Agent 4 C-4 | ARCHITECTURE: DynamicDBManager の `self.translator` は `self._translator`（private） |
| 19 | Agent 4 C-6 | ARCHITECTURE: Utils Layer の `business_area_translator.py` のサブカテゴリ分類が不正確 |
| 20 | Agent 2 C-3 | ANSWER_SUPPORT: Metadata シートのパラメータリストに `keyword_weight` 欠落 |

### Important（情報不足 — 引き継ぎ前に修正推奨）: 35件

| # | 出典 | 内容 |
|---|---|---|
| 1 | Cross X-1 | **検索設定の2軸（search_mode × search_type）**が体系的に説明されていない |
| 2 | Cross X-3 | **top_k の文脈依存値**（batch=4, ui=3, eval=130）が1箇所にまとまっていない |
| 3 | Cross X-5 | **Excel出力列構成**が3ドキュメントすべてで実装と不一致（最大のリスク） |
| 4 | Agent 2 I-1 | ANSWER_SUPPORT: 入力Excelのカラム仕様（位置ベース検出、命名規則）が完全未記載 |
| 5 | Agent 2 I-5 | ANSWER_SUPPORT: バッチ実行時のDB自動更新動作が未記載 |
| 6 | Agent 2 I-4 | ANSWER_SUPPORT: UIで検索タイプが `hybrid` 固定であることが未記載 |
| 7 | Agent 2 I-3 | ANSWER_SUPPORT: UIセッション内設定変更がYAMLに保存されない旨が未記載 |
| 8 | Agent 2 I-2 | ANSWER_SUPPORT: `keyword_weight` プロパティの説明不足 |
| 9 | Agent 1 I-02 | CONFIGURATION: `SearchConfig` の未記載フィールドが多数 |
| 10 | Agent 1 I-04 | README: テスト実行方法（pytest, requirements-dev.txt）が未記載 |
| 11 | Agent 1 I-06 | CONFIGURATION: `evaluation` セクションの設定詳細が不足 |
| 12 | Agent 1 I-08 | README: ディレクトリツリーに `.streamlit/config.toml` が欠落 |
| 13 | Agent 1 I-03 | README: Python バージョン要件が曖昧（3.9 vs 最新 chromadb 要件） |
| 14 | Agent 1 I-05 | README: `requirements-dev.txt` の依存関係説明が欠如 |
| 15 | Agent 1 I-07 | CONFIGURATION: `business_areas.yaml` の `smile_tablet` マッピングの用途不明 |
| 16 | Agent 3 I-R01 | REVISION_OPS: `business_areas.yaml` と `settings.yaml` の二重登録の必要性が説明不足 |
| 17 | Agent 3 I-R02 | REVISION_OPS: `keyword_filter` で多段階検索(Stage 1/2/3)が適用されないことが未記載 |
| 18 | Agent 3 I-R04 | REVISION_OPS: `reference/変更前シナリオ/` がドキュメントに記載されるが実在しない |
| 19 | Agent 3 I-R05 | REVISION_OPS: `rev07_積立定期預金_未評価.md` が文書化されていない |
| 20 | Agent 3 I-R06 | REVISION_OPS: 入力ファイルの「変更内容」列が文書化されていない |
| 21 | Agent 3 I-R07 | REVISION_OPS: サマリーシート「改定内容（先頭50文字）」が実装では全文出力 |
| 22 | Agent 3 I-R03 | REVISION_OPS: `keyword_filter` 時の `vector_weight` 不要が暗黙的 |
| 23 | Agent 3 I-T01 | TROUBLESHOOTING: ChromaDB 再構築手順で回答支援AI用と改定DB用の区別なし |
| 24 | Agent 3 I-T02 | TROUBLESHOOTING: `--limit`/`--business` オプションの実在確認が必要 |
| 25 | Agent 3 I-T03 | TROUBLESHOOTING: 検索結果0件の原因にStreamlit再起動の件が不十分 |
| 26 | Agent 3 I-T04 | TROUBLESHOOTING: `ENABLE_LLM_ANALYSIS` のトラブルシューティングがない |
| 27 | Agent 4 I-1 | ARCHITECTURE: `src/types/` ディレクトリが文書化されていない |
| 28 | Agent 4 I-2 | ARCHITECTURE: InputHandler の4サブクラスが文書化されていない |
| 29 | Agent 4 I-4 | ARCHITECTURE: `config/` ディレクトリがレイヤー図に未記載 |
| 30 | Agent 4 I-5 | ARCHITECTURE: MetadataVectorDB の LRUCache 機構が未文書化 |
| 31 | Agent 4 I-6 | ARCHITECTURE: SearchStrategy の各戦略の execute() が統一的に文書化されていない |
| 32 | Agent 4 I-7 | ARCHITECTURE: InputHandler の settings.yaml カラム動的解決が未記載 |
| 33 | Agent 4 I-3 | ARCHITECTURE: run_eval.py の max_workers 記載が不完全 |
| 34 | Cross X-2 | `GOOGLE_APPLICATION_CREDENTIALS` が CONFIGURATION.md に未記載（注記で可） |
| 35 | Agent 1 I-01 | CONFIGURATION: `search_source` のUI動的切替（情報提供のみ） |

### Minor（文体・明瞭性・体裁）: 24件

| # | 出典 | 内容 |
|---|---|---|
| 1 | Cross X-4 | 2つの存在しないパス参照（reference/変更前シナリオ, reference/vector_db） |
| 2 | Cross X-6 | 読み順は概ね妥当（改善の余地はあるが実用上問題なし） |
| 3 | Agent 1 M-01 | CONFIGURATION: `keyword_weight` 自動計算の重複説明 |
| 4 | Agent 1 M-02 | CONFIGURATION: 未実装ログファイル（error.log, access.log）の記載 |
| 5 | Agent 1 M-03 | README: `CLAUDE.md` が引き継ぎ対象外の理由が不明 |
| 6 | Agent 1 M-04 | CONFIGURATION: 手動検証コマンドに dotenv の load が欠如 |
| 7 | Agent 1 M-05 | README: Step 参照不整合（Step 4 → Step 3） |
| 8 | Agent 1 M-06 | CONFIGURATION: 環境変数の条件付き必須の分類が曖昧 |
| 9 | Agent 2 M-1 | ANSWER_SUPPORT: キーワード Top-5 の正確性確認 |
| 10 | Agent 2 M-2 | ANSWER_SUPPORT: multi_stage Note が回答支援AI文脈で不要 |
| 11 | Agent 2 M-3 | ANSWER_SUPPORT: 処理フロー図のフォーマット不統一 |
| 12 | Agent 2 M-4 | ANSWER_SUPPORT: チャット履歴とバッチ出力の列差異 |
| 13 | Agent 3 M-R01 | REVISION_OPS: FILTER_MODE デフォルト値のコード/YAML区別 |
| 14 | Agent 3 M-R02 | REVISION_OPS: データ準備フローのパス表記混在 |
| 15 | Agent 3 M-R03 | REVISION_OPS: コマンド例の提示順序 |
| 16 | Agent 3 M-T01 | TROUBLESHOOTING: Windows/Linux コマンドの混在度 |
| 17 | Agent 3 M-T02 | TROUBLESHOOTING: ポート番号（問題なし） |
| 18 | Agent 3 M-T03 | TROUBLESHOOTING: Vertex AI SDK 移行の関数名確認 |
| 19 | Agent 4 M-1 | ARCHITECTURE: レイヤー構造とAPIリファレンスの二重記載 |
| 20 | Agent 4 M-2 | ARCHITECTURE: scripts/ の説明が曖昧 |
| 21 | Agent 4 M-3 | ARCHITECTURE: テストカバレッジ情報がない |
| 22 | Agent 4 M-4 | ARCHITECTURE: `VALID_EMBEDDING_PROVIDERS` の現在値が未明記 |
| 23 | Agent 4 M-5 | ARCHITECTURE: 外部ライブラリ依存に tenacity が欠落 |
| 24 | Agent 1 I-01 | CONFIGURATION: search_source UI動的切替の記述場所 |

---

## 引き継ぎ適性 総合評価

### ゼロからシステムを立ち上げられるか？

**結論: 条件付きで Yes。**

README.md の Step 1-6 は論理的な順序で記載されており、認証情報と環境変数が正しく設定されれば動作確認まで到達できる。ただし以下のギャップがある:

1. **テスト実行方法が未記載** — 引き継ぎ者がコードの正しさを検証する手段がない
2. **settings.yaml のデフォルト値ロードが未説明** — 設定変更時に「どこを変えればいいか」が不明確
3. **`search_type` (hybrid/keyword_filter) が CONFIGURATION.md から完全欠落** — 改定影響調査の一部で使用される重要な設定

### 各AIを運用できるか？

| AI | バッチ運用 | UI運用 | 評価 |
|---|---|---|---|
| 回答支援AI | ⚠️ 入力Excel仕様の欠落が障壁 | ✅ UIは直感的 | 入力仕様の追記で解決 |
| 改定影響調査AI | ❌ Excel出力の列構成が実装と大幅乖離 | ⚠️ UIは別途説明が必要 | 列構成の書き直しが必須 |

### 新しい改定を追加できるか？

REVISION_OPS.md の「新しい改定の追加手順」は網羅的だが、`business_areas.yaml` と `settings.yaml` の二重登録の背景説明が不足。手順通りに実行すれば動作するが、「なぜ2箇所に登録が必要か」を理解できないと応用が利かない。

---

## 修正優先度ロードマップ

### 最優先（引き継ぎ 3/3 までに修正）

1. **REVISION_OPS.md のExcel出力列構成を実装に合わせて全面書き直し**（C-R01, C-R02）
   - サマリーシート: 14列の正確な定義
   - 詳細シート: 共通列7 + 各プロバイダー10列 + 未発見セクション5列
   - 架空の「修正案」列を削除

2. **ANSWER_SUPPORT.md のバッチ出力列構成を修正**（C-2, C-3）
   - `Generated_Tags` 削除、`Scenario_ID`/`Sheet_Name`/`Row_Index` 追加
   - Metadata シートに `keyword_weight` 追加

3. **ANSWER_SUPPORT.md に入力Excel仕様を追記**（I-1）
   - 列は位置ベース検出（1列目=番号, 2列目=質問, 3列目=回答）
   - ファイル命名規則: `{業務分野名}_{YYYYMMDD}.xlsx`

### 高優先（引き継ぎ 3/10 までに修正）

4. **CONFIGURATION.md に `search_type` の説明を追加**（C-04, X-1）
5. **CONFIGURATION.md のデフォルト値動的ロードを説明**（C-06）
6. **ARCHITECTURE.md の C-1, C-5 を修正**（存在しないメソッド名・ファイル参照）
7. **TROUBLESHOOTING.md の C-T01, C-T02 を修正**（古いエラーメッセージ、不正確な解決策）
8. **README.md にテスト実行方法を追記**（I-04）
9. **REVISION_OPS.md の `ENABLE_LLM_ANALYSIS` 説明を明確化**（C-R04）

### 中優先（引き継ぎ後に改善）

10. ARCHITECTURE.md のAPIシグネチャ修正（C-2, C-3, C-4）
11. ARCHITECTURE.md の未文書化モジュール追加（I-1〜I-7）
12. CONFIGURATION.md のバリデーション範囲追記（C-03）
13. REVISION_OPS.md の `reference/変更前シナリオ/` パス修正（I-R04）
14. 残りの Important/Minor 指摘

---

## docs/plans/ 設計書の状態

| 設計書 | 状態 | 推奨 |
|---|---|---|
| code-simplification-design.md | 未実装（設計レビュー完了） | 維持（アクティブ計画） |
| code-simplification-plan.md | 未実装（14件の5Phase実装計画） | 維持（アクティブ計画） |
| terminal-log-redesign.md | 部分実装（logger.py ノイズ抑制済み、ダッシュボード進行中） | 維持（進行中と明記推奨） |

---

## 付録: レビュー体制

| エージェント | 担当 | 完了 |
|---|---|---|
| config-reviewer | CONFIGURATION.md + README.md | ✅ |
| answer-support-reviewer | ANSWER_SUPPORT.md | ✅ |
| revision-ops-reviewer | REVISION_OPS.md + TROUBLESHOOTING.md | ✅ |
| architecture-reviewer | ARCHITECTURE.md | ✅ |
| team-lead | Phase 2 統合レビュー + 最終レポート | ✅ |
