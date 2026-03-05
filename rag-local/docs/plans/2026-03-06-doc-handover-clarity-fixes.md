# ドキュメント引き継ぎ品質修正 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 引き継ぎ先が理解できない記述（矛盾・廃止項目の残存・情報の誤配置・説明不足）を6ドキュメントから修正する

**Architecture:** ドキュメントのみの修正。コード変更なし。ただし processor.py L230 の vestigial フォールバック参照を1箇所クリーンアップする。

**Tech Stack:** Markdown

---

## 事前調査結果サマリ

修正の判断根拠として、コードベース調査で確定した事実:

| 項目 | 事実 | 根拠 |
|------|------|------|
| コールチェーン | `processor → searcher → search_strategy` | processor.py:64, searcher.py:217-236 |
| DEFAULT_EMBEDDING_MODEL | 廃止済み。processor.py:230 にフォールバック残存（vestigial） | config.py:236-245 で自動解決 |
| run_eval.py --provider | `both`/`azure`/`vertex` 選択可能 | run_eval.py:1183-1185 |
| ファイル名形式 | 日本語・英語どちらでも動作。正規表現 `^([^_]+)_(\d{8})\.xlsx$` | config.py:133 |
| LLM vs Embedding | LLM=Geminiのみ、Embedding=vertex_ai/azure_openai | config.py:264-269, 95-99 |
| デフォルト search_mode | `original` | config/settings.yaml:29 |
| keyword_weight | `1.0 - vector_weight` の自動計算プロパティ | config.py:301-304 |
| DynamicDBManager 差分更新 | ファイルのmtime vs JSON記録を比較、新しければスキップ | dynamic_db_manager.py:515-541 |
| マージ版シナリオ | 手動管理の参照Excel。自動生成スクリプトなし | reference/マージ版シナリオ/ |
| Vertex AI SDK | google-genai に完全移行済み。旧SDK参照ゼロ | requirements.txt, gemini_embedding.py |
| 日本語コレクション名 | BusinessAreaTranslator で自動変換。再発不可 | business_area_translator.py |
| MRL | CONFIGURATION.md に「対応」と記載だがコード上は3072固定。未実装 | gemini_embedding.py:55 |

---

### Task 1: TROUBLESHOOTING.md — 不要セクション2件を削除

**Files:**
- Modify: `docs/TROUBLESHOOTING.md:249-267` (コレクション名エラー)
- Modify: `docs/TROUBLESHOOTING.md:431-443` (Vertex AI SDK移行済み)

**Step 1: コレクション名エラーセクションを簡略化**

L249-267 の「コレクション名エラー」セクション全体を以下に置換:

```markdown
### コレクション名エラー

**症状:**
```
ValueError: Collection name contains invalid characters
```

**原因:** ChromaDB のコレクション名に使用できない文字が含まれている。

**解決策:** 業務分野名は `BusinessAreaTranslator`（`config/business_areas.yaml`）で自動的に英語変換されるため、通常は発生しない。発生した場合は `business_areas.yaml` に対象の日本語名→英語名マッピングが登録されているか確認する。
```

**Step 2: Vertex AI SDK移行セクションを削除**

L431-443 の「Vertex AI SDK 非推奨警告（移行済み）」セクションを完全削除。
理由: `google-genai` SDK に完全移行済みで旧SDK参照はコードベースにゼロ。トラブルシューティングとして不要。

**Step 3: 差分確認**

```bash
cd rag-local && git diff docs/TROUBLESHOOTING.md
```

**Step 4: コミット**

```bash
git add docs/TROUBLESHOOTING.md
git commit -m "docs: TROUBLESHOOTING.md — remove obsolete sections (SDK migration, collection name history)"
```

---

### Task 2: TROUBLESHOOTING.md — ファイル名形式矛盾修正 + LLM/Embedding区別明記

**Files:**
- Modify: `docs/TROUBLESHOOTING.md:318` (ファイル名例)
- Modify: `docs/TROUBLESHOOTING.md:622-628` (LLM FAQ)

**Step 1: ファイル名形式を修正**

L318 を以下に置換:

```markdown
2. ファイル名の確認: `{業務分野名}_{YYYYMMDD}.xlsx` 形式（正規表現: `^([^_]+)_(\d{8})\.xlsx$`）。`{業務分野名}` は `config/business_areas.yaml` に登録されている日本語名または英語名（例: `スマイル_20250301.xlsx` または `smile_20250301.xlsx`）。日本語名は英語DB名に自動変換される。
```

**Step 2: LLM/Embedding区別をFAQに明記**

L622-628 の LLM プロバイダー FAQ を以下に置換:

```markdown
### Q: LLM プロバイダーを変更するには?

現在 LLM は **Gemini のみ**サポートしています（クエリ拡張・関連性判定に使用）。`.env` でモデルを変更できます:
```env
DEFAULT_LLM_PROVIDER=gemini
DEFAULT_LLM_MODEL=gemini-2.5-flash-lite  # gemini-2.5-flash, gemini-2.5-pro も利用可能
```

> **Note:** LLM（文章生成）と埋め込みモデル（文章→数値変換）は別のサービスです。埋め込みモデルは Azure OpenAI / VertexAI の2プロバイダーに対応しています（`DEFAULT_EMBEDDING_PROVIDER` で設定）。
```

**Step 3: 差分確認 → コミット**

```bash
git diff docs/TROUBLESHOOTING.md
git add docs/TROUBLESHOOTING.md
git commit -m "docs: TROUBLESHOOTING.md — fix filename format example, clarify LLM vs embedding providers"
```

---

### Task 3: ARCHITECTURE.md — 依存ツリーの矛盾修正

**Files:**
- Modify: `docs/ARCHITECTURE.md:311-333` (依存関係ツリー)

**Step 1: 依存ツリーに searcher.py を追加**

実際のコールチェーン（`processor.py:64` → `searcher.py:217` → `search_strategy.py`）に合わせて修正。

L311-333 の依存ツリーを以下に置換:

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

ポイント: `processor.py` の直下に `searcher.py` を配置し、`search_strategy.py` や各エンジンは `searcher.py` の子として表現。データフロー図（L248）との整合性を確保。

**Step 2: 差分確認 → コミット**

```bash
git diff docs/ARCHITECTURE.md
git add docs/ARCHITECTURE.md
git commit -m "docs: ARCHITECTURE.md — fix dependency tree to match actual call chain (processor→searcher→strategy)"
```

---

### Task 4: ARCHITECTURE.md — SearchStrategy使い分け + DynamicDBManager説明追加

**Files:**
- Modify: `docs/ARCHITECTURE.md:139-148` (SearchStrategy表)
- Modify: `docs/ARCHITECTURE.md:199-205` (DB管理表)

**Step 1: SearchStrategy表にデフォルト・使い分け情報を追加**

L139-148 を以下に置換:

```markdown
#### SearchStrategy - 戦略パターン

統一インターフェース: `execute(input_number, query_text, original_answer) -> List[Dict]`

| 戦略クラス | search_mode | 処理 | 用途 |
|-----------|-------------|------|------|
| `OriginalSearchStrategy` | original（**デフォルト**） | 原文でベクトル+キーワード検索 | 回答支援AI。固有名詞の欠落なく最も安定 |
| `LLMEnhancedSearchStrategy` | llm_enhanced | LLMクエリ生成後にベクトル+キーワード検索 | 表現の揺れが大きい検索語の場合 |
| `MultiStageSearchStrategy` | multi_stage | 原文+LLMクエリの両検索→OR結合・3分類（運用保守効率化AI専用） | 改定影響調査。漏れを減らすために両方の結果を統合 |
| `KeywordFilterSearchStrategy` | keyword_filter | キーワードマッチのみ（ベクトル検索なし） | 用語の単純置換（AML→GPLEX等）の検出 |

デフォルト値は `config/settings.yaml` の `search_mode: original`。UI ではサイドバーで動的に切替可能。
```

**Step 2: DynamicDBManager の説明を具体化**

L203 を以下に置換:

```markdown
| `DynamicDBManager` | `src/utils/dynamic_db_manager.py` | 業務領域別ベクトルDB管理。参照ファイル（Excel）の更新日時を `data/vector_db/update_timestamps.json` に記録し、ファイルが更新されていなければDB再構築をスキップする（APIコスト削減） |
```

**Step 3: 差分確認 → コミット**

```bash
git diff docs/ARCHITECTURE.md
git add docs/ARCHITECTURE.md
git commit -m "docs: ARCHITECTURE.md — add search strategy defaults/guidance, clarify DynamicDBManager mechanism"
```

---

### Task 5: README.md — 認証条件を表形式に整理

**Files:**
- Modify: `rag-local/README.md:43-76` (Step 2 認証セクション)

**Step 1: L45 の散文を表に置換**

L43-76 の Step 2 冒頭（L45）を以下の構造に置換:

```markdown
### Step 2: 認証情報の準備

**認証要件マトリクス:**

| 使用する機能 | Google Cloud（Step 2a） | Azure OpenAI（Step 2b） |
|---|---|---|
| 回答支援AI + VertexAI埋め込み | **必須**（LLM + 埋め込み） | 不要 |
| 回答支援AI + Azure OpenAI埋め込み | **必須**（LLMのみ） | **必須**（埋め込み） |
| 運用保守効率化AI（`run_eval.py --provider both`） | **必須** | **必須** |
| 運用保守効率化AI（`run_eval.py --provider vertex`） | **必須** | 不要 |
| 運用保守効率化AI（`run_eval.py --provider azure`） | **必須**（LLMのみ） | **必須** |

> **要点:** Google Cloud 認証は**常に必須**です（LLM が Gemini のみ対応のため）。Azure OpenAI 認証は埋め込みプロバイダーの選択に依存します。
```

既存の Step 2a, 2b の内容はそのまま残す。L45 の散文と L76 の Note をこの表で置換。

**Step 2: 差分確認 → コミット**

```bash
git diff rag-local/README.md
git add rag-local/README.md
git commit -m "docs: README.md — replace auth prose with decision matrix table"
```

---

### Task 6: CONFIGURATION.md — DEFAULT_EMBEDDING_MODEL + MRL + keyword_weight 修正

**Files:**
- Modify: `docs/CONFIGURATION.md:30` (DEFAULT_EMBEDDING_MODEL Note)
- Modify: `docs/CONFIGURATION.md:157-165` (MRL記述)
- Modify: `docs/CONFIGURATION.md:273-275` (keyword_weight)
- Modify: `src/core/processor.py:230` (vestigial フォールバック削除)

**Step 1: DEFAULT_EMBEDDING_MODEL の Note を改善**

L30 を以下に置換:

```markdown
> **Note:** `DEFAULT_EMBEDDING_MODEL` は廃止されました。埋め込みモデルはプロバイダーに応じて自動決定されます（azure_openai → `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`、vertex_ai → `VERTEX_AI_EMBEDDING_MODEL`）。旧変数名を `.env` に設定しても無視されます。
```

**Step 2: MRL記述を実態に合わせて修正**

L157-165 を以下に置換:

```markdown
**特徴:**
- Google Cloud統合
- 次元数: 3072（固定）

**性能:**
- バッチサイズ: 5
- API上限: 250テキスト/リクエスト

> **Note:** Gemini embedding-001 は MRL（Matryoshka Representation Learning）に対応しており、API パラメータで次元数を 3072 / 1536 / 768 から選択可能ですが、本システムでは 3072 固定で使用しています。
```

**Step 3: keyword_weight の説明を改善**

L273-275 を以下に置換:

```markdown
### 重み調整

ベクトル検索とキーワード検索の重みバランスを調整します。`vector_weight` を設定すると、`keyword_weight` は `1.0 - vector_weight` で自動計算されます（`config.py` のプロパティ）。

設定場所は `config/settings.yaml` の各セクション（`common`, `ui`, `batch`, `evaluation.revision_areas`）。
```

**Step 4: processor.py の vestigial フォールバックを修正**

`processor.py:228-231` の `_execute_provider_search` 内で vertex_ai プロバイダー使用時のモデル解決が古い:
- 環境変数名: `DEFAULT_EMBEDDING_MODEL`（廃止済み）→ `VERTEX_AI_EMBEDDING_MODEL`（config.py:239 と統一）
- デフォルトモデル: `text-multilingual-embedding-002`（旧モデル）→ `gemini-embedding-001`（config.py:239 と統一）

修正:

```python
        else:  # vertex_ai
            provider_config.embedding_model = os.getenv(
                "VERTEX_AI_EMBEDDING_MODEL", "gemini-embedding-001"
            )
```

**Step 5: 差分確認 → コミット**

```bash
git diff docs/CONFIGURATION.md src/core/processor.py
git add docs/CONFIGURATION.md src/core/processor.py
git commit -m "docs: CONFIGURATION.md — fix MRL claim, improve descriptions; fix: processor.py vestigial embedding model fallback"
```

---

### Task 7: REVISION_OPS.md — 行番号移動 + マージ版定義 + 実験的指針 + reference/説明

**Files:**
- Modify: `docs/REVISION_OPS.md:20` (実験的の注記)
- Modify: `docs/REVISION_OPS.md:357-371` (参照データ管理)
- Modify: `docs/REVISION_OPS.md:386-390` (データ準備フロー)
- Modify: `docs/REVISION_OPS.md:404-411` (既知の問題)

**Step 1: 「実験的な対応」に運用指針を追加**

L20 の Note を以下に置換:

```markdown
> **Note:** 多段階検索は精度向上が実証された手法ではなく、**実験的な対応**である。回答支援AIでは原文検索（`original`）の方が精度が高いという結果が出ているが、改定影響調査では漏れを減らすために両方の結果を統合する方式を採用している。今後の評価結果次第では、`original` のみに戻す判断もありうる。
```

**Step 2: 参照データ管理セクションに入手方法を追加**

L359 を以下に置換:

```markdown
改定評価に必要な参照データは `reference/` ディレクトリに格納（git管理外）。初回セットアップ時は前任者から `reference/` フォルダ一式を受け取り、プロジェクトルートに配置する。
```

**Step 3: マージ版シナリオの定義を追加**

L386-390 の「変更前シナリオDB生成フロー」の先頭に定義を追加:

```markdown
### 変更前シナリオDB生成フロー

> **マージ版シナリオ** とは、ボットごとに全カテゴリのシナリオを1ファイルに結合した参照用Excelファイル（`reference/マージ版シナリオ/` に格納）。正解ID生成や変更前シナリオ作成の基準データとして使用する。自動生成スクリプトはなく、手動で管理する。

1. **マージ版シナリオ** + 修正前カテゴリファイル（手動でカテゴリを置換）→ 変更前シナリオ作成
```

**Step 4: 行番号の注意点を「使用方法」セクション付近に移動**

L409-410 の「行番号の注意点」を「既知の問題と注意事項」から切り出し、**入力ファイル仕様のセクション付近**（正解IDフォーマット説明の直後）に移動する。

移動先の正解IDフォーマット説明の後に以下を追加:

```markdown
> **重要:** 台帳記載行（カテゴリ内の行番号）と Excel行番号は異なる。正解IDには **Excel行番号** を使用する。計算式: `Excel行番号 = カテゴリ開始行 + カテゴリ内行番号 - 1`。`generate_correct_ids.py` は両方のパターンを自動で試行する。
```

元の「既知の問題」セクションからは「行番号の注意点」小見出しを削除。

**Step 5: 差分確認 → コミット**

```bash
git diff docs/REVISION_OPS.md
git add docs/REVISION_OPS.md
git commit -m "docs: REVISION_OPS.md — add merge scenario definition, move row number warning, clarify experimental note and reference data"
```

---

### Task 8: ANSWER_SUPPORT.md — DB更新フロー明確化 + ファイル名規則統一

**Files:**
- Modify: `docs/ANSWER_SUPPORT.md:122-128` (DB更新フロー)
- Modify: `docs/ANSWER_SUPPORT.md:134-136` (ファイル命名規則)

**Step 1: DB更新フローの説明を改善**

L122-128 を以下に置換:

```markdown
> **Note:** バッチ処理実行時、DB は自動更新されます。実行フロー:
> 1. `run_db_update()` で参照ファイル（`data/source/` 配下の FAQ Excel / シナリオ Excel）の**ファイル更新日時**を `data/vector_db/update_timestamps.json` の記録と比較
> 2. ファイル未更新 + DB既存 → スキップ（埋め込みAPI呼び出しなし = コスト発生なし）
> 3. ファイル更新あり or DB未存在 → 埋め込みAPIを呼び出してDB構築/更新
> 4. DB 更新完了後、バッチ処理を開始
>
> UI（インタラクティブ）モードでは DB更新を実行しません。
```

**Step 2: ファイル命名規則を修正**

L134-136 を以下に置換:

```markdown
- `{業務分野名}`: `config/business_areas.yaml` に登録されている名前。日本語名（例: `スマイル`）でも英語名（例: `smile`）でも可。日本語名は英語DB名に自動変換される
- `{YYYYMMDD}`: データ日付。同一業務分野に複数ファイルがある場合、最新日付のファイルが使用される
```

**Step 3: 差分確認 → コミット**

```bash
git diff docs/ANSWER_SUPPORT.md
git add docs/ANSWER_SUPPORT.md
git commit -m "docs: ANSWER_SUPPORT.md — clarify DB update mechanism and filename convention"
```

---

## 修正しないもの（判断根拠）

| 項目 | 理由 |
|------|------|
| processor.py:230 の DEFAULT_EMBEDDING_MODEL フォールバック | Task 6 Step 4 で修正する（環境変数名 + デフォルトモデル名を config.py と統一） |
| 2つのAIシステムの定義 | 引き継ぎ先は既知（ユーザー確認済み） |
| 業務ドメイン用語の定義 | 引き継ぎ先は既知（ユーザー確認済み） |
| Sudachi / Jaccard の説明 | 技術者向けドキュメントとして許容範囲 |
| ANSWER_SUPPORT.md L147 TextInputHandler 言及 | 改定影響調査を知っている人には問題なし |
