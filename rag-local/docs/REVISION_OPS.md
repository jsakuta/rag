# 運用保守効率化AI（改定影響調査）

事務改定前のシナリオデータおよび問い合わせ履歴データ（FAQ）を検索用の数値データに変換（ベクトル化）し、改定内容の説明文を検索語として、影響を受けるシナリオや問い合わせ履歴を検索するシステム。正解ID付きの精度評価と、正解IDなしの実運用向け影響範囲調査の両方に対応する。

## 概要

### 目的
- 事務改定によって変更されるシナリオ行（正解ID）を、改定内容の説明文から正しく検索できるかを評価
- Azure OpenAI と VertexAI の2つの埋め込みプロバイダー（文章を数値に変換するサービス）で精度を**横並び比較**
- **多段階ハイブリッド検索**（原文での検索と、AIが補強したクエリでの検索を組み合わせて候補を網羅する方式）で検索の再現率（漏れの少なさ）を向上

> **2プロバイダー比較の経緯:** 回答支援AIでは VertexAI のみで開発していたが、クラウド実装は B&DX Azure 上で Azure AI Search（組み込みハイブリッド検索・Semantic Ranker 等）を活用する構成が適している。Azure 切替時の精度低下リスクを検証するため、ローカル段階から両モデルで並行検証した結果、同等の精度が確認され、Azure を採用する方向で事務企画部内で合意している。

> **多段階検索について:** 回答支援AIでは原文検索（`original`）の方が精度が高いが、改定影響調査は再現率重視（影響範囲を漏れなく検出）のため、原文検索とLLMクエリ拡張の結果をOR結合する多段階検索を採用している。ただし効果は定量的に検証されておらず、評価結果次第で `original` のみに戻す判断もありうる。

### 処理フロー
```
1. build_db.py --revisions-only
   - 変更前シナリオExcelをベクトル化
   - Azure OpenAI / VertexAI 両方でDB構築

2. run_eval.py（多段階検索版）
   - 多段階ハイブリッド検索を実行
     - Stage 1: 原文クエリでハイブリッド検索
     - Stage 2: LLM（大規模言語モデル）で補強したクエリでハイブリッド検索
     - Stage 3: 結果をマージ＋カテゴリ分類
   - LLM分析（JudgmentSupport）で関連性判定
   - Azure / VertexAI 横並びでExcel出力
```

---

## 多段階ハイブリッド検索

### スコア計算

スコア計算式は回答支援AIと共通。詳細は [ANSWER_SUPPORT.md のスコア計算式](./ANSWER_SUPPORT.md#スコア計算式) を参照。

### 検索フロー
```
入力クエリ
    ↓
キーワード抽出（Sudachi で名詞を最大5個抽出）
    ↓
┌─ Stage 1: 原文クエリでハイブリッド検索
│
└─ Stage 2: LLM強化クエリでハイブリッド検索
       QueryEnhancer.enhance() でAIが検索語を自動補強
    ↓
Stage 3: 結果をマージ＋カテゴリ分類
    - Both: 両方で見つかった（信頼度高）
    - Original_Only: Stage 1のみで見つかった
    - LLM_Enhanced_Only: Stage 2のみで見つかった
    ↓
フィルタリング（filter_mode による切替）
```

### 補足: search_type による動作の違い

- **hybrid**: 多段階検索フロー（Stage 1/2/3）を実行。ベクトル+キーワードのハイブリッド検索
- **keyword_filter**: 意味検索（ベクトル検索）をスキップし、キーワードの一致のみで検索する方式（多段階検索は実行しない）。用語の単純な置き換え（AML→GPLEX等）の検出に適する

### 設定値

改定影響調査でのデフォルト設定値:

| パラメータ | 値 | 説明 |
|-----------|---|------|
| FILTER_MODE | top_k | フィルタリング方式（`top_k` / `threshold`）。settings.yaml で指定、コード上のフォールバックは `threshold` |
| TOP_K | 130 | 上位件数（filter_mode=top_k 時に使用） |
| THRESHOLDS | Azure=0.40, VertexAI=0.50 | プロバイダー別閾値（filter_mode=threshold 時に使用） |
| VECTOR_WEIGHT | 0.9 | ベクトルスコアの重み |
| MAX_RESULTS | 100 | 各検索の最大結果数 |

> **TOP_K=130 の設計意図:**
> 改定影響調査は再現率（漏れの少なさ）を重視するため、閾値ではなく固定件数で上位を取る方式を採用している。閾値モード（`threshold`）はプロバイダー間でスコア分布が異なり、同じ閾値では候補数が大きく変動するため不安定だった。130件は、Stage 1（原文検索）+ Stage 2（LLM拡張検索）のOR結合（最大200件）から上位を取り、コレクション規模（数百件）の約半数をカバーする水準として設定した値。
>
> **注意:** この制限は**エリア単位**で適用される。複数エリアの改定（例: ③=4エリア）では、Excel出力の総件数は最大 130 × エリア数 × プロバイダー数 になる。

---

## DB構造

### ディレクトリ構成
```
data/vector_db/
├── rev01_smile/          # 事務改定①用（smile-bot）
│   ├── azure_openai/
│   │   └── chroma.sqlite3
│   └── vertex_ai/
│       └── chroma.sqlite3
├── rev02_souzoku/        # 事務改定②用（souzoku-bot）
├── rev03_naibujimu/      # 事務改定③用（naibujimu-bot）
├── rev03_smile/          # 事務改定③用（smile-bot）
├── rev03_souzoku/        # 事務改定③用（souzoku-bot）
├── rev03_torikaku/       # 事務改定③用（torikaku-bot）
├── rev04_naibujimu/      # 事務改定④用
├── rev05_smile/          # 事務改定⑤用
├── rev06_smile/          # 事務改定⑥用
└── update_timestamps.json
```

### プロバイダー別DBの理由
- 文章を数値に変換した結果（ベクトル）の特性がプロバイダーによって異なる
- 同じデータベースに異なるモデルで変換した数値は混在できない
- 検索時は、DB構築時と同じモデルで質問文を変換する必要がある

対応プロバイダーの詳細は [ANSWER_SUPPORT.md](./ANSWER_SUPPORT.md#埋め込みプロバイダー) を参照。

---

## 改定番号とDBの対応

| 改定番号 | 台帳No. | 内容 | 対象ボット | 対応DB |
|---------|--------|------|----------|--------|
| ① | 20 | スマイル機能変更 | smile-bot | rev01_smile |
| ② | 21 | 相続少額払い | souzoku-bot | rev02_souzoku |
| ③ | 25-30, 35-36 | 保険証→資格確認証 | smile-bot, naibujimu-bot, souzoku-bot, torikaku-bot | rev03_smile, rev03_naibujimu, rev03_souzoku, rev03_torikaku |
| ④ | 37 | 0円新規開設可能 | naibujimu-bot | rev04_naibujimu |
| ⑤ | 41-42 | AML→GPLEX | smile-bot | rev05_smile |
| ⑥ | 43-45 | DC→MDC | smile-bot | rev06_smile |

---

## 使用方法

### 前提条件

1. **シナリオファイルの配置**
   ```
   data/source/scenarios/revisions/
   ├── rev01smile_シナリオデータ_YYYYMMDD.xlsx
   ├── rev02souzoku_シナリオデータ_YYYYMMDD.xlsx
   ├── rev03naibujimu_シナリオデータ_YYYYMMDD.xlsx
   ├── rev03smile_シナリオデータ_YYYYMMDD.xlsx
   ├── rev03souzoku_シナリオデータ_YYYYMMDD.xlsx
   ├── rev03torikaku_シナリオデータ_YYYYMMDD.xlsx
   ├── rev04naibujimu_シナリオデータ_YYYYMMDD.xlsx
   ├── rev05smile_シナリオデータ_YYYYMMDD.xlsx
   └── rev06smile_シナリオデータ_YYYYMMDD.xlsx
   ```

2. **入力ファイルの配置**
   ```
   data/input/multi_stage_input.xlsx
   ```

   必要な列:
   - `番号`: 改定番号（①②③④⑤⑥）
   - `改定内容`: 検索クエリとなる改定の説明文
   - `正解ID`: カンマ区切りの正解シナリオID（例: `smile-bot_129, smile-bot_185`）
   - `変更内容`（オプション）: 各正解IDごとの具体的な変更内容。未検出シナリオ（「未発見」セクション）に表示される。省略時は自動的に空で初期化

3. **環境変数の設定** — [CONFIGURATION.md](./CONFIGURATION.md) を参照。改定影響調査固有の設定: `ENABLE_LLM_ANALYSIS=true`（LLM関連性判定、デフォルト: false）

### Step 1: DB再構築

```bash
# Streamlit UIを停止してから実行
python scripts/build_db.py --revisions-only
```

処理内容（`--force` なしの場合は差分構築）:
- タイムスタンプを確認し、更新があるDBのみ再構築
- `--force` 指定時: 既存のrev* DBディレクトリを削除 → タイムスタンプリセット → 全DB再構築
- 全9つのDBをAzure OpenAI / VertexAI両方で構築

### Step 2: 評価実行

```bash
# 両プロバイダーで実行（デフォルト、Azure OpenAI の環境変数が必須）
python apps/revision-ops/run_eval.py

# プロバイダーを指定して実行
python apps/revision-ops/run_eval.py --provider vertex   # VertexAI のみ
python apps/revision-ops/run_eval.py --provider azure    # Azure のみ
python apps/revision-ops/run_eval.py --provider both     # 両方（デフォルト）

# 詳細設定を表示して実行
python apps/revision-ops/run_eval.py --verbose
```

> **Note:** デフォルト（`--provider both`）は Azure OpenAI の環境変数（`AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`）が必須です。未設定の場合は `--provider vertex` を指定してください。

処理内容:
- 入力ファイルから改定内容と正解IDを読み込み
- 各改定に対応するDBで多段階ハイブリッド検索を実行
- LLM分析で関連性判定を実行
- Azure / VertexAI 横並びでExcelに出力

### Streamlit UI（ops_ui.py）

バッチ処理（run_eval.py）に加えて、Streamlit UIでも改定影響調査を実行できます。UIは2つのモードを提供します。

```bash
streamlit run apps/revision-ops/ui/ops_ui.py
```

| モード | 用途 | 正解ID | 主な機能 |
|--------|------|--------|---------|
| **評価モード** | 検索精度の定量評価 | 必要 | 正解ID付きExcel入力で正解発見率を計算。プロバイダー横並び比較 |
| **影響調査モード** | 実運用向け影響範囲調査 | 不要 | 改定内容を入力して影響候補を一覧表示。正解IDなしで使用可能 |

**影響調査モードの特徴:**
- **データソース選択**: シナリオ（scenario） / 問い合わせ履歴データ（history_data）を切替可能
- **業務エリア選択**: 検索対象の業務エリアを選択（複数選択対応）
- **検索タイプ選択**: hybrid（ベクトル+キーワード） / keyword_filter（キーワードのみ）
- **プロバイダー並列検索**: Azure OpenAI / VertexAI の結果を並列取得して比較

> **Note:** 評価モードはバッチ版（run_eval.py）と同等の精度検証が可能ですが、大量データの処理にはバッチ版を推奨します。影響調査モードは正解IDが不要なため、新規改定の初期調査に適しています。

### LLM関連性判定の無効化

`ENABLE_LLM_ANALYSIS` は **JudgmentSupport（関連性判定）のみ**を制御します。LLMクエリ拡張（Stage 2 の QueryEnhancer）は常に有効です。

```bash
# 関連性判定を無効化して高速実行（クエリ拡張は引き続き実行される）
ENABLE_LLM_ANALYSIS=false python apps/revision-ops/run_eval.py
```

---

## 出力ファイル

### ファイル名
```
data/output/latest/rev/rev_eval_batch_YYYYMMDD_HHMMSS.xlsx
```

### シート構成

1. **サマリーシート**

   ヘッダーは2行構成: 1行目がグループ（Azure / VertexAI / 未発見）、2行目が列名。

   | # | 列名 | 説明 |
   |---|------|------|
   | 1 | 改定番号 | ①②③④⑤⑥ |
   | 2 | エリア | 検索対象エリア（改定③のように複数エリアがある場合はエリアごとに行が分かれる） |
   | 3 | 改定内容 | クエリの全文 |
   | 4 | 正解数 | 正解IDの総数 |
   | 5 | Azure_候補数 | Azure検索候補数（filter_mode=top_kなら上位K件、thresholdなら閾値以上） |
   | 6 | Azure_正解発見数 | Azure候補のうち正解と一致した数 |
   | 7 | Azure_正解発見率 | 正解発見数 / 正解数 |
   | 8 | Azure_必要確認件数 | 候補数 - 正解発見数（人手で確認が必要な件数） |
   | 9 | VertexAI_候補数 | VertexAI検索候補数（同上） |
   | 10 | VertexAI_正解発見数 | VertexAI候補のうち正解と一致した数 |
   | 11 | VertexAI_正解発見率 | 正解発見数 / 正解数 |
   | 12 | VertexAI_必要確認件数 | 候補数 - 正解発見数 |
   | 13 | 未発見数 | どちらのプロバイダーでも発見できなかった正解IDの数 |
   | 14 | 未発見ID | 未発見の正解IDリスト（カンマ区切り） |

2. **詳細シート（①～⑥）** - Azure / VertexAI / 未発見の横並び

   #### 共通列（7列）
   | # | 列名 | 説明 |
   |---|------|------|
   | 1 | 検出フラグ | AzureまたはVertexAIのどちらかで正解なら `TRUE`（OR判定） |
   | 2 | 改定内容 | 検索クエリ（全文、全行に出力） |
   | 3 | 正解ID一覧 | 期待される正解ID（1行目のみ） |
   | 4 | LLM強化クエリ | QueryEnhancerで生成されたクエリ（1行目のみ） |
   | 5 | 抽出キーワード | Sudachiで抽出したキーワード（1行目のみ） |
   | 6 | 検索タイプ | `類似検索`（hybrid時）/ `キーワード必須`（keyword_filter時）（1行目のみ） |
   | 7 | ベクトル重み | vector_weight（hybrid時のみ有効、keyword_filter時は `0.0`）（1行目のみ） |

   #### Azure側（10列: `Azure_` プレフィックス付き）
   | # | 列名 | 説明 |
   |---|------|------|
   | 1 | Azure_順位 | 検索結果の順位 |
   | 2 | Azure_シナリオID | 検索ヒットID |
   | 3 | Azure_類似度 | ハイブリッドスコア |
   | 4 | Azure_マッチ種別 | Both / Original_Only / LLM_Enhanced_Only |
   | 5 | Azure_正解フラグ | TRUE / FALSE |
   | 6 | Azure_質問 | 検索結果の質問文（全文） |
   | 7 | Azure_回答 | 検索結果の回答文（全文） |
   | 8 | Azure_関連性判定 | LLMによる関連性判定結果（`ENABLE_LLM_ANALYSIS=true` 時のみ） |
   | 9 | Azure_判定根拠 | LLMの判定理由（同上） |
   | 10 | Azure_ソースファイル | 元シナリオファイル名（revision_source_files マッピングから特定） |

   #### VertexAI側（10列: `VertexAI_` プレフィックス付き）

   Azure側と同じ列構成（プレフィックスが `VertexAI_` に変わるのみ）。

   #### 未発見セクション（5列: `未発見_` プレフィックス付き）
   | # | 列名 | 説明 |
   |---|------|------|
   | 1 | 未発見_未発見ID | どちらのプロバイダーでも発見できなかったシナリオID |
   | 2 | 未発見_変更内容 | 該当シナリオの変更内容 |
   | 3 | 未発見_ソースファイル | 該当シナリオの元ファイル名 |
   | 4 | 未発見_質問 | 該当シナリオの質問文 |
   | 5 | 未発見_回答 | 該当シナリオの回答文 |

   **合計: 共通7列 + Azure 10列 + VertexAI 10列 + 未発見5列 = 32列**

   **注意**: Azure/VertexAI/未発見で件数が異なる場合、最大件数に合わせて少ない側は空欄で埋められる

---

## 正解IDフォーマット

```
{ボット名}_{Excel行番号}
```

### ボット名対応表
| ボット名 | 対象システム |
|---------|-------------|
| smile-bot | スマイルタブレット |
| naibujimu-bot | 内部事務 |
| souzoku-bot | 相続 |
| torikaku-bot | 取引時確認 |

### 例
- `smile-bot_129` : smile-botのExcel129行目
- `naibujimu-bot_641` : naibujimu-botのExcel641行目

> **重要:** 台帳記載行（カテゴリ内の行番号）と Excel行番号は異なる。正解IDには **Excel行番号** を使用する。計算式: `Excel行番号 = カテゴリ開始行 + カテゴリ内行番号 - 1`。`generate_correct_ids.py` は両方のパターンを自動で試行する。

---

## トラブルシューティング

### DBが削除できない
```
rm: cannot remove '...': Device or resource busy
```
→ Streamlit UIやPythonプロセスがDBを使用中。すべて停止してから再実行。

### DBが見つからない
```
DBが存在しません: data/vector_db/rev01_smile/azure_openai
```
→ `build_db.py --revisions-only`を先に実行してDBを構築。

### 正解IDが一致しない
- 正解IDのフォーマットを確認（`{ボット名}_{Excel行番号}`）
- シナリオファイルのメタデータ（row_index, source）を確認
- `generate_correct_ids.py`で正解ID対応表を再生成

### LLM関連性判定が遅い
- `ENABLE_LLM_ANALYSIS=false` で関連性判定のみ無効化可能（クエリ拡張は引き続き実行）
- 各候補に対してLLM呼び出しが発生するため、候補数が多いと時間がかかる

---

## 参照データ管理

改定評価に必要な参照データは `reference/` ディレクトリに格納（git管理外）。初回セットアップ時は前任者から `reference/` フォルダ一式を受け取り、プロジェクトルートに配置する。

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

### 差分ファイルの書き方

各改定の差分は `差分.md` に統一フォーマットで記載する。既存の `reference/改定シナリオ/rev01_スマイル機能変更/差分.md` をテンプレートとして使用。主な構成要素:

- メンテナンス管理台帳との照合（台帳No.、変更行一覧）
- ボット名ごとのセクション（ファイル名、カテゴリ、変更箇所の詳細）
- 変更前・変更後の対比

---

## データ準備の詳細フロー

### 変更前シナリオDB生成フロー

> **マージ版シナリオ** とは、ボットごとに全カテゴリのシナリオを1ファイルに結合した参照用Excelファイル（`reference/マージ版シナリオ/` に格納）。正解ID生成や変更前シナリオ作成の基準データとして使用する。自動生成スクリプトはなく、手動で管理する。

1. **マージ版シナリオ** + 修正前カテゴリファイル（手動でカテゴリを置換）→ 変更前シナリオ作成
2. `prepare_before_scenario.py` で前処理（文字数列削除・リネーム）→ `data/source/scenarios/revisions/` に出力
3. `build_db.py --revisions-only` → `data/vector_db/revXX_{bot}/` にベクトルDB生成

### 重要なスクリプト

| スクリプト | 用途 |
|-----------|------|
| `scripts/prepare_before_scenario.py` | 変更前シナリオの前処理（列削除・リネーム） |
| `scripts/build_db.py` | DB構築（回答支援AI用 + 改定別、統合スクリプト） |
| `scripts/generate_correct_ids.py` | 正解ID対応表生成 |

DB構築コマンドの全オプションは [README.md の Step 5](../README.md#step-5-db構築) を参照。改定DB構築は `python scripts/build_db.py --revisions-only`。

---

## 既知の問題と注意事項

### 空行問題
修正前カテゴリファイルの末尾に空行が含まれていると、変更前シナリオにマージされて残存する。対応案: `prepare_before_scenario.py` に `df = df[df['Lv1'].notna()]` を追加、または元ファイルを手動修正。

---

## 新しい改定の追加手順

例: 事務改定⑦（為替対応）を追加する場合

### Step 1: 変更前シナリオの準備

```bash
# 修正前カテゴリExcelを前処理（文字数列削除・リネーム）
python scripts/prepare_before_scenario.py
```

出力先: `data/source/scenarios/revisions/rev07為替_シナリオデータ_YYYYMMDD.xlsx`

### Step 2: マッピング登録（2箇所）

**2つの設定ファイルの役割:**
- `business_areas.yaml` の `revision_mappings`: エリア名と ChromaDB コレクション名の**物理マッピング**（DB検索時の名前解決に使用）
- `settings.yaml` の `evaluation.revision_areas`: 改定番号と検索設定の**ロジックマッピング**（search_type, vector_weight 等のパラメータ）

**config/business_areas.yaml** — `revision_mappings` に追加:

```yaml
revision_mappings:
  # ⑦為替対応
  rev07_smile: rev07_smile
```

**config/settings.yaml** — `evaluation.revision_areas` に追加:

```yaml
revision_areas:
  "⑦":
    areas:
      - rev07_smile
    search_type: hybrid      # hybrid or keyword_filter
    vector_weight: 0.9       # keyword_filter の場合は不要
```

`search_type` の選択基準:
- `hybrid`: 意味的な内容変更（説明文の書き換え、手順の追加等）
- `keyword_filter`: 単純な用語置換（例: ⑤AML→GPLEX、⑥DC→MDC）

新しいボットが関わる場合は `area_to_bot` と `area_to_category` にも追加が必要。

### Step 3: 正解ID生成

```bash
python scripts/generate_correct_ids.py
```

入力: `reference/改定シナリオ/rev07_*/差分.md`
出力: `data/input/multi_stage_input.xlsx` の該当行に正解IDが追加される

### Step 4: DB構築

```bash
# Streamlit UI を停止してから実行
python scripts/build_db.py --revisions-only
```

確認:
```bash
python scripts/check_db_content.py
```

### Step 5: 評価実行

[Step 2: 評価実行](#step-2-評価実行) と同じコマンドで実行。

### 設定変更のみのケース

既存改定のパラメータ調整（`vector_weight` や `search_type` の変更）は `config/settings.yaml` の編集だけで完了する。DB再構築は不要。

