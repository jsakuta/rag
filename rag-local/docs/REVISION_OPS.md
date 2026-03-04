# 改定影響調査システム

事務改定前のシナリオデータをベクトル化し、改定内容をクエリとして検索精度を評価するシステム。

## 概要

### 目的
- 事務改定によって変更されるシナリオ行（正解ID）を、改定内容の説明文から正しく検索できるかを評価
- Azure OpenAI と VertexAI の2つの埋め込みプロバイダーで精度を**横並び比較**
- **多段階ハイブリッド検索**（原文 + LLM強化クエリ）で検索精度を向上

### 処理フロー
```
1. build_db.py --revisions-only
   - 変更前シナリオExcelをベクトル化
   - Azure OpenAI / VertexAI 両方でDB構築

2. run_eval.py（多段階検索版）
   - 多段階ハイブリッド検索を実行
     - Stage 1: 原文クエリでハイブリッド検索
     - Stage 2: LLM強化クエリでハイブリッド検索
     - Stage 3: 結果をマージ＋カテゴリ分類
   - LLM分析（JudgmentSupport）で関連性判定
   - Azure / VertexAI 横並びでExcel出力
```

---

## 多段階ハイブリッド検索

### スコア計算
```
combined_score = vector_weight × ベクトル類似度 + keyword_weight × キーワード類似度
                 (デフォルト0.9)                    (デフォルト0.1)
```

- **ベクトル類似度**: 埋め込みモデルによる意味的類似度
- **キーワード類似度**: Jaccard類似度（Sudachi抽出キーワードの共通割合）

### 検索フロー
```
入力クエリ
    ↓
キーワード抽出（Sudachi: 名詞Top-5）
    ↓
┌─ Stage 1: 原文クエリでハイブリッド検索
│
└─ Stage 2: LLM強化クエリでハイブリッド検索
       QueryEnhancer.enhance() でクエリ生成
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
- **keyword_filter**: ベクトル検索をスキップ。ChromaDB キーワードキャッシュのみ使用（多段階検索は実行しない）。用語の単純置換（AML→GPLEX等）の検出に適する

### 設定値
| パラメータ | 値 | 説明 |
|-----------|---|------|
| FILTER_MODE | top_k | フィルタリング方式（`top_k` / `threshold`）。settings.yaml で指定、コード上のフォールバックは `threshold` |
| TOP_K | 130 | 上位件数（filter_mode=top_k 時に使用） |
| THRESHOLDS | Azure=0.40, VertexAI=0.50 | プロバイダー別閾値（filter_mode=threshold 時に使用） |
| VECTOR_WEIGHT | 0.9 | ベクトルスコアの重み |
| MAX_RESULTS | 100 | 各検索の最大結果数 |

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
- 埋め込みベクトルの次元・特性がプロバイダーによって異なる
- 同じChromaDBコレクションに異なるモデルのベクトルは混在不可
- 検索時はクエリと同じモデルでベクトル化されたDBを使用する必要がある

| プロバイダー | 埋め込みモデル | 次元数 |
|-------------|---------------|--------|
| Azure OpenAI | text-embedding-3-large | 3072 |
| VertexAI | gemini-embedding-001 | 3072 |

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
| ⑦ | - | 積立定期預金 | smile-bot | rev07_smile（未評価） |

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

3. **環境変数の設定**（`.env`）
   ```bash
   # Azure OpenAI（--provider both または --provider azure 使用時）
   AZURE_OPENAI_API_KEY=your-api-key
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
   AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large

   # VertexAI（--provider both または --provider vertex 使用時）
   GEMINI_PROJECT_ID=your-project-id
   VERTEX_AI_EMBEDDING_MODEL=gemini-embedding-001

   # LLM（クエリ拡張・関連性判定、Gemini のみ対応）
   DEFAULT_LLM_PROVIDER=gemini
   DEFAULT_LLM_MODEL=gemini-2.5-flash-lite

   # LLM関連性判定の有効化（オプション、デフォルト: false）
   # JudgmentSupport による関連性判定のみを制御。LLMクエリ拡張（Stage 2）は常に有効
   ENABLE_LLM_ANALYSIS=true
   ```

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

### フォルダ構造

改定評価に必要な参照データは `reference/` ディレクトリに格納されています（git管理外）。

```
reference/
├── 改定内容/                       # 改定内容の説明 (revXX_*.md)
├── 改定シナリオ/
│   ├── rev01_スマイル機能変更/
│   │   ├── 差分.md                 # 統一フォーマットの差分ファイル
│   │   ├── 修正前/
│   │   ├── 修正後/
│   │   └── 参考資料/               # 協議書・通達 (PDF/DOCX/PPTX)
│   ├── rev02_相続少額払い/
│   ├── rev03_保険証→資格確認証/
│   ├── rev04_0円新規開設可能/
│   ├── rev05_AMLフィルター→GPLEX/
│   └── rev06_DC→MDC/
├── マージ版シナリオ/
│   ├── 改定前/                     # 改定評価用マージ版
│   └── 最新/                       # 最新版マージ版
├── 問い合わせ履歴/
└── シナリオボットメンテナンス管理台帳.xlsx
```

### 差分ファイルの書き方（統一フォーマット）

各改定の差分は以下のフォーマットで `差分.md` に記載します:

```markdown
# X番号_タイトル - 事務改定差分

## メンテナンス管理台帳との照合

**台帳No.XXの記載**:
- ボット名: XXX
- 大分類: XXX
- 変更箇所: 行番号X, Y, Z

**変更行一覧（メンテ台帳 vs 実際の差分）**:

| 台帳記載行 | Excel行 | 実際の差分 | 状態 |
|-----------|---------|-----------|------|
| X | Y | あり | 一致 |

## ボット名-bot

### ファイル: シナリオ_XXX.xlsx

**カテゴリ**: Lv1=XXX
**変更前シナリオExcelでの範囲**: 行X～行Y

**黄色ハイライト行（変更前シナリオExcel）**: N行
- 行番号: X, Y, Z

---
変更箇所 N: **カテゴリ内行X** (Excel行Y)
質問遷移: A → B → C

**LvN**:
- 変更前: `...`
- 変更後: `...`

**合計 N 行に変更あり**
```

---

## データ準備の詳細フロー

### 変更前シナリオDB生成フロー

```
1. マージ版シナリオ (reference/マージ版シナリオ/最新/マージ版シナリオ_XXX-bot.xlsx)
        ↓
    + 修正前カテゴリファイル (手動でカテゴリを置換)
        ↓ [自動化スクリプトなし - 手動作成]

2. 変更前シナリオ (reference/改定シナリオ/revXX_*/修正前/revXX_変更前シナリオ_XXX-bot.xlsx)
        ↓
   prepare_before_scenario.py (文字数列削除・リネーム)
        ↓
3. data/source/scenarios/revisions/revXXボット_シナリオデータ_YYYYMMDD.xlsx
        ↓
   build_db.py --revisions-only → DynamicDBManager
        ↓
4. data/vector_db/revXX_{bot}/ (ベクトルDB)
```

### 重要なスクリプト

| スクリプト | 用途 |
|-----------|------|
| `scripts/prepare_before_scenario.py` | 変更前シナリオの前処理（列削除・リネーム） |
| `scripts/build_db.py` | DB構築（回答支援AI用 + 改定別、統合スクリプト） |
| `scripts/generate_correct_ids.py` | 正解ID対応表生成 |

### DB構築コマンド

```bash
# 改定別DBのみ構築（Azure OpenAI + VertexAI 両方）
python scripts/build_db.py --revisions-only

# 全DB一括構築（回答支援AI用 + 改定別）
python scripts/build_db.py --force
```

---

## 正解ID抽出ロジック (generate_correct_ids.py)

### 抽出パターン
1. ボット名: `## smile-bot` 形式のセクションヘッダー
2. 行番号リスト: `- 行番号: 129, 185` または `行番号: X, Y, Z`
3. 変更箇所: `**カテゴリ内行X** (Excel行Y)` → Excel行Yを抽出

### 出力
- ファイル: `data/input/multi_stage_input.xlsx`
- 列: 番号, 改定内容, 正解ID

---

## 既知の問題と注意事項

### 空行問題
**原因**: 修正前カテゴリファイルの末尾に空行が含まれている
- 例: `修正前/喪失/シナリオ_スマイルタブレット_喪失_20250731.xlsx` の行132が空行
- この空行が変更前シナリオにマージされ、Excel行365として残存

**対応案**:
1. `prepare_before_scenario.py` に空行フィルタリング追加: `df = df[df['Lv1'].notna()]`
2. 元ファイルを手動で修正

### 行番号の注意点
- **台帳記載行**: メンテナンス管理台帳に記載された行番号（カテゴリ内の行番号）
- **Excel行番号**: 変更前シナリオExcelでの実際の行番号（ヘッダー行1 + データ行）
- 計算式: `Excel行番号 = カテゴリ開始行 + カテゴリ内行番号 - 1`

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

```bash
# Azure OpenAI 未設定の場合は --provider vertex を指定
python apps/revision-ops/run_eval.py --provider vertex
# 両プロバイダー設定済みの場合
python apps/revision-ops/run_eval.py
```

### 設定変更のみのケース

既存改定のパラメータ調整（`vector_weight` や `search_type` の変更）は `config/settings.yaml` の編集だけで完了する。DB再構築は不要。

---

## 関連ファイル

| ファイル | 説明 |
|---------|------|
| `scripts/build_db.py` | DB構築スクリプト（回答支援AI（類似回答検索）+ 改定別 統合） |
| `apps/revision-ops/run_eval.py` | 評価スクリプト（多段階検索版） |
| `scripts/generate_correct_ids.py` | 正解ID対応表生成 |
| `scripts/prepare_before_scenario.py` | 変更前シナリオの前処理 |
| `src/core/search/multi_stage_orchestrator.py` | 多段階検索オーケストレーター |
| `src/core/search/query_enhancer.py` | LLMクエリ拡張エンジン |
| `src/core/search/keyword_search_engine.py` | キーワード検索エンジン |
| `src/core/search/vector_search_engine.py` | ベクトル検索エンジン |
| `src/core/judgment_support.py` | LLM判断支援クラス |
| `src/utils/dynamic_db_manager.py` | DB管理クラス |
| `src/utils/gemini_embedding.py` | VertexAI埋め込みモデル |
| `src/utils/azure_embedding.py` | Azure OpenAI埋め込みモデル |
