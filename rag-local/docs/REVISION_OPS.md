# 改定影響調査システム（多段階検索・横並び比較版）

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

2. evaluate_revisions.py（多段階検索版）
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
閾値フィルタリング（≥0.45）
```

### 設定値
| パラメータ | 値 | 説明 |
|-----------|---|------|
| THRESHOLD | 0.45 | 統合スコアの閾値 |
| VECTOR_WEIGHT | 0.9 | ベクトルスコアの重み |
| MAX_RESULTS | 100 | 各検索の最大結果数 |

---

## DB構造

### ディレクトリ構成
```
data/vector_db/
├── general/              # 総則（通常検索用）
│   ├── azure_openai/
│   └── vertex_ai/
├── rev01smile/           # 事務改定①用（smile-bot）
│   ├── azure_openai/
│   │   └── chroma.sqlite3
│   └── vertex_ai/
│       └── chroma.sqlite3
├── rev02souzoku/         # 事務改定②用（souzoku-bot）
├── rev03naibujimu/       # 事務改定③用（naibujimu-bot）
├── rev03smile/           # 事務改定③用（smile-bot）
├── rev03souzoku/         # 事務改定③用（souzoku-bot）
├── rev03torikaku/        # 事務改定③用（torikaku-bot）
├── rev04naibujimu/       # 事務改定④用
├── rev05smile/           # 事務改定⑤用
├── rev06smile/           # 事務改定⑥用
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

| 改定番号 | 台帳No. | 内容 | 対応DB |
|---------|--------|------|--------|
| ① | 20 | スマイル機能変更 | rev01smile |
| ② | 21 | 相続少額払い | rev02souzoku |
| ③ | 25-30, 35-36 | 保険証→資格確認証 | rev03naibujimu, rev03smile, rev03souzoku, rev03torikaku |
| ④ | 37 | 0円新規開設可能 | rev04naibujimu |
| ⑤ | 41-42 | AML→GPLEX | rev05smile |
| ⑥ | 43-45 | DC→MDC | rev06smile |

---

## 使用方法

### 前提条件

1. **シナリオファイルの配置**
   ```
   data/source/scenarios/
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

3. **環境変数の設定**（`.env`）
   ```bash
   # Azure OpenAI
   AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large

   # VertexAI
   VERTEX_AI_EMBEDDING_MODEL=gemini-embedding-001

   # LLM分析の有効化（オプション、デフォルト: true）
   ENABLE_LLM_ANALYSIS=true
   ```

### Step 1: DB再構築

```bash
# Streamlit UIを停止してから実行
python scripts/build_db.py --revisions-only
```

処理内容:
- 既存のrev* DBディレクトリを削除
- タイムスタンプファイルをリセット
- 全9つのDBをAzure OpenAI / VertexAI両方で構築

### Step 2: 評価実行

```bash
python scripts/evaluate_revisions.py
```

処理内容:
- 入力ファイルから改定内容と正解IDを読み込み
- 各改定に対応するDBで多段階ハイブリッド検索を実行
- LLM分析で関連性判定を実行
- Azure / VertexAI 横並びでExcelに出力

### LLM分析の無効化

```bash
# LLM分析を無効化して高速実行
ENABLE_LLM_ANALYSIS=false python scripts/evaluate_revisions.py
```

---

## 出力ファイル

### ファイル名
```
data/output/revision_evaluation_YYYYMMDD_HHMMSS.xlsx
```

### シート構成

1. **サマリーシート**

   | 列名 | 説明 |
   |-----|------|
   | 改定番号 | ①②③④⑤⑥ |
   | 改定内容（先頭50文字） | クエリの概要 |
   | 正解ID数 | 正解IDの総数 |
   | Azure_候補数 | Azure閾値0.45以上の候補数 |
   | Azure_正解一致数 | Azure候補のうち正解数 |
   | VertexAI_候補数 | VertexAI閾値0.45以上の候補数 |
   | VertexAI_正解一致数 | VertexAI候補のうち正解数 |

2. **詳細シート（①～⑥）** - Azure / VertexAI 横並び

   #### 共通列（検索条件）
   | 列 | 説明 |
   |----|------|
   | 改定内容 | 検索クエリ（全文） |
   | 正解ID一覧 | 期待される正解ID |
   | LLM強化クエリ | QueryEnhancerで生成されたクエリ |
   | 抽出キーワード | Sudachiで抽出したTop-5名詞 |
   | ベクトル重み | vector_weight（デフォルト0.9） |

   #### Azure側
   | 列 | 説明 |
   |----|------|
   | Azure_シナリオID | 検索ヒットID |
   | Azure_類似度 | ハイブリッドスコア（0.45以上） |
   | Azure_カテゴリ | Both / Original_Only / LLM_Enhanced_Only |
   | Azure_正解フラグ | TRUE / FALSE |
   | Azure_質問 | Search_Result_Q（全文） |
   | Azure_回答 | Search_Result_A（全文） |
   | Azure_関連性判定 | 関連あり / 要確認 / 関連なし |
   | Azure_判定根拠 | LLMの判定理由 |
   | Azure_修正案 | LLMの修正提案 |
   | Azure_ソース | 元ファイル名（rev01smile等） |

   #### VertexAI側
   | 列 | 説明 |
   |----|------|
   | VertexAI_シナリオID | 検索ヒットID |
   | VertexAI_類似度 | ハイブリッドスコア（0.45以上） |
   | VertexAI_カテゴリ | Both / Original_Only / LLM_Enhanced_Only |
   | VertexAI_正解フラグ | TRUE / FALSE |
   | VertexAI_質問 | Search_Result_Q（全文） |
   | VertexAI_回答 | Search_Result_A（全文） |
   | VertexAI_関連性判定 | 関連あり / 要確認 / 関連なし |
   | VertexAI_判定根拠 | LLMの判定理由 |
   | VertexAI_修正案 | LLMの修正提案 |
   | VertexAI_ソース | 元ファイル名（rev01smile等） |

   **注意**: Azure/VertexAIで候補数が異なる場合、多い方に合わせて少ない方は空欄

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
DBが存在しません: data/vector_db/rev01smile/azure_openai
```
→ `build_db.py --revisions-only`を先に実行してDBを構築。

### 正解IDが一致しない
- 正解IDのフォーマットを確認（`{ボット名}_{Excel行番号}`）
- シナリオファイルのメタデータ（row_index, source）を確認
- `generate_correct_ids.py`で正解ID対応表を再生成

### LLM分析が遅い
- `ENABLE_LLM_ANALYSIS=false`で無効化可能
- 各候補に対してLLM呼び出しが発生するため、候補数が多いと時間がかかる

---

## 関連ファイル

| ファイル | 説明 |
|---------|------|
| `scripts/build_db.py` | DB構築スクリプト（回答支援AI（類似回答検索）+ 改定別 統合） |
| `scripts/evaluate_revisions.py` | 評価スクリプト（多段階検索版） |
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
