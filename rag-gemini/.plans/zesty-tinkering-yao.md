# 評価システム再設計 - 列定義とアーキテクチャ

## 背景
- AzureとVertexAIは埋め込みモデルが異なり、類似度スケールも異なる
- 同じ土俵で比較するのは不適切 → **プロバイダー別に評価**
- **多段階検索**を使用（原文検索 + LLM強化検索）
- 閾値 **0.45以上** の候補を取得
- 精度計算は不要（ユーザーが手動判断）

---

## 出力ファイル構成

**改定別シート + Azure/VertexAI横並び比較**

```
output/revision_evaluation_YYYYMMDD_HHMMSS.xlsx
├── サマリー   # 各改定×プロバイダーの候補数・正解数一覧
├── ①         # 改定①の結果（Azure + VertexAI横並び）
├── ②
├── ③
├── ④
├── ⑤
└── ⑥
```

---

## 列定義

### サマリーシート

| 列 | 列名 | 説明 |
|----|------|------|
| A | 改定番号 | ①②③④⑤⑥ |
| B | 改定内容（先頭50文字） | クエリの概要 |
| C | 正解ID数 | 正解IDの総数 |
| D | Azure_候補数 | Azure閾値0.45以上の候補数 |
| E | Azure_正解一致数 | Azure候補のうち正解数 |
| F | VertexAI_候補数 | VertexAI閾値0.45以上の候補数 |
| G | VertexAI_正解一致数 | VertexAI候補のうち正解数 |

### 詳細シート（①～⑥）- Azure/VertexAI横並び

#### 共通列（検索条件）

| 列 | 列名 | 説明 |
|----|------|------|
| A | 改定内容 | 検索クエリ（全文） |
| B | 正解ID一覧 | 期待される正解ID |
| C | LLM強化クエリ | QueryEnhancerで生成されたクエリ |
| D | 抽出キーワード | Sudachiで抽出したTop-5名詞 |
| E | ベクトル重み | vector_weight（デフォルト0.9） |

#### Azure側

| 列 | 列名 | 説明 |
|----|------|------|
| F | Azure_シナリオID | 検索ヒットID |
| G | Azure_類似度 | ハイブリッドスコア（0.45以上） |
| H | Azure_カテゴリ | Both / Original_Only / LLM_Enhanced_Only |
| I | Azure_正解フラグ | TRUE / FALSE |
| J | Azure_質問 | Search_Result_Q（全文） |
| K | Azure_回答 | Search_Result_A（全文） |
| L | Azure_関連性判定 | 関連あり / 要確認 / 関連なし |
| M | Azure_判定根拠 | LLMの判定理由 |
| N | Azure_修正案 | LLMの修正提案 |
| O | Azure_ソース | 元ファイル名 |

#### VertexAI側

| 列 | 列名 | 説明 |
|----|------|------|
| P | VertexAI_シナリオID | 検索ヒットID |
| Q | VertexAI_類似度 | ハイブリッドスコア（0.45以上） |
| R | VertexAI_カテゴリ | Both / Original_Only / LLM_Enhanced_Only |
| S | VertexAI_正解フラグ | TRUE / FALSE |
| T | VertexAI_質問 | Search_Result_Q（全文） |
| U | VertexAI_回答 | Search_Result_A（全文） |
| V | VertexAI_関連性判定 | 関連あり / 要確認 / 関連なし |
| W | VertexAI_判定根拠 | LLMの判定理由 |
| X | VertexAI_修正案 | LLMの修正提案 |
| Y | VertexAI_ソース | 元ファイル名 |

**注意**: Azure/VertexAIで候補数が異なる場合、多い方に合わせて少ない方は空欄

---

## 多段階ハイブリッド検索の仕組み

### スコア計算（ハイブリッド）
```
combined_score = vector_weight × ベクトル類似度 + keyword_weight × キーワード類似度
                 (デフォルト0.9)                    (デフォルト0.1)
```

- **ベクトル類似度**: 埋め込みモデルによる意味的類似度
- **キーワード類似度**: Jaccard類似度（抽出キーワードの共通割合）

### 検索フロー
```
入力クエリ
    ↓
キーワード抽出（Sudachi: 名詞Top-5）
    ↓
┌─ Stage 1: 原文クエリでハイブリッド検索
│      ベクトル検索 + キーワード類似度
│
└─ Stage 2: LLM強化クエリでハイブリッド検索
       QueryEnhancer.enhance() でクエリ生成
       ベクトル検索 + キーワード類似度
    ↓
Stage 3: 結果をマージ＋カテゴリ分類
    - Both: 両方で見つかった（信頼度高）
    - Original_Only: Stage 1のみで見つかった
    - LLM_Enhanced_Only: Stage 2のみで見つかった
```

### 出力するクエリ情報
- **原文クエリ**: 入力の改定内容そのまま
- **LLM強化クエリ**: QueryEnhancerで生成（prompt/summarize_v1.0.txt使用）
- **抽出キーワード**: Sudachiで抽出したTop-5名詞

---

## 処理フロー

```
1. 入力ファイル読み込み (input/multi_stage_input.xlsx)
   - 番号, 改定内容, 正解ID

2. 各改定について：
   a. Azure多段階検索
      - MultiStageOrchestratorを使用
      - 閾値 ≥ 0.45 でフィルタ
      - カテゴリ分類を取得

   b. VertexAI多段階検索
      - 同上

   c. LLM分析（JudgmentSupport）
      - 各候補に対して関連性判定を実行
      - 判定根拠と修正案を取得

3. 結果を横並びでマージ
   - 行数 = max(Azure候補数, VertexAI候補数)
   - 少ない方は空欄で埋める

4. 結果をExcel出力
   - サマリーシート
   - ①～⑥シート（横並び詳細）
```

---

## 技術的考慮事項

1. **閾値フィルタリング**: 0.45以上
   - MultiStageOrchestratorで閾値設定可能か確認要

2. **質問と回答の分離**: Search_Result_Q, Search_Result_A
   - 現行の多段階検索結果に含まれている

3. **LLM分析**: JudgmentSupportクラスを使用
   - evaluate_batch()でバッチ処理
   - APIコストに注意

4. **横並びマージ**: itertools.zip_longestで行数を揃える

---

## 入力ファイル（変更なし）

**ファイル**: `input/multi_stage_input.xlsx`

| 列名 | 説明 |
|------|------|
| 番号 | 改定番号（①②③④⑤⑥） |
| 改定内容 | 検索クエリテキスト |
| 正解ID | 正解シナリオID（カンマ区切り） |

---

## 修正対象ファイル

- `scripts/evaluate_revisions.py`: 多段階検索、横並びシート構成、LLM分析追加

---

## 実装手順

1. **多段階検索の統合**
   - MultiStageOrchestratorを使用
   - 閾値0.45でフィルタリング
   - カテゴリ分類を取得

2. **LLM分析の統合**
   - JudgmentSupportクラスを使用
   - 各候補に対して評価実行

3. **Excel出力構成変更**
   - サマリーシート（Azure/VertexAI比較列）
   - ①～⑥シート（横並びレイアウト）
   - 質問と回答を別列

4. **横並びマージ処理**
   - Azure結果とVertexAI結果を横に並べる
   - 行数が異なる場合は空欄で埋める

---

## 検証方法

1. `python scripts/evaluate_revisions.py` を実行
2. 出力ファイル `output/revision_evaluation_*.xlsx` を確認
3. 各シートでAzure/VertexAIが横並びになっているか確認
4. 閾値0.45以上の候補のみ出力されているか確認
5. カテゴリ（Both/Original_Only/LLM_Enhanced_Only）が正しく付与されているか確認
6. LLM分析結果（関連性判定、判定根拠、修正案）が出力されているか確認
