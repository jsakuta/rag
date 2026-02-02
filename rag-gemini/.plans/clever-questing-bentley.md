# 変更内容列の追加（Phase 2）

## 概要

前回実装した「未発見シナリオ表示機能」に加えて：
1. 入力ファイルの「変更種別」列を「変更内容」列に置き換え
2. 各正解IDの具体的な変更内容を入力ファイルに記載
3. 出力に変更内容を表示

※ 「変更種別」列は廃止し、「変更内容」列のみとする

---

## 変更対象ファイル

| ファイル | 変更内容 |
|---------|---------|
| `input/multi_stage_input.xlsx` | 「変更種別」→「変更内容」に置換・データ入力 |
| `scripts/evaluate_revisions.py` | 変更種別→変更内容に変更 |

---

## 入力ファイル形式

### 現状（カンマ区切り形式）
| 番号 | 改定内容 | 正解ID |
|-----|---------|--------|
| ① | 個人事業主の場合でも... | smile-bot_130, smile-bot_186, smile-bot_187 |

### 変更後（行ごと形式）
| 番号 | 改定内容 | 正解ID | 変更内容 |
|-----|---------|--------|---------|
| ① | 個人事業主の場合でも... | smile-bot_130 | 【本人確認】の文言変更 |
| ① | 個人事業主の場合でも... | smile-bot_186 | 【本人確認】の文言変更 |
| ① | 個人事業主の場合でも... | smile-bot_187 | 「キャッシュカード暗証一致もしくは」を追加 |

※ 1行に1正解ID、改定番号でグループ化して処理

---

## 入力データ（台帳記載に基づく）

### ① 台帳No.20（スマイル機能変更）
| 正解ID | 変更内容 |
|--------|---------|
| smile-bot_130 | 【本人確認】の文言変更 |
| smile-bot_186 | 【本人確認】の文言変更 |
| smile-bot_187 | 「キャッシュカード暗証一致もしくは」を追加 |

### ② 台帳No.21（相続少額払い）
| 正解ID | 変更内容 |
|--------|---------|
| souzoku-bot_146 | 少額払いフローチャート画像変更 |
| souzoku-bot_149 | 設問新設/書類内容修正 |
| souzoku-bot_156 | 設問新設/書類内容修正 |
| souzoku-bot_157 | 設問新設/書類内容修正 |
| souzoku-bot_162 | 書類内容修正 |
| souzoku-bot_163 | 書類内容修正 |
| souzoku-bot_164 | 書類内容修正 |
| souzoku-bot_165 | 書類内容修正 |
| souzoku-bot_166 | 書類内容修正 |
| souzoku-bot_167 | 書類内容修正 |
| souzoku-bot_168 | 書類内容修正 |
| souzoku-bot_169 | 書類内容修正 |
| souzoku-bot_174 | 書類内容修正 |
| souzoku-bot_175 | 書類内容修正 |

### ③ 台帳No.25-30,35-36（保険証→資格確認証）
| 正解ID | 変更内容 |
|--------|---------|
| naibujimu-bot_96 | 画像01修正 |
| naibujimu-bot_99 | 健康保険被保険者証→資格確認書 |
| naibujimu-bot_138 | 健康保険証→資格確認書 |
| naibujimu-bot_141 | 健康保険証→資格確認書 |
| naibujimu-bot_642 | 画像02修正 |
| naibujimu-bot_707 | 問答文修正 |
| souzoku-bot_146 | 画像01.02修正 |
| souzoku-bot_167 | 保険証→資格確認書 |
| torikaku-bot_22 | 画像7修正 |
| torikaku-bot_25 | 画像9修正 |
| torikaku-bot_35 | 画像12修正 |
| torikaku-bot_88 | 保険証→資格確認書 |
| torikaku-bot_89 | 保険証→資格確認書 |
| torikaku-bot_90 | 保険証→資格確認書 |
| smile-bot_436 | 画像03内の文言修正 |
| smile-bot_366 | 「国民健康保険被保険者証、国民年金手帳等」削除 |

### ④ 台帳No.37（0円新規開設可能）
| 正解ID | 変更内容 |
|--------|---------|
| naibujimu-bot_641 | 「住宅ローン推進室等の場合」→「0円新規可能」 |

### ⑤ 台帳No.41-42（AML→GPLEX）
| 正解ID | 変更内容 |
|--------|---------|
| smile-bot_41 | AML検索→GPLEX |
| smile-bot_237 | AMLメニュー→GPLEX |
| smile-bot_268 | AMLメニュー→GPLEX |
| smile-bot_316 | AMLメニュー→GPLEX |

### ⑥ 台帳No.43-45（DC→MDC）
全32件すべて：変更内容=「DC→MDC」

---

## 実装計画

### 1. 入力ファイル（multi_stage_input.xlsx）の更新

xlsxスキルを使用して行ごと形式に変換：
1. カンマ区切りの正解IDを1行1IDに展開
2. 「変更種別」列を削除、「変更内容」列を追加
3. 各行に対応する変更内容を入力
4. **条件付き書式**: 同じ改定番号の2行目以降は「番号」「改定内容」列を灰色文字に

**例: ①の場合**
```
| ① | 個人事業主の場合でも... | smile-bot_130 | 【本人確認】の文言変更 |           ← 先頭行：通常表示
| ① | 個人事業主の場合でも... | smile-bot_186 | 【本人確認】の文言変更 |           ← 灰色文字
| ① | 個人事業主の場合でも... | smile-bot_187 | 「キャッシュカード暗証一致もしくは」を追加 | ← 灰色文字
```

**条件付き書式の設定**:
- 適用範囲: A列（番号）、B列（改定内容）
- 条件: `=A2=A1`（上の行と同じ番号の場合）
- 書式: フォント色をグレー（#808080）に

### 2. evaluate_revisions.py の更新

#### 2.1 load_input_data の変更
```python
def load_input_data(self) -> pd.DataFrame:
    df = pd.read_excel(INPUT_FILE)
    if "変更内容" not in df.columns:
        df["変更内容"] = ""
    return df
```

#### 2.2 evaluate_all_revisions の大幅変更
- 改定番号（番号列）でグループ化
- 各グループ内の正解IDリストと変更内容辞書を構築
- evaluate_revisionに渡す

```python
def evaluate_all_revisions(self) -> Dict[str, Dict[str, Any]]:
    input_df = self.load_input_data()

    # 改定番号でグループ化
    grouped = input_df.groupby("番号")

    for revision, group in grouped:
        revision_content = group.iloc[0]["改定内容"]
        correct_ids = group["正解ID"].tolist()

        # 正解IDと変更内容の辞書を構築
        change_details_map = {
            row["正解ID"]: row["変更内容"]
            for _, row in group.iterrows()
        }

        results = self.evaluate_revision(
            revision, revision_content, correct_ids, change_details_map
        )
```

#### 2.3 evaluate_revision の変更
- `correct_ids_with_types` → `change_details_map: Dict[str, str]` に変更
- 未発見シナリオの変更内容取得: `change_details_map.get(scenario_id, "")`

#### 2.4 _parse_correct_ids_with_types の削除
- 行ごと形式では不要（1行1ID）

#### 2.5 Excel出力
- 「変更種別」→「変更内容」にリネーム
- 列幅の調整

---

## 実装順序

1. **入力ファイルの更新**（xlsxスキル使用）
   - 行ごと形式に変換（全70件程度）
   - 変更内容を入力

2. **evaluate_revisions.py の更新**
   - load_input_data: 変更内容列対応
   - evaluate_all_revisions: グループ化処理に変更
   - evaluate_revision: change_details_map引数追加
   - _parse_correct_ids_with_types: 削除
   - unfound_scenarios: 変更内容取得方法変更
   - Excel出力: ヘッダー変更

3. **動作確認**
   ```bash
   python scripts/evaluate_revisions.py
   ```

---

## 検証方法

1. **入力ファイル確認**:
   - 行数が正しいこと（全正解ID数 = 70件程度）
   - 各行に正解ID、変更内容が正しく入力されていること

2. **スクリプト実行**:
   ```bash
   python scripts/evaluate_revisions.py
   ```

3. **出力Excel確認**:
   - 詳細シートの未発見シナリオに「変更内容」列が表示されること
   - 変更内容が正しく表示されること
   - サマリーシートは変更なし（未発見数・未発見IDのみ）
