# 変更前シナリオDB構築計画（修正版）

## 概要
6つの事務改定（①～⑥）の変更前シナリオをベクトルDBに格納し、改定・ボット別に切り替えて検索できるようにする。

**方針**: 既存のベクトル化フローを活用し、データ前処理スクリプトのみ新規作成

## DB構成（合計9つ）

既存コードは `reference/scenario/` 直下のファイルを検出し、ファイル名から業務分野を抽出する。
そのため、**サブフォルダではなく、リネームしたファイルを直下に配置**する。

| 改定 | ボット | 出力ファイル名 |
|------|--------|----------------|
| ①スマイル機能変更 | smile | rev01smile_シナリオデータ_20250127.xlsx |
| ②相続少額払い | souzoku | rev02souzoku_シナリオデータ_20250127.xlsx |
| ③保険証→資格確認証 | smile | rev03smile_シナリオデータ_20250127.xlsx |
| ③保険証→資格確認証 | naibujimu | rev03naibujimu_シナリオデータ_20250127.xlsx |
| ③保険証→資格確認証 | souzoku | rev03souzoku_シナリオデータ_20250127.xlsx |
| ③保険証→資格確認証 | torikaku | rev03torikaku_シナリオデータ_20250127.xlsx |
| ④0円新規開設可能 | naibujimu | rev04naibujimu_シナリオデータ_20250127.xlsx |
| ⑤AMLフィルター→GPLEX | smile | rev05smile_シナリオデータ_20250127.xlsx |
| ⑥DC→MDC | smile | rev06smile_シナリオデータ_20250127.xlsx |

→ 既存コードが `rev01smile`, `rev02souzoku` などを別々の業務分野として認識し、独立したDBを自動生成

## 実装ステップ

### Step 0: 前回実装の削除（クリーンアップ）

#### 削除対象ファイル
- `src/utils/before_scenario_db_manager.py`
- `scripts/generate_before_scenario_db.py`
- `scripts/generate_before_scenarios.py`

#### 変更をリバート
- `config.py` → git checkout
- `src/core/searcher.py` → git checkout
- `ui/chat.py` → git checkout

```bash
# 実行コマンド
git checkout config.py src/core/searcher.py ui/chat.py
rm src/utils/before_scenario_db_manager.py
rm scripts/generate_before_scenario_db.py
rm scripts/generate_before_scenarios.py
```

### Step 1: データ前処理スクリプト作成（新規）
ファイル: [scripts/prepare_before_scenario.py](scripts/prepare_before_scenario.py)

処理内容：
1. 「データ整理/変更前シナリオ」からExcelファイルを読み込み
2. **文字数列を削除**（Lv1, Lv2, ... Lv10, シナリオパスのみ残す）
3. **ファイル名を変換**して `reference/scenario/` 直下に配置
   - 元: `①変更前シナリオ_smile-bot.xlsx`
   - 後: `rev01smile_シナリオデータ_20250127.xlsx`

```bash
# 実行例
python scripts/prepare_before_scenario.py
```

### Step 2: 既存のベクトル化フローを使用
- 既存の `DynamicDBManager` と `HierarchicalExcelInputHandler` をそのまま活用
- `reference/scenario/` に配置したファイルを既存フローでベクトル化

### Step 3: UI確認（変更不要）

既存UIのサイドバー「業務分野」選択で、自動的に `rev01smile`, `rev03naibujimu` などが選択肢として表示される。
→ コード変更は不要

## 対象ファイル

### 新規作成
- [scripts/prepare_before_scenario.py](scripts/prepare_before_scenario.py) - データ前処理スクリプト

### 変更不要（既存を活用）
- config.py
- src/utils/dynamic_db_manager.py
- src/core/searcher.py
- src/handlers/input_handler.py
- ui/chat.py

## 検証方法

1. 前処理スクリプト実行
   ```bash
   python scripts/prepare_before_scenario.py
   ```

2. 配置確認
   ```bash
   ls reference/scenario/
   # rev01smile_シナリオデータ_20250127.xlsx などが存在することを確認
   ```

3. 既存のベクトル化フローでDB生成（自動検出される）
   ```bash
   python main.py --business-filter rev01smile
   ```

4. UI起動してテスト（業務分野として rev01smile, rev03naibujimu などが選択可能に）
   ```bash
   streamlit run ui/chat.py
   ```
