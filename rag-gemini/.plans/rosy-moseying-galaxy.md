# 変更前シナリオDB構築計画（最終版）

## 現在の状況

### 完了済み
- [x] `scripts/prepare_before_scenario.py` 作成（不要列削除：文字数列、シナリオパス列）
- [x] 9つのシナリオファイルを `reference/scenario/` に配置済み
- [x] `HierarchicalExcelInputHandler` に `scenario_file` パラメータ追加
- [x] `_prepare_reference_data_for_vectorization()` に `latest_scenario` パラメータ追加
- [x] `ui/chat.py` に動的業務分野読み込み機能追加
- [x] `scripts/rebuild_before_scenario_db.py` 作成（再構築スクリプト）

### バグ修正済み
- [x] シナリオパス列が検索結果に表示される問題
- [x] 全DBが同じファイル（rev06smile）のデータで構築される問題
  - 原因: `MultiFolderInputHandler` が最新作成時刻のファイルを自動検出していた
  - 修正: 業務分野ごとに対応するファイルのみを読み込むよう変更

### 残りの作業
- [ ] 残りのDB再構築（rev05smile, rev06smile）
- [ ] 動作確認

## 残り作業の計画

### Step 1: 残りのDBを再構築

Streamlit UIを停止した状態で再構築スクリプトを実行:

```bash
python scripts/rebuild_before_scenario_db.py
```

または、残りの2つ（rev05smile, rev06smile）のみを手動で再構築:

```bash
# Pythonで直接実行
python -c "
from config import SearchConfig
from src.utils.dynamic_db_manager import DynamicDBManager

config = SearchConfig(base_dir='.')
with DynamicDBManager(config) as db_manager:
    areas = db_manager.analyze_reference_files()
    for area in ['rev05smile', 'rev06smile']:
        if area in areas:
            db_manager.update_business_db(area, areas[area])
"
```

### Step 2: 動作確認

1. Streamlit UIを起動:
   ```bash
   streamlit run ui/chat.py
   ```

2. サイドバーで業務分野を `rev02souzoku` に切り替え

3. 相続関連のキーワード（例：「相続」「遺産」）で検索

4. 結果がrev02souzokuのデータのみで構成されていることを確認
   - rev06smile のデータが混在していないことを確認

## 修正済みファイル一覧

| ファイル | 変更内容 |
|----------|----------|
| `src/handlers/input_handler.py` | `scenario_file` パラメータ追加 |
| `src/utils/dynamic_db_manager.py` | `latest_scenario` パラメータ対応 |
| `ui/chat.py` | 動的業務分野読み込み |
| `scripts/prepare_before_scenario.py` | シナリオパス列削除対応 |
| `scripts/rebuild_before_scenario_db.py` | 再構築スクリプト（新規） |

## DB構成（合計9つ）

| 改定 | ボット | ファイル名 | 状態 |
|------|--------|------------|------|
| ①スマイル機能変更 | smile | rev01smile_シナリオデータ_20260127.xlsx | 再構築済み |
| ②相続少額払い | souzoku | rev02souzoku_シナリオデータ_20260127.xlsx | 再構築済み |
| ③保険証→資格確認証 | smile | rev03smile_シナリオデータ_20260127.xlsx | 再構築済み |
| ③保険証→資格確認証 | naibujimu | rev03naibujimu_シナリオデータ_20260127.xlsx | 再構築済み |
| ③保険証→資格確認証 | souzoku | rev03souzoku_シナリオデータ_20260127.xlsx | 再構築済み |
| ③保険証→資格確認証 | torikaku | rev03torikaku_シナリオデータ_20260127.xlsx | 再構築済み |
| ④0円新規開設可能 | naibujimu | rev04naibujimu_シナリオデータ_20260127.xlsx | 再構築済み |
| ⑤AMLフィルター→GPLEX | smile | rev05smile_シナリオデータ_20260127.xlsx | **未完了** |
| ⑥DC→MDC | smile | rev06smile_シナリオデータ_20260127.xlsx | **未完了** |
