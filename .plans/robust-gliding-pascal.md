# バッチ処理動作確認計画（引き継ぎ前検証）

## Context
引き継ぎ前に、回答支援AIと運用保守効率化AI（事務改定評価AI）のバッチ処理が正しく動作するか確認する。
既存DBは構築済み（`data/vector_db/` に naibujimu, smile, rev01〜rev06 が存在）なので、DB再構築は不要。

## 確認対象バッチ処理

| # | バッチ | スクリプト | 入力 | 出力 |
|---|--------|-----------|------|------|
| 1 | DB内容確認 | `scripts/check_db_content.py` | 既存DB | コンソール出力 |
| 2 | 回答支援AIバッチ | `apps/answer-support/main.py --limit 3` | `data/input/naibujimu_20250829.xlsx` | `data/output/` 配下Excel |
| 3 | データ前処理（ドライラン） | `scripts/prepare_before_scenario.py --dry-run` | `reference/変更前シナリオ/` | コンソール出力 |
| 4 | 改定評価バッチ | `apps/revision-ops/run_eval.py --provider azure` | `data/input/multi_stage_input.xlsx` | `data/output/latest/rev/rev_eval_batch_*.xlsx` |

## 実行手順

### Step 1: 環境確認
```bash
cd /c/VSCode/rag/rag-local
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print('EMBEDDING:', os.getenv('DEFAULT_EMBEDDING_PROVIDER')); print('LLM:', os.getenv('DEFAULT_LLM_PROVIDER'))"
```
- `.env` の環境変数が正しく読み込まれるか確認
- Azure OpenAI / Vertex AI の認証情報が設定されているか

### Step 2: DB内容確認（`check_db_content.py`）
```bash
cd /c/VSCode/rag/rag-local
python scripts/check_db_content.py
```
- 各コレクションのドキュメント数、重複有無を確認
- 期待値: naibujimu, smile + rev01〜rev06 の各DBにデータが存在

### Step 3: 回答支援AIバッチ（`main.py --limit 3`）
```bash
cd /c/VSCode/rag/rag-local
python apps/answer-support/main.py --limit 3
```
- `data/input/naibujimu_20250829.xlsx` から先頭3件のみ処理
- DB更新（差分チェック）→ 検索実行 → Excel出力のフローを確認
- **確認ポイント**:
  - DB更新が正常に完了（既に最新ならスキップされるはず）
  - 検索結果が返ってくる（空でないこと）
  - `data/output/` に結果Excelが出力される

### Step 4: データ前処理ドライラン（`prepare_before_scenario.py`）
```bash
cd /c/VSCode/rag/rag-local
python scripts/prepare_before_scenario.py --dry-run
```
- 変更前シナリオファイルの検出・変換ルールが正しいか確認
- 実際のファイル変換は行わない（ドライランのみ）
- **確認ポイント**:
  - ①〜⑥の改定番号がrev01〜rev06に正しくマッピングされる
  - 出力ファイル名が `revXX_{bot}_シナリオデータ_{YYYYMMDD}.xlsx` 形式

### Step 5: 改定評価バッチ（`run_eval.py`）
```bash
cd /c/VSCode/rag/rag-local
python apps/revision-ops/run_eval.py --provider azure
```
- `multi_stage_input.xlsx` を入力に改定①〜⑥の影響候補を検索
- Azureプロバイダーのみで実行（時間短縮のため）
- **確認ポイント**:
  - DB存在確認テーブルが全てOK
  - 各改定番号で検索結果が返る
  - `data/output/latest/rev/rev_eval_batch_*.xlsx` が出力される
  - 正解発見率が表示される

## 判定基準

各ステップで以下を確認:
1. **エラーなく完了する**（exitコード 0）
2. **出力ファイルが生成される**（該当する場合）
3. **検索結果が妥当**（空でない、スコアが付いている）

## 注意事項
- Step 3 の `main.py` はDB更新を自動実行するが、既存DBが最新なら差分なしでスキップされる
- Step 5 は API 呼び出しを伴うため、Azure OpenAI のクォータに注意
- `--provider azure` で片側のみ実行することで時間とコストを節約
- LLM判定は `ENABLE_LLM_ANALYSIS=false`（デフォルト）のためスキップされる

## 主要ファイルパス
- `rag-local/apps/answer-support/main.py` — 回答支援AIバッチ
- `rag-local/apps/revision-ops/run_eval.py` — 改定評価バッチ
- `rag-local/scripts/check_db_content.py` — DB確認
- `rag-local/scripts/prepare_before_scenario.py` — データ前処理
- `rag-local/config/settings.yaml` — 設定ファイル
- `rag-local/src/core/processor.py` — バッチ処理エンジン
