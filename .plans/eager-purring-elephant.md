# 回答支援AI DB構築スクリプト + ドキュメント化 実装計画

## Context

業務分野を naibujimu/smile に再編したが、ベクトルDBが未構築。UIは既存DBディレクトリからのみ業務分野を取得するため、DB構築が先に必要。既存の `rebuild_before_scenario_db.py`（改定別DB用）と同様のスタンドアロンスクリプトを作成し、引き継ぎ用ドキュメントも整備する。

## Task 1: `scripts/build_answer_support_db.py` 作成

**Files:**
- Create: `rag-local/scripts/build_answer_support_db.py`

`rebuild_before_scenario_db.py` と同じパターンで、回答支援AI用DB（naibujimu, smile）を構築するスクリプト。

**処理内容:**
1. 対象業務分野の既存DBを削除（オプション `--force` で強制再構築）
2. タイムスタンプファイルから対象エントリを削除
3. `analyze_reference_files(include_revisions=False)` で参照ファイルを検出
4. 各業務分野に対して `update_business_db()` でDB構築
5. 構築結果のサマリ出力（業務分野、ドキュメント数、所要時間）

**参考:** `rebuild_before_scenario_db.py`（行100-130のパターンを踏襲）

**スキップロジック（デフォルト動作）:**
- `DynamicDBManager.needs_update()` が内部で判定
- DB存在 + ドキュメント数 > 0 + 参照ファイル未更新 → **スキップ（APIコスト発生なし）**
- DB未存在 or 参照ファイル更新あり → 構築/更新実行
- `--force` 指定時のみ既存DB削除→全再構築

**実行例:**
```bash
python scripts/build_answer_support_db.py              # 差分のみ構築（未構築or更新ありのみ）
python scripts/build_answer_support_db.py --force       # 既存DB削除して全再構築
python scripts/build_answer_support_db.py --business naibujimu  # 指定業務分野のみ
```

**注意:**
- `rebuild_before_scenario_db.py` は両プロバイダー（azure_openai + vertex_ai）で構築するが、回答支援AI用は現在 azure_openai のみでよい（環境変数 DEFAULT_EMBEDDING_PROVIDER に従う）
- Streamlit停止が前提（ChromaDBロック防止）

## Task 2: DB構築手順ドキュメント作成

**Files:**
- Create: `rag-local/docs/DB_BUILD_GUIDE.md`
- Modify: `rag-local/CLAUDE.md`（ドキュメント一覧にリンク追加）

**内容:**
- 業務分野構造の説明（naibujimu = 預金+総則FAQ + naibujimu-botシナリオ、smile = スマイルFAQ + smile-botシナリオ）
- 参照データの配置場所と命名規則（`{業務名}_{履歴データ|シナリオデータ}_{YYYYMMDD}.xlsx`）
- DB構築コマンド（`scripts/build_answer_support_db.py`）
- 改定別DB再構築（`scripts/rebuild_before_scenario_db.py`）
- ディレクトリ構造（data/source/, data/vector_db/）
- トラブルシューティング（ChromaDBロック、API認証エラー等）

## Task 3: DB構築実行 + 動作確認

`scripts/build_answer_support_db.py` を実行し、naibujimu と smile のDBを構築。

## 検証方法

1. `python scripts/build_answer_support_db.py` → naibujimu, smile のDB構築成功
2. `data/vector_db/naibujimu/azure_openai/chroma.sqlite3` が存在
3. `data/vector_db/smile/azure_openai/chroma.sqlite3` が存在（再構築）
4. Streamlit UI起動 → 業務分野に naibujimu, smile が表示
5. 検索実行 → 結果が返る
