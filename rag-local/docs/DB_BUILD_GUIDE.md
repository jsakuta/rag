# DB構築ガイド

## 概要

RAG-Local のベクトルDBは **ChromaDB** で構成され、業務分野ごとに独立したDBを持ちます。
本ガイドでは、回答支援AI（類似回答検索）用DBと改定別DB（運用保守効率化AI（改定影響調査）用）の構築手順を説明します。

---

## 業務分野構造

### 回答支援AI（類似回答検索）用（通常業務）

| 業務分野 | DB名 | 内容 |
|---------|------|------|
| 内部事務 | `naibujimu` | 預金+総則FAQ + naibujimu-botシナリオ |
| スマイル | `smile` | スマイルFAQ + smile-botシナリオ |

### 改定別（運用保守効率化AI（改定影響調査）専用）

| DB名 | 改定内容 |
|------|---------|
| `rev01_smile` | ①スマイル機能変更 |
| `rev02_souzoku` | ②相続少額払い |
| `rev03_smile` | ③保険証→資格確認証（スマイル） |
| `rev03_naibujimu` | ③保険証→資格確認証（内部事務） |
| `rev03_souzoku` | ③保険証→資格確認証（相続） |
| `rev03_torikaku` | ③保険証→資格確認証（取引時確認） |
| `rev04_naibujimu` | ④0円新規開設可能 |
| `rev05_smile` | ⑤AML→GPLEX |
| `rev06_smile` | ⑥DC→MDC |

---

## 参照データの配置

### ディレクトリ構造

```
data/source/
├── faq/latest/             # FAQ（履歴データ）
│   ├── 内部事務_履歴データ_20260224.xlsx
│   └── スマイル_履歴データ_20250205.xlsx
├── scenarios/
│   ├── latest/             # 最新シナリオ（回答支援AI（類似回答検索）用）
│   │   ├── 内部事務_シナリオデータ_20260224.xlsx
│   │   └── スマイル_シナリオデータ_20260224.xlsx
│   └── revisions/          # 変更前シナリオ（改定別DB用）
│       ├── rev01smile_シナリオデータ_20250731.xlsx
│       └── ...
```

### ファイル命名規則

```
{業務名}_履歴データ_{YYYYMMDD}.xlsx     # FAQ
{業務名}_シナリオデータ_{YYYYMMDD}.xlsx  # シナリオ
```

- `{業務名}`: 日本語名（スマイル、内部事務等）。`config/business_areas.yaml` のマッピングで英語DB名に自動変換
- `{YYYYMMDD}`: データ日付。同一業務分野に複数ファイルがある場合、最新日付のファイルが使用される

---

## DB構築コマンド

統合スクリプト `build_db.py` で回答支援AI（類似回答検索）用DB・改定別DBの両方を構築できます。

### 回答支援AI（類似回答検索）用DB

```bash
cd rag-local

# 差分のみ構築（未構築 or 参照ファイル更新時のみ実行）
python scripts/build_db.py

# 既存DB削除して全再構築
python scripts/build_db.py --force

# 指定業務分野のみ
python scripts/build_db.py --business naibujimu
python scripts/build_db.py --business smile

# 回答支援AI（類似回答検索）用DBのみ（改定別を除外）
python scripts/build_db.py --no-revisions
```

**スキップロジック（デフォルト動作）:**
- DB存在 + ドキュメント数 > 0 + 参照ファイル未更新 → スキップ（APIコスト発生なし）
- DB未存在 or 参照ファイル更新あり → 構築/更新実行
- `--force` 指定時のみ既存DB削除→全再構築

### 改定別DB（運用保守効率化AI（改定影響調査）用）

```bash
# 改定別DBのみ構築（Azure OpenAI + VertexAI 両方）
python scripts/build_db.py --revisions-only

# 全DB一括構築（回答支援AI（類似回答検索）用 + 改定別）
python scripts/build_db.py --force
```

---

## DB出力先

```
data/vector_db/
├── update_timestamps.json          # 更新日時記録
├── naibujimu/                      # 回答支援AI（類似回答検索）: 内部事務
│   └── azure_openai/
│       └── chroma.sqlite3
├── smile/                          # 回答支援AI（類似回答検索）: スマイル
│   └── azure_openai/
│       └── chroma.sqlite3
├── rev01_smile/                    # 改定別
│   ├── azure_openai/
│   │   └── chroma.sqlite3
│   └── vertex_ai/
│       └── chroma.sqlite3
└── ...
```

---

## 前提条件

1. **Streamlit UIの停止**: ChromaDBはファイルロックを使用するため、UIと同時にスクリプトを実行するとロックエラーが発生します
2. **環境変数の設定**: `.env` に以下が必要
   - `DEFAULT_EMBEDDING_PROVIDER` / `DEFAULT_EMBEDDING_MODEL`
   - `AZURE_OPENAI_ENDPOINT` / `AZURE_OPENAI_API_KEY` / `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`
   - `DEFAULT_LLM_PROVIDER` / `DEFAULT_LLM_MODEL`
3. **参照データの配置**: `data/source/faq/latest/` と `data/source/scenarios/latest/` に対象Excelファイルが必要

---

## トラブルシューティング

### ChromaDBロックエラー

```
sqlite3.OperationalError: database is locked
```

**原因**: Streamlit UIまたは別のプロセスがDBファイルをロック中
**対処**: Streamlit UI を停止（`Ctrl+C`）してからスクリプトを再実行

### API認証エラー

```
openai.AuthenticationError: ...
```

**対処**: `.env` の `AZURE_OPENAI_API_KEY` と `AZURE_OPENAI_ENDPOINT` を確認

### 業務分野が検出されない

**対処**:
1. `data/source/faq/latest/` と `data/source/scenarios/latest/` にファイルが存在するか確認
2. ファイル名が `{業務名}_履歴データ_{YYYYMMDD}.xlsx` / `{業務名}_シナリオデータ_{YYYYMMDD}.xlsx` の形式か確認
3. `config/business_areas.yaml` に業務名のマッピングが登録されているか確認

### DB構築後にUIで業務分野が表示されない

**対処**: `data/vector_db/{業務名}/azure_openai/chroma.sqlite3` が存在するか確認。存在しない場合は `--force` で再構築
