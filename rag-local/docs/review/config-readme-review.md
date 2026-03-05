# CONFIGURATION.md + README.md レビューレポート

**レビュー日**: 2026-03-03
**対象ファイル**: `docs/CONFIGURATION.md` (~531行), `README.md` (~328行)
**突き合わせ対象**: `config.py`, `config/settings.yaml`, `config/business_areas.yaml`, `.env.example`, `scripts/build_db.py`, `scripts/create_handover_package.py`, `requirements.txt`, `requirements-dev.txt`, `pytest.ini`

---

## Critical（事実誤認 / コードと不一致）

### C-01: CONFIGURATION.md — settings.yaml 欠損時の動作説明が不正確

**箇所**: `CONFIGURATION.md:290`
> settings.yaml は起動時に必須です。ファイルが存在しないか common セクションが欠落している場合、`RuntimeError` が発生します。

**実際のコード** (`config.py:25-28`):
```python
if not settings_path.exists():
    logger.warning(f"設定ファイルが見つかりません: {settings_path}")
    return {}
```

`load_settings()` はファイル不在時に `RuntimeError` ではなく空辞書 `{}` を返す。`RuntimeError` が発生するのは、その後のモジュールレベルコード (`config.py:62-63`) で `_common_settings` が空のときのみ。

**修正案**: 「settings.yaml が存在しない場合、`load_settings()` は空辞書を返し、その結果モジュールレベルで `RuntimeError: config/settings.yaml の読み込みに失敗しました（common セクション）` が発生します。」と正確に記述する。

---

### C-02: CONFIGURATION.md — `SearchConfig` のバリデーション記述にLLMプロバイダー制限の欠落

**箇所**: `CONFIGURATION.md:469-479` (設定検証セクション)

**文書の記述**:
```
# - embedding_provider が必須（未設定時 ValueError）
```

**実際のコード** (`config.py:264-267`):
```python
if not self.llm_provider:
    raise ValueError("DEFAULT_LLM_PROVIDER環境変数が設定されていません（gemini）")
if self.llm_provider != "gemini":
    raise ValueError(f"LLMプロバイダーは 'gemini' のみサポートしています（指定値: {self.llm_provider}）")
```

LLM プロバイダーが `gemini` **のみ**サポートという制約と、`llm_model` 必須チェック (`config.py:268-269`) が文書の検証リストから欠落している。

**修正案**: 設定検証セクションに以下を追加:
```
# - llm_provider が "gemini" であること（他の値は ValueError）
# - llm_model が必須（未設定時 ValueError）
```

---

### C-03: CONFIGURATION.md — `VECTOR_DB_BATCH_SIZE` のバリデーション範囲が未記載

**箇所**: `CONFIGURATION.md:474`

文書には `EMBEDDING_BATCH_SIZE が 1~250 の範囲` のみ記載。

**実際のコード** (`config.py:213-214`):
```python
if self.VECTOR_DB_BATCH_SIZE < 1 or self.VECTOR_DB_BATCH_SIZE > 1000:
    raise ValueError(...)
```

`VECTOR_DB_BATCH_SIZE` (1~1000) のバリデーションが未記載。また `VECTOR_SEARCH_MULTIPLIER >= 1`, `multi_stage_threshold` (0~1), `multi_stage_max_results >= 1` のバリデーションも漏れている。

---

### C-04: CONFIGURATION.md — `search_type` の設定値が未記載

**箇所**: CONFIGURATION.md 全体

`config.py:75-76` で `search_type` (hybrid | keyword_filter) が定義され、`settings.yaml` にも記載されているが、CONFIGURATION.md の「SearchConfig パラメータ」セクション (`CONFIGURATION.md:159-177`) に `search_type` フィールドの説明がない。

コード例 (`CONFIGURATION.md:166-176`) にも `search_type` が含まれていない。

**修正案**: SearchConfig パラメータのコード例に `search_type="hybrid"` を追加し、有効値 `hybrid | keyword_filter` を説明する。

---

### C-05: README.md — `create_handover_package.py` の出力例フラグ説明が不正確

**箇所**: `README.md:308-309`
```
# パッケージ作成（出力例を含む）
python scripts/create_handover_package.py ./handover_package --include-examples
```

**実際のコード** (`create_handover_package.py:69-76`): `--include-examples` は `data/output/latest/` ではなく `data/output/examples/` から種類ごとに最新1件を収集する（4カテゴリ: 回答支援×バッチ/UI, 運用保守×バッチ/UI）。

README の説明「出力例を含む」は曖昧。実際の動作（4種類から各1件、`data/output/examples/` から収集）を明記すべき。

---

### C-06: CONFIGURATION.md — `SearchConfig` のデフォルト値説明が不完全

**箇所**: `CONFIGURATION.md:163-177`

文書のコード例:
```python
config = SearchConfig(
    top_k=4,
    vector_weight=0.9,
    search_mode="original",
    reference_type="multi_folder",
)
```

**実際のコード** (`config.py:71-72`):
- `DEFAULT_TOP_K` = `_batch_settings["top_k"]` → settings.yaml の `batch.top_k = 4` からロード
- `DEFAULT_VECTOR_WEIGHT` = `_batch_settings["vector_weight"]` → settings.yaml の `batch.vector_weight = 0.9` からロード

デフォルト値がハードコードではなく **settings.yaml から動的にロード**されることが未説明。文書は `4` や `0.9` をリテラルとして記載しているが、実際には settings.yaml を変更すればデフォルト値が変わる。

**修正案**: 「デフォルト値は `config/settings.yaml` の `batch` セクションから読み込まれます」と注記を追加。

---

## Important（情報不足 / 文書化されていない機能・設定）

### I-01: CONFIGURATION.md — `search_source` のUI動的切替の記述場所が不適切

**箇所**: `CONFIGURATION.md:297-311`

`search_source` セクションにUI vs バッチでの挙動が記載されているが、CONFIGURATION.md は設定リファレンスのため、UIの動的挙動の詳細は ANSWER_SUPPORT.md に委ねるべき。現在の記述は冗長ではないが、UIでの初期値としての利用とバッチでの固定利用を明記するのは有用。

**指摘なし（情報提供のみ）**。

---

### I-02: CONFIGURATION.md — `SearchConfig` の未記載フィールドが多数

**箇所**: `CONFIGURATION.md:159-177`

以下の `SearchConfig` フィールドが CONFIGURATION.md に未記載:

| フィールド | デフォルト値 | 用途 |
|-----------|------------|------|
| `MULTI_STAGE_THRESHOLD` | 0.45 | 多段階検索の統合スコア閾値 |
| `MULTI_STAGE_MAX_RESULTS` | 100 | 各検索の最大結果数 |
| `EMBEDDING_BATCH_SIZE` | 250 | 埋め込みAPIバッチサイズ |
| `VECTOR_DB_BATCH_SIZE` | 100 | ChromaDB書き込みバッチサイズ |
| `VECTOR_SEARCH_MULTIPLIER` | 2 | top_k 取得倍率 |
| `force_db_update` | False | 強制DB更新フラグ |
| `include_hierarchy_in_vector` | True | 階層情報のベクトル化包含 |
| `dual_provider_mode` | False | 両プロバイダー比較モード |

これらはチューニングや運用で変更する可能性がある値。少なくとも上位4件は記載推奨。

---

### I-03: README.md — Python バージョン要件の不正確さ

**箇所**: `README.md:35`
> Python 3.9 以上が必要です

**実際**: `config.py:53` で `list[str]` 型ヒント（Python 3.9+）、`build_db.py:53` で `list[str]`、`create_handover_package.py:98` で `list[tuple[Path, Path]]` を使用。これらは Python 3.9+ で動作する。

ただし `requirements.txt` の `chromadb>=1.0.15` が Python 3.9 をサポートしているか未確認。chromadb 1.x は通常 Python 3.9+ を要求するが、最新版では 3.10+ を要求する可能性がある。

**修正案**: 実際に動作確認済みの Python バージョン（例: 3.11）を明記するか、`python --version` で確認済みのバージョンを記載する。

---

### I-04: README.md — テスト実行方法の欠如

**箇所**: README.md 全体

README にテスト実行コマンドの記載がない。`pytest.ini` が存在し、`requirements-dev.txt` に pytest 関連パッケージが定義されているが、以下の情報が欠落:

```bash
# 開発用依存関係のインストール
pip install -r requirements-dev.txt

# テスト実行
pytest

# カバレッジ付き
pytest --cov=src
```

**修正案**: セットアップ手順の末尾またはトラブルシューティングセクションにテスト実行方法を追加。

---

### I-05: README.md — `requirements-dev.txt` の依存関係説明の欠如

**箇所**: README.md 全体

`requirements-dev.txt` の存在は「同梱するもの」(`README.md:296`) で言及されているが、中身（pytest, pytest-cov, pytest-mock, pytest-asyncio, faker）の説明やインストール手順がない。

---

### I-06: CONFIGURATION.md — `evaluation` セクションの詳細が不足

**箇所**: `CONFIGURATION.md:288`

`evaluation` セクションは REVISION_OPS.md への参照リンクのみ。しかし `settings.yaml` の `evaluation` セクション (`settings.yaml:116-244`) には以下の重要な設定がある:

- `max_results`, `filter_mode`, `top_k`, `thresholds`, `enable_judgment_support`
- `revision_areas` (改定番号とDBエリアのマッピング)
- `area_to_bot`, `area_to_category`
- `revision_source_files`

これらは CONFIGURATION.md で概要レベルの記載があるべき（詳細は REVISION_OPS.md に委ねるとしても）。

---

### I-07: CONFIGURATION.md — `business_areas.yaml` の `スマイルタブレット: smile_tablet` マッピングの用途不明

**箇所**: `CONFIGURATION.md:352`

CONFIGURATION.md では `mappings` に「スマイル→smile, 内部事務→naibujimu 等」と記載。

**実際の `business_areas.yaml:9`**: `スマイルタブレット: smile_tablet` というマッピングも存在するが、通常コレクション名は `smile` のみで `smile_tablet` は使用されていない（回答支援AIは `smile` コレクションのみ）。

この追加マッピングの用途が不明であり、引き継ぎ者が混乱する可能性がある。

---

### I-08: README.md — ディレクトリツリーの正確性

**箇所**: `README.md:168-259`

ディレクトリツリーに `ui/` ディレクトリが含まれているが説明が不足:
```
├── ui/
│   └── shared.py                 # 共通UI部品
```

実際のディレクトリには `__init__.py` と `__pycache__/` も存在し、`.streamlit/config.toml` も存在する。ツリー図に `.streamlit/` が欠落。

**修正案**: ツリー図に `.streamlit/config.toml` を追加:
```
├── .streamlit/
│   └── config.toml              # Streamlit設定
```

---

## Minor（文体・明瞭性・体裁の改善）

### M-01: CONFIGURATION.md — `keyword_weight` の自動計算の重複説明

**箇所**: `CONFIGURATION.md:169` と `CONFIGURATION.md:221`

`keyword_weight` が `1.0 - vector_weight` で自動計算される旨が2箇所で説明されている（SearchConfig コード例のコメント + 重み調整セクション）。1箇所に統一するか、片方を他方への参照にする。

---

### M-02: CONFIGURATION.md — ログファイルセクションの未実装機能

**箇所**: `CONFIGURATION.md:410-412`
```
├── error.log        # エラーログ（将来実装予定）
└── access.log       # アクセスログ（将来実装予定）
```

「将来実装予定」の機能をリファレンスに記載すると、引き継ぎ者が混乱する可能性。既存の `app.log` のみ記載し、将来の計画はコメントアウトするか削除推奨。

---

### M-03: README.md — `CLAUDE.md` が「引き継ぎ対象外」の理由が不明

**箇所**: `README.md:288`

| `CLAUDE.md` | 開発メモ | 引き継ぎ対象外 |

CLAUDE.md がプロジェクトの重要な技術知識（コード構成、環境変数、変更前シナリオDBフロー等）を含んでおり、引き継ぎ対象外とする理由の説明がない。「Claude Code専用の開発メモであり、ドキュメントとしての信頼性は docs/ に集約」等の注記があると良い。

---

### M-04: CONFIGURATION.md — 手動検証コマンドのインポートパス

**箇所**: `CONFIGURATION.md:488`
```python
python -c "from config import SearchConfig; config = SearchConfig(); print(config)"
```

このコマンドは `rag-local/` ディレクトリで実行する必要があるが、その前提が明記されていない。また `.env` の読み込みがないため `SearchConfig()` の初期化で環境変数エラーが発生する可能性が高い。

**修正案**:
```python
cd rag-local
python -c "from dotenv import load_dotenv; load_dotenv(); from config import SearchConfig; config = SearchConfig(); print(config)"
```

---

### M-05: README.md — セットアップ Step 4 のタイトルが不正確

**箇所**: `README.md:97`

Step 4 は「ソースデータの配置」だが、Step 3 の環境変数テーブルに `DEFAULT_EMBEDDING_PROVIDER` が含まれており、Step 2 の説明文で「Step 4 で設定する `DEFAULT_EMBEDDING_PROVIDER`」と参照している。しかし Step 4 はデータ配置であり、`DEFAULT_EMBEDDING_PROVIDER` の設定は Step 3。

参照先の不整合。Step 2 の記述を「Step 3 で設定する」に修正すべき。

---

### M-06: CONFIGURATION.md — 環境変数テーブルの Required/Optional 分類の曖昧さ

**箇所**: `CONFIGURATION.md:22-78`

「必須環境変数」セクション (`CONFIGURATION.md:22`) に `DEFAULT_LLM_PROVIDER`, `DEFAULT_LLM_MODEL`, `DEFAULT_EMBEDDING_PROVIDER` の3つが記載。
しかし `config.py` の `__post_init__()` では以下も事実上必須:
- `GEMINI_PROJECT_ID` — LLM (Gemini) 使用時に必要（全環境で必須）
- `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` — `azure_openai` プロバイダー使用時は必須

現在これらは「Google Cloud / VertexAI 設定」「Azure OpenAI 設定」セクションに分かれて記載されているが、Required/Optional の条件付き分類が不十分。

**修正案**: 条件付き必須を明記する表記を導入（例: 「必須（`azure_openai` 使用時）」）。

---

## 引き継ぎ適性の評価

### 良い点
- README のセットアップ手順 (Step 1-6) は論理的な順序で記載
- CONFIGURATION.md は環境変数一覧が充実しており、`.env.example` とほぼ一致
- `build_db.py` のフラグは README の説明と完全一致
- `business_areas.yaml` のマッピング説明は正確
- 「同梱しないもの / するもの」の表は引き継ぎに有用

### 改善が必要な点
1. **テスト実行方法が未記載** (I-04): 引き継ぎ者がコードの正しさを検証できない
2. **settings.yaml からのデフォルト値ロードが未説明** (C-06): 設定変更時に混乱する
3. **`search_type` の説明欠落** (C-04): hybrid vs keyword_filter の切替方法が不明
4. **手動検証コマンドが動作しない可能性** (M-04): `.env` の load_dotenv() が欠如
5. **Python バージョンの曖昧さ** (I-03): 3.9 で本当に動作するか未確認

### 総合評価

README.md + CONFIGURATION.md の2文書で「環境構築からアプリケーション起動まで」はカバーできる。ただし、Critical 6件の事実誤認を修正し、Important の「テスト実行方法」と「settings.yaml デフォルト値の動的ロード」を追記しないと、引き継ぎ者がつまずく可能性が高い。

---

## 指摘サマリ

| 優先度 | 件数 | 主な内容 |
|--------|------|---------|
| Critical | 6件 | settings.yaml 動作説明不正確、LLMプロバイダー制限欠落、バリデーション範囲未記載、search_type 欠落、handover フラグ説明不正確、デフォルト値ロード未説明 |
| Important | 8件 | SearchConfig 未記載フィールド多数、テスト実行方法欠如、Python バージョン曖昧、evaluation セクション不足、ディレクトリツリー不正確 |
| Minor | 6件 | 重複説明、未実装機能記載、インポートパス、Step 参照不整合 |
| **合計** | **20件** | |
