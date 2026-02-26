# DB構築スクリプト統合設計

## 概要

`build_answer_support_db.py`（通常業務用）と `rebuild_before_scenario_db.py`（改定別用）を `build_db.py` に統合する。

### 背景

- 現在2つのスクリプトが存在し、役割が分断されている
- `rebuild_before_scenario_db.py` は毎回全削除→全再構築でスキップロジックがない
- 検証用途だけでなく、本番展開の参考として通常業務+改定別を一本で構築したい
- `DynamicDBManager.analyze_reference_files()` が `scenarios/latest/` しかスキャンせず、`scenarios/revisions/` の rev* ファイルを検出できない問題がある

### 決定事項

- **アプローチ**: DynamicDBManager を拡張して `scenarios/revisions/` もスキャン（アプローチA）
- **スクリプト名**: `build_db.py`
- **廃止**: `build_answer_support_db.py`, `rebuild_before_scenario_db.py`

---

## 変更対象ファイル

| ファイル | 変更種別 | 内容 |
|---|---|---|
| `scripts/build_db.py` | 新規作成 | 統合DB構築スクリプト |
| `scripts/build_answer_support_db.py` | 削除 | build_db.py に統合 |
| `scripts/rebuild_before_scenario_db.py` | 削除 | build_db.py に統合 |
| `src/utils/dynamic_db_manager.py` | 修正 | revisions/ スキャン対応 |
| `docs/DB_BUILD_GUIDE.md` | 更新 | 新スクリプトの使用方法 |

---

## 設計詳細

### 1. DynamicDBManager の変更

#### `__init__` の変更

```python
# 既存
self.reference_scenario_path = os.path.join(config.base_dir, "data", "source", "scenarios", "latest")

# 追加
self.reference_revision_scenario_path = os.path.join(config.base_dir, "data", "source", "scenarios", "revisions")
```

#### `analyze_reference_files()` の変更

```python
def analyze_reference_files(self, include_revisions: bool = True):
    # 1. latest/ のFAQ + シナリオをスキャン（現行通り）
    #    → naibujimu, smile 等の通常業務分野を検出

    # 2. include_revisions=True の場合のみ revisions/ もスキャン
    #    → rev01_smile, rev02_souzoku 等を business_areas に追加

    # 3. include_revisions=False なら revisions/ はスキャンしない
    #    → 回答支援AI（chat.py）からの呼び出しは影響なし
```

#### `_prepare_reference_data_for_vectorization()` への影響

rev* 開始の業務分野名の場合は `reference_revision_scenario_path` を参照する分岐が必要。

```python
def _get_scenario_base_path(self, business_area: str) -> str:
    """業務分野名に応じたシナリオベースパスを返す"""
    if business_area.startswith("rev"):
        return self.reference_revision_scenario_path
    return self.reference_scenario_path
```

### 2. build_db.py（統合スクリプト）

#### CLI インターフェース

```
python scripts/build_db.py                          # 全業務分野、差分のみ構築
python scripts/build_db.py --force                   # 全業務分野、全再構築
python scripts/build_db.py --business naibujimu      # 特定業務分野のみ
python scripts/build_db.py --revisions-only          # 改定別（rev*）のみ
python scripts/build_db.py --no-revisions            # 通常業務のみ
```

#### 処理フロー

```
1. 引数パース（--force, --business, --revisions-only, --no-revisions）
2. for provider in [azure_openai, vertex_ai]:
     3. SearchConfig 初期化（embedding_provider を provider に設定）
     4. DynamicDBManager 初期化
     5. analyze_reference_files(include_revisions=...) で業務分野検出
     6. フィルタ適用:
        - --business X → 指定分野のみ
        - --revisions-only → rev* のみ
        - --no-revisions → rev* を除外
     7. --force の場合:
        - 対象DBディレクトリを shutil.rmtree で削除
        - タイムスタンプファイルから対象エントリを削除
     8. for area in target_areas:
          update_business_db(area, files)
          ← 内部で needs_update() による差分検知
          ← DB最新なら「DBは最新です」でスキップ
     9. サマリ出力（業務分野×プロバイダーの結果一覧）
```

#### build_answer_support_db.py からの主な差分

| 項目 | build_answer_support_db.py | build_db.py |
|---|---|---|
| プロバイダー | 環境変数の1つ | azure_openai + vertex_ai 両方 |
| 対象 | 通常業務のみ (include_revisions=False) | 全業務分野 (include_revisions=True) |
| フィルタ | --business のみ | --business, --revisions-only, --no-revisions |
| スキップ | あり（needs_update） | あり（needs_update） |
| --force | あり | あり |

### 3. 既存コードへの影響

#### 回答支援AI（chat.py）

`include_revisions=False` で `analyze_reference_files()` を呼んでいるため影響なし。
revisions/ のスキャンは `include_revisions=True` の場合のみ実行される。

#### 事務改定評価AI（evaluate_revisions.py）

現在 `DynamicDBManager` を直接使用。`analyze_reference_files()` の戻り値に rev* が含まれるようになるが、
既存のフィルタロジックで問題なし。

#### テスト

`tests/unit/` の既存テスト（63件）への影響を確認する必要がある。
特に `analyze_reference_files()` のモックを使用しているテストがあれば更新。

---

## データ構造（参考）

### ソースファイル

```
data/source/
├── scenarios/
│   ├── latest/           ← 通常業務
│   │   ├── 内部事務_シナリオデータ_20260224.xlsx
│   │   └── スマイル_シナリオデータ_20260224.xlsx
│   └── revisions/        ← 改定別
│       ├── rev01_smile_シナリオデータ_20260203.xlsx
│       ├── rev02_souzoku_シナリオデータ_20260203.xlsx
│       └── ... (9ファイル)
└── faq/
    └── latest/           ← 通常業務のみ（改定別FAQはなし）
        ├── スマイル_履歴データ_20250205.xlsx
        └── 内部事務_履歴データ_20260224.xlsx
```

### vector_db 構造

```
data/vector_db/
├── {業務分野}/{プロバイダー}/chroma.sqlite3
│   例: smile/azure_openai/chroma.sqlite3
│   例: rev01_smile/vertex_ai/chroma.sqlite3
└── update_timestamps.json
```
