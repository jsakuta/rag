# 引き継ぎ用リポジトリ再編成 — 設計ドキュメント

> 作成日: 2026-02-17
> ステータス: 承認待ち

---

## 1. 背景・目的

本プロジェクトを別チームのエンジニア（プロジェクト固有文脈なし）に引き継ぐにあたり、
リポジトリ構造を整理して**初見でも理解可能な状態**にする。

### 現状の問題

- `rag-gemini` に回答支援AI（main.py + chat.py）と事務改定評価（evaluate_revisions.py）が混在
- 2つのAIの区別が名前から読み取れない
- 非推奨の `rag-batch`, `rag-streamlit`, `rag-reranker` がトップレベルに残存
- 引き継ぎ先が「どのファイルから着手すべきか」判断できない

### 方針

**案3: リネーム + 内部 apps/ 分離（物理リポジトリ分割なし）**を採用。

理由: 2つのAIは `src/`（5,157行）+ `config.py`（288行）の共有コアに強依存しており、
物理分割はコード重複 or symlink運用を強いるため不採用。

---

## 2. AI分類

| AI | エントリポイント | 配置先 | 技術 |
|----|-----------------|--------|------|
| 回答支援AI（バッチ + UI） | `main.py`, `ui/chat.py` | `rag-local/apps/answer-support/` | Python, ChromaDB, Gemini |
| 事務改定評価AI | `scripts/evaluate_revisions.py` | `rag-local/apps/revision-eval/` | Python, ChromaDB, Azure OpenAI |
| 運用効率化AI（Teams Bot） | `maintenance-bot/src/` | `rag-maintenance/`（変更なし） | TypeScript, M365 Agents SDK |

---

## 3. 最終ディレクトリ構造

```
rag/
├── rag-local/                          ← 旧 rag-gemini（git mv でリネーム）
│   ├── apps/
│   │   ├── answer-support/
│   │   │   ├── main.py                ← 旧 rag-gemini/main.py
│   │   │   └── ui/
│   │   │       ├── __init__.py
│   │   │       └── chat.py            ← 旧 rag-gemini/ui/chat.py
│   │   └── revision-eval/
│   │       └── evaluate_revisions.py   ← 旧 rag-gemini/scripts/evaluate_revisions.py
│   ├── src/                            ← 共有コア（変更なし）
│   │   ├── core/
│   │   │   ├── processor.py            (257行)
│   │   │   ├── searcher.py             (647行)
│   │   │   ├── judgment_support.py     (120行)
│   │   │   └── search/                 (1,309行)
│   │   ├── handlers/
│   │   │   ├── input_handler.py        (609行)
│   │   │   └── output_handler.py       (430行)
│   │   ├── types/
│   │   │   └── search_types.py         (280行)
│   │   └── utils/
│   │       ├── auth.py                 (175行)
│   │       ├── azure_embedding.py      (209行)
│   │       ├── dynamic_db_manager.py   (1,043行)
│   │       ├── logger.py               (330行)
│   │       ├── vector_db.py            (277行)
│   │       └── ...                     (他 4ファイル)
│   ├── scripts/                        ← ユーティリティ（残り 8本）
│   │   ├── archive_output_files.py
│   │   ├── check_db_content.py
│   │   ├── generate_correct_ids.py
│   │   ├── generate_db_source.py
│   │   ├── prepare_before_scenario.py
│   │   ├── rebuild_before_scenario_db.py
│   │   ├── rebuild_faq_db.py
│   │   └── remove_empty_dirs.py
│   ├── config.py                       (288行 — 変更なし)
│   ├── config/
│   │   ├── business_areas.yaml
│   │   └── settings.yaml
│   ├── data/
│   ├── prompt/
│   ├── tests/
│   ├── requirements.txt
│   └── README.md                       ← 新規作成（rag-local 用）
│
├── rag-maintenance/                    ← 変更なし
│   ├── maintenance-bot/
│   │   ├── src/
│   │   ├── m365agents.yml
│   │   └── package.json
│   ├── docs/
│   ├── scripts/
│   └── README.md
│
├── archive/                            ← 新規（git mv で移動）
│   ├── rag-batch/
│   ├── rag-streamlit/
│   └── rag-reranker/
│
├── docs/
│   └── plans/
│       └── 2026-02-17-rag-reorganization-design.md  ← 本文書
│
└── README.md                           ← 既存（引き継ぎ資料、更新対象）
```

---

## 4. 共有コード依存関係

### 4.1 依存マップ

```
apps/answer-support/main.py
  └── config.SearchConfig, src.core.processor.Processor,
      src.utils.dynamic_db_manager.DynamicDBManager, src.utils.logger

apps/answer-support/ui/chat.py
  └── config.SearchConfig/load_settings, src.core.processor.Processor,
      src.utils.dynamic_db_manager.DynamicDBManager, src.utils.logger

apps/revision-eval/evaluate_revisions.py
  └── config.SearchConfig/load_settings, src.core.judgment_support,
      src.core.search.* (5モジュール), src.types.search_types,
      src.utils.auth, src.utils.logger, src.utils.vector_db
```

### 4.2 共有コア規模

| ディレクトリ | 行数 | 主要モジュール |
|-------------|------|---------------|
| `src/core/` | 1,024行 | processor, searcher, judgment_support |
| `src/core/search/` | 1,309行 | 検索エンジン群（6モジュール） |
| `src/handlers/` | 1,039行 | input_handler, output_handler |
| `src/types/` | 322行 | search_types |
| `src/utils/` | 2,463行 | auth, dynamic_db_manager, logger 等 |
| **合計** | **6,157行** | |

### 4.3 分離不可の理由

- 3つのエントリポイント全てが `config.py` + `src/` に依存
- `evaluate_revisions.py` は `src/core/search/` の5モジュールを直接import
- `config.py` は `config/settings.yaml` をロードし、全エントリポイントで共有
- 物理分割すると **6,000行超の重複** or **symlink/サブモジュール運用** が必要

→ **apps/ による論理分離 + README補完** が最適解

---

## 5. import修正計画

移動するファイルは3つ。各ファイルの `sys.path` と `PROJECT_ROOT` を修正する。

### 5.1 `apps/answer-support/main.py`

**現状** (rag-gemini/main.py:127):
```python
config = SearchConfig(base_dir=os.path.dirname(os.path.abspath(__file__)))
```

**修正後**:
```python
import sys
from pathlib import Path

# apps/answer-support/ → rag-local/ への2階層上を追加
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ...
config = SearchConfig(base_dir=str(PROJECT_ROOT))
```

### 5.2 `apps/answer-support/ui/chat.py`

**現状** (rag-gemini/ui/chat.py:4,52):
```python
sys.path.insert(0, str(Path(__file__).parent.parent))      # ui/ → rag-gemini/
PROJECT_ROOT = Path(__file__).parent.parent                  # ui/ → rag-gemini/
```

**修正後**:
```python
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))  # ui/ → answer-support/ → apps/ → rag-local/
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent             # 同上
```

### 5.3 `apps/revision-eval/evaluate_revisions.py`

**現状** (rag-gemini/scripts/evaluate_revisions.py:27-28):
```python
PROJECT_ROOT = Path(__file__).parent.parent      # scripts/ → rag-gemini/
sys.path.insert(0, str(PROJECT_ROOT))
```

**修正後**:
```python
PROJECT_ROOT = Path(__file__).parent.parent.parent  # revision-eval/ → apps/ → rag-local/
sys.path.insert(0, str(PROJECT_ROOT))
```

---

## 6. 作業ステップ

### Step 1: ブランチ作成
```bash
git checkout -b refactor/rag-reorganization
```

### Step 2: rag-gemini → rag-local リネーム
```bash
git mv rag-gemini rag-local
```

### Step 3: apps/ ディレクトリ作成 + ファイル移動
```bash
mkdir -p rag-local/apps/answer-support/ui
mkdir -p rag-local/apps/revision-eval

git mv rag-local/main.py rag-local/apps/answer-support/main.py
git mv rag-local/ui/__init__.py rag-local/apps/answer-support/ui/__init__.py
git mv rag-local/ui/chat.py rag-local/apps/answer-support/ui/chat.py
git mv rag-local/scripts/evaluate_revisions.py rag-local/apps/revision-eval/evaluate_revisions.py
```

### Step 4: 空になった旧ディレクトリ削除
```bash
rmdir rag-local/ui  # git mv 後に空になった場合
```

### Step 5: import修正（3ファイル）
- セクション5の修正を適用

### Step 6: アーカイブ移動
```bash
mkdir -p archive
git mv rag-batch archive/rag-batch
git mv rag-streamlit archive/rag-streamlit
git mv rag-reranker archive/rag-reranker
```

### Step 7: README作成・更新
- `rag-local/README.md` — 新規作成（apps/の説明、実行方法、共有コア構造）
- `README.md`（ルート） — 既存の引き継ぎ資料を新構造に合わせて更新

### Step 8: 動作確認
```bash
cd rag-local
python apps/answer-support/main.py --help    # import エラーなし確認
python apps/revision-eval/evaluate_revisions.py --help  # 同上
```

### Step 9: 不要ファイル整理
- `DEPRECATED.md`（ルート）— archive に内容を統合後、削除
- `nul` ファイル（rag-gemini/, rag-maintenance/ 内の Windows予約語ファイル）— 可能なら削除

---

## 7. リスク・注意事項

| リスク | 対策 |
|--------|------|
| git mv 後のgit履歴追跡 | `git log --follow` で確認。リネーム検出閾値内に収まるよう、ファイル内容変更は最小限に |
| import パス間違い | Step 8 の動作確認で検証。`--help` レベルでimportチェーン全体が走る |
| .env パス参照 | `config.py` の `load_dotenv()` はカレントディレクトリの `.env` を読む → apps/ から実行時は `PROJECT_ROOT/.env` を明示的にロードする必要あり |
| venv 内の絶対パス | `rag-gemini/venv/` → `rag-local/venv/` でパスが壊れる可能性 → venv再作成を推奨 |
| Windows `nul` ファイル | `del \\?\C:\VSCode\rag\rag-gemini\nul` 等の特殊コマンドが必要 |

---

## 8. 成果物一覧

| 成果物 | 種別 |
|--------|------|
| `rag-local/apps/answer-support/` | ファイル移動 + import修正 |
| `rag-local/apps/revision-eval/` | ファイル移動 + import修正 |
| `rag-local/README.md` | 新規作成 |
| `archive/` | ディレクトリ移動 |
| `README.md`（ルート） | 更新 |
| 本設計ドキュメント | 新規作成 |
