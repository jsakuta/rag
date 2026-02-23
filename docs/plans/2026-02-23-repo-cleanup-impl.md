# リポジトリ全体クリーンアップ 実装計画

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** リポジトリ全体の重複ネスト構造・残存フォルダ・古い計画ファイルを整理し、見通しの良い構造にする

**Architecture:** 3フェーズで段階的にクリーンアップ。Phase 1は明らかなゴミ、Phase 2は残存フォルダ、Phase 3はドキュメント整理。各フェーズ完了時にコミット。

**Tech Stack:** Git, Bash（ファイル操作）

---

## Phase 1 — 明らかなゴミ掃除（リスクゼロ）

### Task 1: rag-local 内の重複ネスト構造を削除

**Files:**
- Delete: `rag-local/data/data/` （空の重複フォルダツリー: input/, output/, source/, vector_db/）
- Delete: `rag-local/data/output/output/` （空フォルダ）
- Delete: `rag-local/logs/logs/` （古いログ app.log 999KB + 空archive/）
- Delete: `rag-local/prompt/prompt/` （judgment_support.txt, summarize_v1.0.txt の重複コピー）

**Step 1: 削除前に各フォルダの中身を最終確認**

```bash
# 各フォルダが本当に空/重複か最終確認
ls -la rag-local/data/data/
ls -la rag-local/data/output/output/
ls -la rag-local/logs/logs/
ls -la rag-local/prompt/prompt/
```

Expected: data/data/ は空サブフォルダのみ、output/output/ は空、logs/logs/ は古いapp.log、prompt/prompt/ はルートと同じファイル

**Step 2: 重複ネストを削除**

```bash
rm -rf rag-local/data/data/
rm -rf rag-local/data/output/output/
rm -rf rag-local/logs/logs/
rm -rf rag-local/prompt/prompt/
```

**Step 3: 削除後の構造を確認**

```bash
ls rag-local/data/
ls rag-local/logs/
ls rag-local/prompt/
```

Expected: 各ディレクトリに重複ネストがないこと

### Task 2: ルートの nul ファイルと archive 内の .venv を削除

**Files:**
- Delete: `nul` （Windows予約名ファイル）
- Delete: `archive/rag-batch/.venv/`
- Delete: `archive/rag-reranker/.venv/`
- Delete: `archive/rag-streamlit/.venv/`

**Step 1: nul ファイルを削除**

Windows予約名のため、Git Bash から削除:
```bash
rm -f nul
```

**Step 2: archive 内の .venv を削除**

```bash
rm -rf archive/rag-batch/.venv/
rm -rf archive/rag-reranker/.venv/
rm -rf archive/rag-streamlit/.venv/
```

**Step 3: 削除確認**

```bash
ls -la nul 2>&1  # "No such file" が出ること
ls archive/rag-batch/ | grep venv  # 空であること
```

### Task 3: Phase 1 コミット

```bash
git add -A
git status
git commit -m "chore: Phase 1 — 重複ネスト構造・nul・archive .venv を削除

- rag-local/data/data/, data/output/output/, logs/logs/, prompt/prompt/ 削除
- Windows予約名 nul ファイル削除
- archive/rag-{batch,reranker,streamlit}/.venv/ 削除

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Phase 2 — 残存フォルダ整理

### Task 4: rag-gemini 残存フォルダを削除

**調査結果:** rag-gemini/ には `data/output/latest/`（空フォルダ）のみ残存。全データは rag-local に移行済み。

**Files:**
- Delete: `rag-gemini/` （フォルダ全体）

**Step 1: 最終確認**

```bash
find rag-gemini/ -type f 2>/dev/null
```

Expected: ファイルなし（空フォルダのみ）

**Step 2: 削除**

```bash
rm -rf rag-gemini/
```

### Task 5: rag-maintenance/.plans/ をアーカイブ整理

**調査結果:**
- 進行中（保持）: `検索UI改善_引き継ぎ.md`, `検索UI改善_実装計画.md`, `2026-02-18-bugfix-excel-save-folder.md`
- 実装済み/古い: その他15ファイル（ランダム名含む）

**Files:**
- Create: `rag-maintenance/.plans/archive/`
- Move: 実装済み・古い計画ファイル → `rag-maintenance/.plans/archive/`

**Step 1: archive ディレクトリ作成**

```bash
mkdir -p rag-maintenance/.plans/archive
```

**Step 2: 進行中ファイル以外を archive に移動**

```bash
cd rag-maintenance/.plans
# 進行中の3ファイルを残して他を全て archive へ
for f in *.md; do
  case "$f" in
    "検索UI改善_引き継ぎ.md"|"検索UI改善_実装計画.md"|"2026-02-18-bugfix-excel-save-folder.md")
      echo "KEEP: $f"
      ;;
    *)
      mv "$f" archive/
      echo "ARCHIVE: $f"
      ;;
  esac
done
cd ../..
```

**Step 3: 確認**

```bash
ls rag-maintenance/.plans/
ls rag-maintenance/.plans/archive/
```

Expected: ルートに3ファイル、archive に15ファイル

### Task 6: ルート .plans/ をアーカイブ整理

**調査結果:** 2ファイルとも実装済み（rag再編成計画）

**Files:**
- Move: `.plans/2026-02-17-rag-reorganization-design.md` → `.plans/archive/`
- Move: `.plans/2026-02-18-rag-reorganization-impl.md` → `.plans/archive/`

**Step 1: archive ディレクトリ作成・移動**

```bash
mkdir -p .plans/archive
mv .plans/2026-02-17-rag-reorganization-design.md .plans/archive/
mv .plans/2026-02-18-rag-reorganization-impl.md .plans/archive/
```

### Task 7: Phase 2 コミット

```bash
git add -A
git status
git commit -m "chore: Phase 2 — rag-gemini削除・.plans整理

- rag-gemini/ 残存フォルダ削除（空フォルダのみだった）
- rag-maintenance/.plans/ の実装済み計画を archive/ に移動
- ルート .plans/ の実装済み計画を archive/ に移動

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Phase 3 — ドキュメント整理

### Task 8: rag-local/docs/ のドキュメント確認と整理

**調査結果:**
- 保持: ARCHITECTURE.md, CONFIGURATION.md, SECURITY.md, TROUBLESHOOTING.md, REVISION_EVALUATION.md, GOOGLE_CLOUD_AUTH.md
- 要更新: API_REFERENCE.md（クラス名不一致）, PROMPTS.md（内容不足）
- revisions/ サブフォルダ: 事務改定説明7ファイル → 要確認

**Files:**
- Modify: `rag-local/docs/API_REFERENCE.md` （実装との不一致を修正）
- Modify: `rag-local/docs/PROMPTS.md` （プロンプトファイルとの対応を確認・更新）
- Check: `rag-local/docs/revisions/` （7ファイルの必要性を確認）

**Step 1: API_REFERENCE.md を実装に合わせて修正**

実装の実際のクラス名・関数シグネチャを確認し、ドキュメントを更新:
- `DataProcessor` → `Processor` に修正
- `SearchConfig` の記載を `config.py` の実装に合わせる

**Step 2: PROMPTS.md を確認・更新**

`prompt/` ディレクトリ内のファイルと対応を確認し、不足があれば追記。

**Step 3: revisions/ の確認**

```bash
ls rag-local/docs/revisions/
```

各ファイルの内容を確認し、不要なものがあれば削除。

### Task 9: Phase 3 コミット

```bash
git add -A
git status
git commit -m "docs: Phase 3 — API_REFERENCE.md・PROMPTS.md を実装に合わせて更新

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## 完了確認

### Task 10: 最終構造確認

**Step 1: リポジトリ全体のツリーを確認**

```bash
# トップレベル
ls -la
# rag-local の主要構造
ls rag-local/
ls rag-local/data/
ls rag-local/logs/
ls rag-local/prompt/
ls rag-local/docs/
```

**Expected 最終構造:**
```
rag/
├── .plans/
│   └── archive/          ← 実装済み計画
├── archive/              ← 旧プロジェクト（.venv削除済み）
├── docs/
│   └── plans/            ← 設計書
├── rag-local/            ← メインプロジェクト（ネスト解消済み）
│   ├── data/             ← data/data/ 解消
│   ├── logs/             ← logs/logs/ 解消
│   ├── prompt/           ← prompt/prompt/ 解消
│   └── docs/             ← 更新済み
├── rag-maintenance/      ← Teams Bot
│   └── .plans/
│       ├── (進行中3件)
│       └── archive/      ← 実装済み15件
└── README.md
```

**Step 2: git log で3フェーズのコミット確認**

```bash
git log --oneline -5
```
