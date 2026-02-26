# 引き継ぎ前一括クリーンアップ 実装計画

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** rag-local の引き継ぎ前に旧参照・不要ファイル・キャッシュを徹底整理する。

**Architecture:** 2コミット（ドキュメント修正 + ファイル削除）+ git追跡外キャッシュ清掃。

**Tech Stack:** Markdown, Git, Bash

**設計書:** `docs/plans/2026-02-26-handoff-cleanup-design.md`

---

### Task 1: ドキュメントの deposit/general 旧参照を修正

**Files:**
- Modify: `rag-local/docs/CONFIGURATION.md:293-322`
- Modify: `rag-local/docs/TROUBLESHOOTING.md:82,200-207,419-430`
- Modify: `rag-local/CLAUDE.md` (vector_db ディレクトリ一覧)

**注意:** `docs/API_REFERENCE.md` の deposit 参照は `business_area_translator.py` の TRANSLATION_MAP のドキュメントであり、実コードに対応する正確な記述のため変更しない。

**Step 1: `docs/CONFIGURATION.md` — コレクション命名規則テーブルを更新**

lines 293-305 を以下に置換:

```markdown
### コレクション命名規則

| 日本語 | コレクション名 | 備考 |
|--------|----------------|------|
| 内部事務 | naibujimu | 預金+総則を統合 |
| スマイル | smile | |
| rev01_smile | rev01_smile | 改定別（事務改定評価AI用） |
| rev02_souzoku | rev02_souzoku | 改定別 |
| ... | rev{XX}_{bot} | 改定番号_ボット名 |
```

**Step 2: `docs/CONFIGURATION.md` — タイムスタンプ JSON サンプルを更新**

lines 311-322 を以下に置換:

```markdown
**ファイル:** `data/vector_db/update_timestamps.json`

```json
{
  "naibujimu_azure_openai_faq": 1735567200.0,
  "naibujimu_azure_openai_scenario": 1735567200.0,
  "smile_azure_openai_faq": 1735567200.0,
  "smile_azure_openai_scenario": 1735567200.0
}
```
```

**Step 3: `docs/TROUBLESHOOTING.md` — エラー例を更新**

line 82: `deposit_DB` → `naibujimu`

```
ValueError: Collection 'naibujimu' does not exist.
```

**Step 4: `docs/TROUBLESHOOTING.md` — COLLECTION_NAME_MAP 例を更新**

lines 200-207 を以下に置換:

```python
# 業務分野名は自動的に英語変換されます
# 例: "預金" → "deposit" → naibujimu に統合済み
# 現在の主要コレクション: naibujimu, smile, rev{XX}_{bot}
```

**Step 5: `docs/TROUBLESHOOTING.md` — check_db_content.py 出力例を更新**

lines 421-430 を以下に置換:

```text
=== ChromaDB Content Analysis ===
Collection: naibujimu
Total documents: 11439
Unique documents: 11439
Duplicate documents: 0

Source distribution:
  scenario: 1384
  faq_data: 10055
```

**Step 6: `rag-local/CLAUDE.md` — vector_db ディレクトリ一覧を更新**

`general/` と `deposit/` を削除し、現在の構成に合わせる:

```
│   ├── vector_db/                # ベクトルDB
│   │   ├── update_timestamps.json
│   │   ├── naibujimu/            # 内部事務DB
│   │   ├── smile/                # スマイルDB
│   │   ├── rev01_smile/          # 改定DB
│   │   ├── rev02_souzoku/
│   │   └── ... (rev03-06)
```

**Step 7: コミット**

```bash
git add rag-local/docs/CONFIGURATION.md rag-local/docs/TROUBLESHOOTING.md rag-local/CLAUDE.md
git commit -m "docs: deposit/general の旧参照を naibujimu/smile に更新"
```

---

### Task 2: 不要スクリプト・データファイル・空ディレクトリを削除

**Files:**
- Delete: `rag-local/scripts/generate_db_source.py`
- Delete: `rag-local/scripts/remove_empty_dirs.py`
- Delete: `rag-local/scripts/archive_output_files.py`
- Delete: `rag-local/data/source/faq/archive/総則_履歴データ_20250829.xlsx`
- Delete: `rag-local/data/source/faq/archive/預金_履歴データ_20250830.xlsx`
- Delete: `rag-local/data/source/scenarios/individual/` (空ディレクトリ)

**Step 1: 不要スクリプトを削除**

```bash
cd /c/VSCode/rag
git rm rag-local/scripts/generate_db_source.py
git rm rag-local/scripts/remove_empty_dirs.py
git rm rag-local/scripts/archive_output_files.py
```

**Step 2: 旧業務分野の FAQ アーカイブを削除**

```bash
git rm "rag-local/data/source/faq/archive/総則_履歴データ_20250829.xlsx"
git rm "rag-local/data/source/faq/archive/預金_履歴データ_20250830.xlsx"
```

**Step 3: 空ディレクトリを削除**

```bash
rmdir rag-local/data/source/scenarios/individual
```

Git は空ディレクトリを追跡しないため、rmdir のみでOK。

**Step 4: CLAUDE.md のスクリプト一覧を更新**

`rag-local/CLAUDE.md` の scripts/ ディレクトリ一覧から削除した3スクリプトを除去し、残りの4スクリプトのみにする:

```
├── scripts/                      # ユーティリティスクリプト
│   ├── build_db.py                    # DB構築（回答支援AI + 改定別 統合）
│   ├── generate_correct_ids.py        # 正解ID生成
│   ├── prepare_before_scenario.py     # データ前処理
│   └── check_db_content.py            # DB内容確認
```

**Step 5: 残存参照チェック**

```bash
grep -r "generate_db_source\|remove_empty_dirs\|archive_output_files" rag-local/ --include="*.md" --include="*.py" -l
```

Expected: 0件（plan ファイルは除外）

**Step 6: コミット**

```bash
git add -A rag-local/scripts/ rag-local/data/source/faq/archive/ rag-local/CLAUDE.md
git commit -m "chore: 不要スクリプト3本と旧データファイルを削除"
```

---

### Task 3: キャッシュ・一時ファイル清掃

**Files:** git 追跡外のみ（コミット不要）

**Step 1: Python キャッシュを全削除**

```bash
cd /c/VSCode/rag/rag-local
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
```

**Step 2: テスト・ツールキャッシュを削除**

```bash
rm -rf .pytest_cache
rm -rf .serena
```

**Step 3: ログをトランケート**

```bash
: > logs/app.log
```

**Step 4: 確認**

```bash
# キャッシュが消えたことを確認
find . -name __pycache__ -type d 2>/dev/null | wc -l
# Expected: 0

# git status に変化なし（全て .gitignore 済み）
git status
```
