# 引き継ぎ前リポジトリ整理 — 修正計画 v3

## Context

引き継ぎ（3/3, 3/10）に向けて、リポジトリの不要ファイル・古い参照・ドキュメント不整合を解消する。
**前提**: rag-local/ と rag-maintenance/ を**別々にフォルダコピーで渡す**。rag直下のファイルは引き継がれない。

3つのレビューエージェント + 1つのレビューエージェントで網羅的に調査・検証済み。
**コミット粒度**: 全ステップを **1コミット** にまとめる。

### ユーザーフィードバック（v2→v3）
- ルートREADME の「Phase1」表記は**変えない**（そのまま残す）
- venv → .venv は**実際にフォルダ名を変える**（忘れないこと）
- rag-local/README.md を**完璧にする**

---

## Step 0: 未コミット変更の処理

現在4ファイルに未コミット変更あり:
- `.gitignore` — `.keyword_cache/` 追加（Step 1-4 と統合）
- `.plans/eager-purring-elephant.md` — Step 1-2 で `git rm -f` で削除
- `rag-local/apps/revision-eval/ui/eval_ui.py` — 未コミットのコード変更
- `rag-local/src/core/search/keyword_search_engine.py` — 未コミットのコード変更

**対応**: eval_ui.py と keyword_search_engine.py の変更内容を確認し、整理コミットに含めるか判断する。

---

## Step 1: Git追跡からの削除（不要ファイル）

### 1-1. Docker関連（2ファイル + ローカル1ファイル）
```bash
git rm rag-local/Dockerfile
git rm docs/DOCKER.md
# rag-local/.dockerignore はGit未追跡 → ローカル手動削除（Step 6）
# archive内Dockerfileは Step 1-3 で一括対応
```

### 1-2. 開発メタデータ（10ファイル）
```bash
git rm .codex/context/files.md
git rm .codex/context/task.md
git rm -r .review/
git rm -f .plans/eager-purring-elephant.md   # -f: 未コミット変更があるため
```
- `.plans/archive/` (2ファイル) は**残す**（ユーザー指定）

### 1-3. archive/ をGit追跡から除外（60ファイル）
```bash
git rm -r --cached archive/
# ⚠ --cached 必須！省略するとローカルファイルも消える
```

### 1-4. .gitignore 更新
既存の `.keyword_cache/` 追加（未コミット分）と統合して以下を追加:
```gitignore
# 旧プロジェクトアーカイブ（引き継ぎ対象外）
archive/

# 開発メタデータ
.codex/
.review/

# ベンチマーク
.benchmarks/
```

---

## Step 2: rag-local/README.md を完璧にする

対象: `rag-local/README.md`

### 2-1. 「旧名: rag-gemini」削除
行3 `> 旧名: rag-gemini` → 行ごと削除（引き継ぎ先で混乱の元）

### 2-2. venv → .venv に統一
- 行20: `python -m venv venv` → `python -m venv .venv`
- 行21: `venv\Scripts\activate` → `.venv\Scripts\activate`
- 行22: `# source venv/bin/activate` → `# source .venv/bin/activate`

### 2-3. ドキュメントテーブルに欠落ファイルを追加（行154-163）
現在のテーブルに以下が欠落:
- `docs/DB_BUILD_GUIDE.md` — DB構築ガイド
- `docs/PROMPTS.md` — プロンプト詳細
- `docs/API_REFERENCE.md` — API仕様

追加後:
```markdown
| [docs/API_REFERENCE.md](./docs/API_REFERENCE.md) | API仕様 |
| [docs/DB_BUILD_GUIDE.md](./docs/DB_BUILD_GUIDE.md) | DB構築ガイド |
| [docs/PROMPTS.md](./docs/PROMPTS.md) | プロンプト詳細 |
```

---

## Step 3: rag-local/CLAUDE.md 修正

対象: `rag-local/CLAUDE.md`

### 3-1. ディレクトリ構造からDockerfile行を削除
行193 `├── Dockerfile                    # Docker設定` → 削除

---

## Step 4: ルート README.md 修正

対象: `README.md`（ルート）

### 4-0. 最終更新日付の更新
行3: `> 最終更新: 2026-02-17` → `> 最終更新: 2026-03-02`

### 4-1. 目次の更新（行10-17）
「アーカイブ対象」「アーカイブ実施手順」の目次行を削除。
**注意**: Phase1 の表記はそのまま残す。

### 4-2. プロジェクト全体マップの更新（行23-40）
```markdown
rag/
├── rag-maintenance/     [現行] Phase2 PoC 本番 — Teams Bot (TypeScript)
├── rag-local/           [現行] Phase1 ローカル検証・評価基盤 (Python)
│   ├── apps/
│   │   ├── answer-support/   回答支援AI（バッチ + Streamlit UI）
│   │   └── revision-eval/    事務改定評価AI（バッチ + 評価UI）
│   ├── src/              共有コア（検索エンジン、DB管理等）
│   ├── config/           設定ファイル
│   ├── scripts/          ユーティリティスクリプト
│   └── docs/             ドキュメント
├── archive/              [Git管理外] 旧版プロジェクト群（ローカル参照用）
└── docs/                 共通ドキュメント（SECURITY.md, TROUBLESHOOTING.md）
```
テーブルも同様に更新。`(旧rag-gemini)` の記述を削除。

### 4-3. 「rag-local」セクションの書き換え（行155-203）

**注意**: 見出しの「Phase1」は変えない。

**「3つの機能」→「2つのAIアプリケーション」に書き換え**:

```markdown
### 2つのAIアプリケーション

#### 回答支援AI（`apps/answer-support/`）

FAQ/シナリオを対象にハイブリッド検索（ベクトル+キーワード）を実行。

| モード | コマンド |
|--------|---------|
| バッチ（Excel入出力） | `python apps/answer-support/main.py` |
| Streamlit UI | `python apps/answer-support/main.py interactive` |

#### 事務改定評価AI（`apps/revision-eval/`）

改定内容→変更対象シナリオを Azure/VertexAI 両方で検索し、正解IDとのマッチ率を評価。

| モード | コマンド |
|--------|---------|
| バッチ（Excel出力） | `python apps/revision-eval/evaluate_revisions.py` |
| 評価UI（Streamlit） | `streamlit run apps/revision-eval/ui/eval_ui.py` |
```

旧パスの修正:
- `main.py batch` → `apps/answer-support/main.py`（引数なしがバッチモード）
- `ui/chat.py` → `apps/answer-support/ui/chat.py`
- `scripts/evaluate_revisions.py` → `apps/revision-eval/evaluate_revisions.py`

### 4-4. 「未完了タスク」セクション更新（行137-142）

完了状況（MEMORY.mdから確認済み）:
- ~~F5デバッグ動作確認~~ → 2026-02-25検証で完了
- ~~Excel出力E2Eテスト~~ → 2026-02-25検証で完了（BUG-2解消、Excel全量出力確認）
- **Toolkit再Deploy** → 未完了（コミット済みだがデプロイ待ち）
- **手順書完成** → 未完了

書き換え:
```markdown
### 残存タスク（rag-maintenance）

1. **Toolkit再Deploy**: コミット済み修正のデプロイ実施が必要
2. **手順書完成**: スクリーンショット追加、最終Word化
```

### 4-5. 「アーカイブ対象」「アーカイブ実施手順」セクション削除（行216-350）

archive/ はGit管理外になるため、これらのセクションは丸ごと削除。
「進化の系譜」セクション（行44-65）は歴史的参考として残す。

### 4-6. ディレクトリ構造図の `DOCKER.md` 削除
行348: `│   ├── DOCKER.md` → 削除

### 4-7. 環境構築リンク集の更新（行285-296）
- `rag-local README` のパスは正しい（`rag-local/README.md`）
- Docker関連リンクがあれば削除

---

## Step 5: 対応しない項目（理由付き）

| 項目 | 理由 |
|------|------|
| rag-maintenance/CLAUDE.md 日付更新 | rag-maintenanceのコード変更なし。引き継ぎ時に別途更新 |
| rag-local/CLAUDE.md の `../README.md` リンク | CLAUDE.mdはClaude Code専用。引き継ぎ先がClaude Code使わないなら実害なし |
| rag-local/README.md のクイックスタート環境変数不足 | 「前提条件・環境構築」セクション（行16-35）に.env設定手順が既にある |
| ルートREADME「進化の系譜」の技術名称精査 | 歴史記録として概要レベルで十分。引き継がれないファイル |

---

## Step 6: venv → .venv リネーム（コミット前に実施）

⚠ **venv は mv では動作しない**（内部パスがハードコードされている）。再作成が必要。

```bash
cd /c/VSCode/rag/rag-local
rm -rf venv
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
```

Git未追跡（.gitignore済み）なのでコミットには影響しないが、README の `.venv` 表記と実態を一致させるため実施。

---

## Step 7: ローカルのみの手動クリーンアップ（引き継ぎコピー前に実施）

- `rag-local/.dockerignore` 削除
- `rag-local/gemini_credentials.json` は渡さない（Git未追跡だが存在）
- `rag-local/.env` は渡さない（Git未追跡）
- `rag-local/__pycache__/` は渡さない
- `rag-local/.pytest_cache/` は渡さない

---

## 修正対象ファイル一覧

| ファイル | 操作 |
|---------|------|
| `rag-local/Dockerfile` | git rm |
| `docs/DOCKER.md` | git rm |
| `.codex/context/files.md` | git rm |
| `.codex/context/task.md` | git rm |
| `.review/` (7ファイル) | git rm -r |
| `.plans/eager-purring-elephant.md` | git rm -f |
| `archive/` (60ファイル) | git rm -r --cached |
| `.gitignore` | 編集（archive/, .codex/, .review/, .benchmarks/ 追加 + 既存変更統合） |
| `rag-local/README.md` | 編集（旧名削除, venv→.venv, ドキュメントテーブル3件追加） |
| `rag-local/CLAUDE.md` | 編集（Dockerfile行削除） |
| `README.md` | 編集（日付更新, 目次更新, 2アプリ構成書換, 未完了タスク更新, アーカイブセクション削除, ディレクトリ構造更新） |
| `rag-local/venv/` | 削除→`.venv/`で再作成（Git未追跡、コミット前に実施） |

---

## 検証

1. `git status` で意図通りの変更になっているか確認
2. `git ls-files --cached archive/` が空であること確認
3. `git ls-files --cached | grep -i docker` が空であること確認
4. ルートREADME の目次アンカーリンクが全て機能すること確認
5. `rag-local/README.md` のリンク先が全て存在するか確認
