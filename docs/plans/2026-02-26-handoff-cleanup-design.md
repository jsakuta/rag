# 引き継ぎ前一括クリーンアップ 設計書

**日付**: 2026-02-26
**目的**: rag-local の引き継ぎ前に不要ファイル・旧参照・キャッシュを徹底整理

## 背景

2026-02-24 の業務分野再編（deposit+general → naibujimu）後、ドキュメントに旧参照が残存。
また build_db.py 統合後に不要になったスクリプトや旧データファイルがある。

## 設計

### セクション1: ドキュメント修正（HIGH）

deposit/general の旧参照を naibujimu/smile に更新。

| ファイル | 修正内容 |
|----------|---------|
| `docs/CONFIGURATION.md` | 業務分野テーブル・JSONサンプルの deposit/general → naibujimu/smile |
| `docs/TROUBLESHOOTING.md` | エラー例・JSONマッピングの deposit/general → naibujimu/smile |
| `docs/API_REFERENCE.md` | リクエスト/レスポンス例の deposit/general → naibujimu/smile |

### セクション2: 不要スクリプト削除（MED）

| ファイル | 理由 |
|----------|------|
| `scripts/generate_db_source.py` | build_db.py 統合で不要 |
| `scripts/remove_empty_dirs.py` | ワンオフユーティリティ、コアワークフロー外 |
| `scripts/archive_output_files.py` | 同上 |

### セクション3: 不要データ削除（MED）

| 対象 | サイズ | 理由 |
|------|--------|------|
| `data/source/faq/archive/総則_履歴データ_20250829.xlsx` | 767KB | 旧業務分野 |
| `data/source/faq/archive/預金_履歴データ_20250830.xlsx` | 977KB | 旧業務分野 |
| `data/source/scenarios/individual/` | 0 | 空ディレクトリ、参照なし |

### セクション4: キャッシュ・一時ファイル清掃（LOW）

| 対象 | 方法 |
|------|------|
| `__pycache__/` (全階層) | rm -rf |
| `.pytest_cache/` | rm -rf |
| `.serena/` | rm -rf |
| `logs/app.log` | truncate |

### コミット計画

| # | メッセージ | 内容 |
|---|-----------|------|
| 1 | `docs: deposit/general の旧参照を naibujimu/smile に更新` | ドキュメント3ファイル |
| 2 | `chore: 不要スクリプト3本と旧データファイルを削除` | スクリプト3 + Excel2 + 空ディレクトリ |

キャッシュ清掃は git 追跡外のためコミット不要。

## 不要なもの（YAGNI）

- cleanup.sh スクリプトは作らない（キャッシュは .gitignore 済み）
- ログローテーション設定は今回スコープ外
