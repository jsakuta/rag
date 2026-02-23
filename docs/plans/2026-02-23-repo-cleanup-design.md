# リポジトリ全体クリーンアップ設計書

**作成日**: 2026-02-23
**スコープ**: rag-local + リポジトリ全体のフォルダ構造整理
**アプローチ**: 段階的クリーンアップ（3フェーズ）

---

## 背景

リポジトリ内に重複ネスト構造（`data/data/`, `logs/logs/`, `prompt/prompt/`等）、残存フォルダ（`rag-gemini/`）、
古いドキュメント等が蓄積しており、構造の見通しが悪くなっている。

## Phase 1 — 明らかなゴミ掃除（リスクゼロ）

使われていない・空・重複のフォルダ/ファイルを削除。機能影響なし。

| 対象 | 理由 |
|------|------|
| `rag-local/data/data/` | 使われていない重複ネスト |
| `rag-local/data/output/output/` | 空フォルダ |
| `rag-local/logs/logs/` | 古いログの重複 |
| `rag-local/prompt/prompt/` | ファイルの重複コピー |
| `rag/nul` | Windows予約名ファイル |
| `archive/rag-batch/.venv/` | 仮想環境（再作成可能） |
| `archive/rag-reranker/.venv/` | 同上 |
| `archive/rag-streamlit/.venv/` | 同上 |

## Phase 2 — 残存フォルダ整理

| 対象 | アクション |
|------|-----------|
| `rag-gemini/` | data/ の中身を rag-local/data/ と比較。重複なら削除、ユニークなら統合 |
| `rag-maintenance/.plans/` | 実装済み計画を `.plans/archive/` に移動 |
| ルート `.plans/` | 実装済みならアーカイブ |

## Phase 3 — ドキュメント整理

`rag-local/docs/` の各ファイルを確認し、古い・不整合なものを削除/統合。

**保持確定**: `GOOGLE_CLOUD_AUTH.md`（ユーザー指定）

**確認対象**: ARCHITECTURE.md, API_REFERENCE.md, CONFIGURATION.md, SECURITY.md,
TROUBLESHOOTING.md, PROMPTS.md, REVISION_EVALUATION.md

## 対象外

- `rag-local/src/` 内のコードリファクタリング
- `rag-local/データ整理/`（業務データ、保持）
- `rag-maintenance/` のコード変更
