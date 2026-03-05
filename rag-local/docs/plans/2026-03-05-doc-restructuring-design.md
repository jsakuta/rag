# ドキュメント再構成 設計書

## 背景

rag-local のドキュメント（6ファイル、計約4,100行）が逐次追記により以下の問題を抱えている:

1. **大量の重複（DRY違反）**: ディレクトリ構造、環境変数、DB構造、ファイル命名規則など10箇所以上で同一情報が複数ファイルに存在
2. **ARCHITECTURE.md の肥大化**: 1,518行。APIリファレンス（~700行）がファイルの半分を占める
3. **読者ペルソナの混在**: チュートリアル・How-to・リファレンス・説明が各ファイル内で未整理
4. **情報の不整合リスク**: 同一情報の複数箇所管理により、片方だけ更新されるリスク

## 制約

- ファイル数: 現行6ファイル維持（README.md + docs/5ファイル）
- 読者: RAG理解済み、プロジェクト固有設計を知りたい人
- 用途: 開発（コード変更）と運用（設定変更・改定追加）の両方
- CLAUDE.md は対象外

## 設計方針

### Diataxis フレームワークによる責務再定義

各ファイルの責務を Diataxis の4分類（チュートリアル / How-to / リファレンス / 説明）に基づいて明確化する。

| ファイル | 責務 | Diataxis 分類 |
|---------|------|--------------|
| README.md | セットアップ手順 + 全体像 + 引き継ぎ | チュートリアル |
| ANSWER_SUPPORT.md | 回答支援AIの使い方・設定 | How-to ガイド |
| REVISION_OPS.md | 改定影響調査の使い方・設定 | How-to ガイド |
| CONFIGURATION.md | 環境変数・YAML 完全リファレンス | リファレンス |
| ARCHITECTURE.md | 技術アーキテクチャ・モジュール設計 | 説明 + リファレンス |
| TROUBLESHOOTING.md | 問題解決 | How-to ガイド |

### SSOT（Single Source of Truth）ルール

「ある情報は1箇所にだけ書く。他の箇所からはリンクで参照する。」

| 情報 | SSOT | 現在の重複箇所（リンクに置換） |
|------|------|-------------------------------|
| ディレクトリ構造 | README.md | ARCHITECTURE.md |
| 環境変数一覧・詳細 | CONFIGURATION.md | README.md（主要変数の概要表のみ残す） |
| 通常DB構造 | ANSWER_SUPPORT.md | - |
| 改定DB構造 | REVISION_OPS.md | - |
| ファイル命名規則 | README.md (Step 4) | ANSWER_SUPPORT.md |
| AI使用箇所マップ | README.md | ARCHITECTURE.md（削除） |
| スコア計算式 | ANSWER_SUPPORT.md | REVISION_OPS.md（リンク + 差分のみ） |
| 正解IDフォーマット | REVISION_OPS.md | - |
| ドキュメント一覧 | README.md | 他5ファイル末尾（全削除） |
| build_db.py 使い方 | README.md (Step 5) | ANSWER_SUPPORT.md（リンク + 固有オプションのみ） |
| settings.yaml 構成 | CONFIGURATION.md | ANSWER_SUPPORT.md（リンクに置換） |
| プロバイダー表 | ANSWER_SUPPORT.md | REVISION_OPS.md（リンクに置換） |

## 各ファイルの変更詳細

### README.md（363行 → ~300行）

**変更:**
- 環境変数テーブル: 主要3変数 + CONFIGURATION.md へのリンクに圧縮
- 引き継ぎパッケージ: フラグ表を圧縮（コマンド例 + --help 参照）

**維持:**
- セットアップ手順（Step 1-6）
- ドキュメント一覧表（推奨読み順）
- AI使用箇所マップ（概要表）
- ディレクトリ構造（SSOT）
- 引き継ぎ時の注意

### ANSWER_SUPPORT.md（363行 → ~250行）

**削除・リンク化:**
- DB構築セクション → README.md Step 5 へリンク（固有オプションのみ言及）
- ファイル命名規則 → README.md Step 4 へリンク
- 設定パラメータセクション → CONFIGURATION.md へリンク（3行サマリ表は維持）
- 埋め込みプロバイダー表 → ARCHITECTURE.md 内で扱う

**維持:**
- 処理フロー Mermaid 図
- 検索モード・スコア計算式（SSOT）
- 使用方法（バッチ/UI/プレフライト）
- 出力ファイル列構成
- DB構造（業務分野表・ディレクトリ構成、通常DB の SSOT）

### REVISION_OPS.md（619行 → ~480行）

**削除・リンク化:**
- スコア計算式 → ANSWER_SUPPORT.md へリンク + 差分のみ記述
- プロバイダー表 → ANSWER_SUPPORT.md へリンク
- 参照データ管理のフォルダ構造 → 圧縮

**維持:**
- 2プロバイダー比較の背景・多段階検索の設計意図
- 改定番号とDBの対応表（SSOT）
- 使用方法（Step 1-5）
- 出力ファイルのシート構成・列詳細
- 新しい改定の追加手順

### CONFIGURATION.md（630行 → ~530行）

**削除:**
- 末尾トラブルシューティング（TROUBLESHOOTING.md に SSOT）
- 末尾「関連ドキュメント」リンクリスト

**維持:**
- 環境変数の完全リファレンス（SSOT）
- settings.yaml の詳細（SSOT）
- 設定検証（バリデーション表）

### ARCHITECTURE.md（1,518行 → ~600行）

**大幅削減:**
- API リファレンス（~700行）→ 主要クラス一覧表（~80行）に圧縮
  - 理由: メソッドシグネチャはコード自体が最新のリファレンス。ドキュメントに全シグネチャを複製するとコード変更時に古くなるリスクが高い
- プロンプトセクション → 削除（README.md の AI使用箇所マップに統合済み）
- AI使用箇所マップ重複 → 削除（README.md が SSOT）
- ディレクトリ構造重複 → 削除（README.md が SSOT）

**再構成後の目次:**
1. 全体構成（レイヤー図）
2. データフロー（通常バッチ / 多段階検索 / 改定評価）
3. 主要クラス一覧（表形式、1クラス1行）
4. モジュール依存関係（ツリー + 外部ライブラリ表）
5. 拡張ガイド（業務分野追加 / 埋め込みモデル追加）
6. パフォーマンス（キャッシング・並列処理）
7. テスト（テストファイル一覧・実行方法）

### TROUBLESHOOTING.md（649行 → ~600行）

**変更:**
- 末尾「関連ドキュメント」リンクリスト削除
- 微調整のみ（既に SSOT として機能）

### 全ファイル共通

- 各ファイル末尾の「関連ドキュメント」セクション削除（README.md のドキュメント一覧表が SSOT）

## 削減見込み

| ファイル | 現在 | 目標 | 削減率 |
|---------|------|------|--------|
| README.md | 363行 | ~300行 | 17% |
| ANSWER_SUPPORT.md | 363行 | ~250行 | 31% |
| REVISION_OPS.md | 619行 | ~480行 | 22% |
| CONFIGURATION.md | 630行 | ~530行 | 16% |
| ARCHITECTURE.md | 1,518行 | ~600行 | 60% |
| TROUBLESHOOTING.md | 649行 | ~600行 | 8% |
| **合計** | **4,142行** | **~2,760行** | **33%** |

## 参考

- [Diataxis フレームワーク](https://diataxis.fr/) — ドキュメントの4分類体系
- [DRY原則](https://en.wikipedia.org/wiki/Don't_repeat_yourself) — 情報の重複排除
- [Code handover best practices](https://understandlegacycode.com/blog/7-practices-successful-handover/) — コード引き継ぎのベストプラクティス
