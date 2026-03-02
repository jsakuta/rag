# rag-local 引き継ぎドキュメント再構成計画

## Context

rag-local を顧客（後任開発者）にファイル転送で引き継ぐ。
現状9ドキュメントに情報が分散し、セットアップ手順が複数箇所に重複、初心者導線が弱い。
CLAUDE.md は引き継ぎパッケージからは除外するが、リポジトリには残す（自分用）。
そこにしかない業務知識は他ドキュメントに移植して、後任も参照できるようにする。
rag-maintenance は今回対象外。rag-local 単体で完結するドキュメントにする。

---

## 新ドキュメント構成

**9ファイル + CLAUDE.md → 6ファイル（+ plans/）に再構成**

| 新構成 | 役割 | 統合元 |
|--------|------|--------|
| **README.md** | 概要 + 完全セットアップガイド + 引き継ぎ注意 | README + GOOGLE_CLOUD_AUTH + DB_BUILD_GUIDE(共通部分) + SECURITY(認証管理の要点のみ) |
| **docs/ANSWER_SUPPORT.md** | 回答支援AI（類似回答検索）の詳細ガイド（新規） | README(回答支援部分) + DB_BUILD_GUIDE(回答支援DB部分) + コード調査結果 |
| **docs/REVISION_OPS.md** | 改定影響調査AIの完全ガイド | REVISION_OPS + CLAUDE.md(業務知識) + DB_BUILD_GUIDE(改定DB部分) |
| **docs/ARCHITECTURE.md** | 技術アーキテクチャ + API仕様 + プロンプト | ARCHITECTURE + API_REFERENCE + PROMPTS + CLAUDE.md(技術注意事項) |
| **docs/CONFIGURATION.md** | 設定リファレンス（環境変数・YAML詳細） | CONFIGURATION(リファレンスに特化) |
| **docs/TROUBLESHOOTING.md** | トラブルシューティング（統合） | TROUBLESHOOTING + DB_BUILD_GUIDE(TS部分) |
| **docs/plans/** | 設計書アーカイブ（変更なし） | そのまま |

### 削除するファイル（統合後）

| 削除ファイル | 統合先 | 情報欠落リスク |
|------------|--------|-------------|
| `docs/GOOGLE_CLOUD_AUTH.md` | README Step 2 | 低: 核心部分（L9-31）を移植。Pythonコード例は開発者向けなので省略可 |
| `docs/DB_BUILD_GUIDE.md` | ANSWER_SUPPORT + REVISION_OPS | 低: 回答支援/改定別で分割移植。TS部分はTROUBLESHOOTING.mdへ |
| `docs/API_REFERENCE.md` | ARCHITECTURE 付録 | 低: 全量移植 |
| `docs/SECURITY.md` | README（認証管理の要点のみ） | 中: Key Vault/Docker/pre-commit等は省略。ただしローカル専用PJでGitなし引き渡しのため実質不要 |
| `docs/PROMPTS.md` | ARCHITECTURE | 低: 全量移植 |
| `CLAUDE.md` | リポジトリに残す（パッケージ除外） | なし: 情報は移植、原本も残る |

### CLAUDE.md 情報の移植先マッピング

| CLAUDE.md セクション | 行範囲 | 移植先 | 移植方法 |
|---------------------|--------|--------|---------|
| ドキュメント構成（リンク表） | L6-22 | 移植不要 | 新README.mdのドキュメント一覧で代替 |
| 事務改定差分ファイル構成 | L25-62 | REVISION_OPS.md「参照データ管理」 | 新セクション追加 |
| 差分.md 統一フォーマット | L65-102 | REVISION_OPS.md「差分ファイルの書き方」 | 新セクション追加 |
| 台帳番号と改定内容の対応 | L106-115 | REVISION_OPS.md 既存テーブル置換 | 完全版で上書き |
| 変更前シナリオDB生成フロー | L119-142 | REVISION_OPS.md「データ準備フロー」 | 新セクション追加 |
| 既知の問題（空行・行番号） | L146-161 | REVISION_OPS.md「既知の問題と注意事項」 | 新セクション追加 |
| 正解ID抽出ロジック | L164-174 | REVISION_OPS.md 正解IDセクション拡充 | 既存セクション拡充 |
| コード構成（ディレクトリ） | L177-267 | README.md ディレクトリ構造 | 詳細版で置換 |
| 主要モジュールの役割 | L269-286 | 移植不要 | ARCHITECTURE.mdと重複 |
| 環境変数設定 | L291-315 | 移植不要 | CONFIGURATION.md + .env.exampleで十分 |
| 技術的な注意事項 | L319-326 | ARCHITECTURE.md「規約と注意事項」 | 新セクション追加 |

---

## 実装ステップ

### Step 1: CLAUDE.md から情報を救出 → REVISION_OPS.md 拡充

REVISION_OPS.md に以下を追加・修正:

1. **「参照データ管理」セクション新設** — reference/ フォルダ構造（CLAUDE.md L27-47）
2. **「差分ファイルの書き方」セクション新設** — 統一フォーマットテンプレート（CLAUDE.md L65-102）
3. **台帳番号対応表の置換** — CLAUDE.md L106-115 の完全版で L106-113 を上書き
4. **「データ準備の詳細フロー」セクション新設** — 変更前シナリオDB生成フロー（CLAUDE.md L119-142）
5. **「正解ID抽出ロジック」拡充** — generate_correct_ids.py の抽出パターン仕様（CLAUDE.md L164-174）
6. **「既知の問題と注意事項」セクション新設** — 空行問題 + 行番号注意点（CLAUDE.md L146-161）
7. **改定DB構築コマンド移植** — DB_BUILD_GUIDE.md L93-101

**DB名の不整合修正**: REVISION_OPS.md 内の `rev01smile`（L76-88）を `rev01_smile` 形式に統一。
実コード（`config/business_areas.yaml`, `settings.yaml`）での命名を確認して正しい方に合わせる。

**対象ファイル**: `rag-local/docs/REVISION_OPS.md`

**完了基準**:
- [ ] CLAUDE.md の移植対象7セクション全てが REVISION_OPS.md に存在する
- [ ] DB名が全箇所でアンダースコア付き（`rev01_smile`）に統一されている
- [ ] DB_BUILD_GUIDE.md の改定DB構築コマンドが移植されている
- [ ] 既存の内容（処理フロー、出力フォーマット等）が損なわれていない

### Step 2: ANSWER_SUPPORT.md を新規作成

REVISION_OPS.md と対称的な構成で回答支援AIの専用ガイドを作成:

```markdown
# 回答支援AI（類似回答検索）

## 概要
- 目的: FAQ/シナリオからの類似回答検索
- 3モード: バッチ / Streamlit UI / プレフライト検証

## 処理フロー
- ハイブリッド検索の図解（ベクトル + キーワード → 加重スコア合算）
- 検索モード: original / llm_enhanced
- スコア計算式: final = vector_weight × vec + (1-vector_weight) × kw

## DB構造
- naibujimu（内部事務）: 預金+総則FAQ + naibujimu-botシナリオ = 11,439件
- smile（スマイル）: スマイルFAQ + smile-botシナリオ = 9,237件
- プロバイダー: azure_openai のみ（回答支援は単一プロバイダー）

## 使用方法
### バッチ処理
- コマンド: python apps/answer-support/main.py [--business X] [--limit N]
- 入力: data/input/ 配下のExcel
- 出力: data/output/latest/answer_batch_YYYYMMDD.xlsx

### Streamlit UI
- コマンド: streamlit run apps/answer-support/ui/chat.py
- 機能: 業務分野選択、検索パラメータ調整（ベクトル重み/モード/候補数）
- チャット履歴Excel保存

### プレフライト
- コマンド: python apps/answer-support/main.py preflight
- DB更新可否の事前検証（本番更新なし）

## DB構築
- DB_BUILD_GUIDE.md L69-91 から移植
- 差分: python scripts/build_db.py
- 全再構築: python scripts/build_db.py --force
- 指定業務分野: python scripts/build_db.py --business smile
- スキップロジックの説明

## 出力ファイル
- 列構成（8列）: #, ユーザーの質問, 検索クエリ, 類似質問, 類似回答, 類似度, ベクトルの重み, 候補数
- 1質問=複数行出力（候補数分）

## 設定パラメータ
- settings.yaml の ui/batch/common セクション概要
- 「詳細は docs/CONFIGURATION.md を参照」

## トラブルシューティング
- 「docs/TROUBLESHOOTING.md を参照」
```

**対象ファイル**: `rag-local/docs/ANSWER_SUPPORT.md`（新規作成）

**完了基準**:
- [ ] REVISION_OPS.md と同レベルの情報量（概要→フロー→DB→使い方→出力→TS）
- [ ] バッチ/UI/プレフライトの3モード全て記載
- [ ] DB構築手順が自己完結している（DB_BUILD_GUIDE.mdへのリンクなし）
- [ ] 出力Excel列構成が正確

### Step 3: README.md を再構成

rag-maintenance の言及を完全除外。セットアップ完全ガイドとして再構成:

```markdown
# RAG-Local（ローカル検証・評価基盤）

## 概要
- 2つのAIアプリ表（rag-maintenance列を削除）
  | AI | バッチ | UI | 用途 | 詳細ドキュメント |
  |---|---|---|---|---|

## ドキュメント一覧
- 6ファイル + 読む順序の導線
  1. README.md（今読んでいるもの）— セットアップ
  2. docs/ANSWER_SUPPORT.md — 回答支援AIの詳細
  3. docs/REVISION_OPS.md — 改定影響調査AIの詳細
  4. docs/CONFIGURATION.md — 設定リファレンス
  5. docs/ARCHITECTURE.md — 技術アーキテクチャ
  6. docs/TROUBLESHOOTING.md — 問題解決

## セットアップ手順（ゼロから動かすまで）

### Step 1: Python環境の構築
- venv作成 + pip install（現行維持）

### Step 2: 認証情報の準備
- **Google Cloud**: gemini_credentials.json の配置手順
  - GOOGLE_CLOUD_AUTH.md L9-31（ファイル配置 + 環境変数設定、Windows/Linux両方）
  - 必要権限: Vertex AI User
- **Azure OpenAI**: APIキーとエンドポイント
- **注意**: 認証ファイルは共有フォルダに置かない。パーミッション制限推奨

### Step 3: 環境変数の設定
- cp .env.example .env → 編集
- 主要変数の説明（4-5個）
- 「全変数の詳細は docs/CONFIGURATION.md を参照」

### Step 4: ソースデータの配置
- data/source/ のディレクトリ構造（DB_BUILD_GUIDE.md L35-62）
- ファイル命名規則
- 「ソースExcelと改定資料（reference/）は別途提供」を明記

### Step 5: DB構築
- 初回: python scripts/build_db.py --force
- 「回答支援AI用DBの詳細は docs/ANSWER_SUPPORT.md を参照」
- 「改定DB構築は docs/REVISION_OPS.md を参照」

### Step 6: 動作確認
- 回答支援AI: python apps/answer-support/main.py
- 改定影響調査: python apps/revision-ops/run_eval.py
- 「各AIの詳細は専用ドキュメントを参照」

## ディレクトリ構造
- CLAUDE.md L179-267 の詳細版を採用（src/内の全モジュール記載）

## AI使用箇所マップ（現行維持）

## 引き継ぎ時の注意

### 同梱しないもの（別途渡す / 除外）
| 対象 | 理由 | 対処 |
|------|------|------|
| .env | 認証情報 | .env.example から作成 |
| gemini_credentials.json | 認証情報 | 別経路で受け渡し |
| data/vector_db/ | ベクトルDB | build_db.py で再構築 |
| data/.keyword_cache/ | キャッシュ | 自動生成 |
| data/output/ | 出力ファイル | 実行時生成 |
| reference/ | 改定資料 | 別途提供 |
| .venv/ | Python仮想環境 | pip install で再作成 |
| logs/ | ログ | 実行時生成 |
| CLAUDE.md | 開発メモ | 引き継ぎ対象外 |

### 同梱するもの
- ソースコード（apps/, src/, ui/, scripts/, config/, prompt/, tests/）
- ドキュメント（README.md, docs/）
- 設定テンプレート（.env.example, requirements.txt, .streamlit/）
- ソースデータ（data/source/, data/input/）— 別途渡す場合は空ディレクトリ
```

**対象ファイル**: `rag-local/README.md`

**完了基準**:
- [ ] `rag-maintenance` への言及がゼロ
- [ ] セットアップ Step 1-6 を上から順に実行すれば動く（外部ドキュメント参照は補足のみ）
- [ ] Step 2 に Google Cloud 認証の具体的な手順がある（GOOGLE_CLOUD_AUTH.md なしで完結）
- [ ] Step 4 にソースデータの配置構造とファイル命名規則がある
- [ ] 引き継ぎ注意セクションに全除外対象が列挙されている
- [ ] ドキュメント一覧が新構成の6ファイルに更新されている

### Step 4: 引き継ぎパッケージ作成スクリプト

**許可リスト方式**のPythonスクリプト（Windows環境対応）。

```python
#!/usr/bin/env python3
"""引き継ぎパッケージ作成スクリプト（許可リスト方式）"""

# 許可リスト: これらのみコピーする
INCLUDE = [
    # ソースコード
    "apps/",
    "src/",
    "ui/",
    "config.py",
    "config/",
    "scripts/",          # このスクリプト自身も含む
    "prompt/",
    "tests/",
    # ドキュメント
    "README.md",
    "docs/",             # plans/ 含む
    # 設定テンプレート
    ".env.example",
    "requirements.txt",
    "requirements-dev.txt",
    "pytest.ini",
    ".streamlit/",
    # データ（空ディレクトリ構造のみ or 実データ含む）
    "data/source/",      # --include-data オプション時のみ実データコピー
    "data/input/",       # 同上
]

# 明示的に除外（許可リスト内でも除外）
EXCLUDE_PATTERNS = [
    "__pycache__",
    "*.pyc",
    ".pytest_cache",
]

# 機能:
# 1. DEST ディレクトリに許可リストのファイルのみコピー
# 2. --include-data: data/source/ と data/input/ の実データも含める
# 3. --dry-run: コピーせずに対象ファイル一覧を表示
# 4. 完了後にパッケージ内容サマリ出力（ファイル数、合計サイズ）
# 5. 秘密情報チェック（.env, *credentials*, *.key が含まれていないか検証）
```

**対象ファイル**: `rag-local/scripts/create_handover_package.py`（新規作成）

**完了基準**:
- [ ] `python scripts/create_handover_package.py --dry-run` で対象ファイル一覧が正しい
- [ ] 生成パッケージに `.env`, `gemini_credentials.json`, `CLAUDE.md` が含まれない
- [ ] 生成パッケージに `data/vector_db/`, `data/.keyword_cache/`, `data/output/` が含まれない
- [ ] 生成パッケージに `__pycache__/`, `.pytest_cache/` が含まれない
- [ ] `--include-data` で data/source/ と data/input/ の実データがコピーされる
- [ ] サマリ出力にファイル数と合計サイズが表示される
- [ ] 秘密情報チェックが PASS する

### Step 5: ARCHITECTURE.md に API_REFERENCE.md と PROMPTS.md を統合

- 末尾に「## API リファレンス」セクション追加（API_REFERENCE.md 全量）
- 末尾に「## プロンプト」セクション追加（PROMPTS.md 全量）
- 「## 規約と注意事項」セクション追加（CLAUDE.md L319-326 の技術注意事項）
- 古い参照を修正: `searcher.py` → `src/core/search/` 各モジュール
- rag-maintenance への言及を削除

**対象ファイル**: `rag-local/docs/ARCHITECTURE.md`

**完了基準**:
- [ ] API_REFERENCE.md の全クラス・メソッド定義が含まれている
- [ ] PROMPTS.md のプロンプトファイル一覧と説明が含まれている
- [ ] `searcher.py` への参照がゼロ（`search/` 配下に更新済み）
- [ ] `rag-maintenance` への言及がゼロ

### Step 6: CONFIGURATION.md を整理

- 冒頭に「初回セットアップは README.md を参照してください」を追加
- セットアップ手順的な記述（.envの作り方等）を削除（README に移動済み）
- settings.yaml と business_areas.yaml の説明を充実
- rag-maintenance への言及を削除

**対象ファイル**: `rag-local/docs/CONFIGURATION.md`

**完了基準**:
- [ ] 冒頭に README.md への誘導がある
- [ ] 環境変数テーブルが最新かつ正確
- [ ] settings.yaml の各セクション（common/ui/batch/evaluation）の説明がある
- [ ] `rag-maintenance` への言及がゼロ

### Step 7: TROUBLESHOOTING.md に散在するTS情報を統合

- DB_BUILD_GUIDE.md L137-166 のTS 4項目を移植（重複チェック後）
- REVISION_OPS.md の「既知の問題」へのクロスリンクを追加
- rag-maintenance 関連の記述があれば削除

**対象ファイル**: `rag-local/docs/TROUBLESHOOTING.md`

**完了基準**:
- [ ] DB_BUILD_GUIDE.md の4項目（ロックエラー、認証エラー、業務分野未検出、UI未表示）が含まれている
- [ ] 重複項目は統合されている（同じエラーが2回出ない）
- [ ] `rag-maintenance` への言及がゼロ

### Step 8: 統合済みファイルの削除

以下を削除（統合先に情報移植済み）:
- `rag-local/docs/GOOGLE_CLOUD_AUTH.md`
- `rag-local/docs/DB_BUILD_GUIDE.md`
- `rag-local/docs/API_REFERENCE.md`
- `rag-local/docs/SECURITY.md`
- `rag-local/docs/PROMPTS.md`

**注意**: `CLAUDE.md` はリポジトリに残す（引き継ぎパッケージには含めない）

**完了基準**:
- [ ] 上記5ファイルがファイルシステムから削除されている
- [ ] CLAUDE.md は存在する

### Step 9: クロスリファレンス確認 + 引き継ぎパッケージテスト

**対象ファイル**: 全 `.md` ファイル + 引き継ぎスクリプト

**完了基準**:
- [ ] grep で削除ファイル名（GOOGLE_CLOUD_AUTH, DB_BUILD_GUIDE, API_REFERENCE, SECURITY, PROMPTS）を検索 → ヒットゼロ
- [ ] grep で `rag-maintenance` を検索 → ヒットゼロ
- [ ] 引き継ぎスクリプト `--dry-run` の出力が期待通り
- [ ] 引き継ぎスクリプト実行で生成されたパッケージに秘密情報が含まれない
- [ ] パッケージ内の README.md を読み、Step 1-6 が自己完結している

---

## セルフレビュー: 計画の妥当性チェック

### 対処済みの懸念

| 懸念 | 対処 |
|------|------|
| 2つのAIアプリが非対称 | ANSWER_SUPPORT.md を新設し対称構成に |
| rag-maintenance が混入 | 全ステップで「rag-maintenance言及ゼロ」を完了基準に |
| CLAUDE.md を削除してほしくない | リポジトリに残す。パッケージ除外のみ |
| 引き継ぎスクリプトの信頼性 | 許可リスト方式 + dry-run + 秘密情報チェック + サマリ出力 |
| Windows環境 | .sh → .py に変更（shutil.copytree + pathlib） |
| data/source の扱い | --include-data オプションで明示選択。デフォルトは空ディレクトリ構造のみ |
| 情報移植の完全性 | CLAUDE.md 全セクション → 移植先マッピング表を明記 |

### 残存リスク

| リスク | 影響度 | 対策 |
|--------|-------|------|
| REVISION_OPS.md が肥大化（現315行→推定500行超） | 低 | 目次を追加。日常参照するのは「使用方法」だけ |
| docs/plans/ の設計書内リンク切れ | 低 | アーカイブ扱いなので修正不要 |
| ANSWER_SUPPORT.md の情報が実コードと乖離 | 中 | 実コードのmain.py, chat.py, output_handler.pyを読んで正確に記載 |
