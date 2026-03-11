# 導入手順書 網羅的レビュー修正 設計書

- 文書ID: REVIEW-SETUP-001
- 作成日: 2026-03-11
- 対象: `docs/導入手順書.md`（全2,239行）

---

## 1. レビュー概要

導入手順書を網羅的にレビューし、32件の指摘事項を検出。
高3件・中15件・低9件、加えてユーザー指示による方針変更5件。

## 2. 調査結果

### Entra ID 権限（#7）
- Application Administrator は `microsoft.directory/servicePrincipals/appRoleAssignedTo/update` 権限を持ち、appRoleAssignments 管理が可能
- **結論: 現行記載「アプリケーション管理者以上」で正確。修正不要**
- 出典: [Microsoft Entra built-in roles](https://learn.microsoft.com/en-us/entra/identity/role-based-access-control/permissions-reference)

### API バージョン（#16）
- `2025-09-01` は Azure AI Search REST API の最新安定版（GA）
- **結論: 修正不要**
- 出典: [API Versions - Azure AI Search](https://learn.microsoft.com/en-us/rest/api/searchservice/search-service-api-versions)

### Key Vault（#9）
- Bot ソースコード（`maintenance-bot/src/`）では Key Vault を未参照
- Managed Identity によりサービス間認証が資格情報なしで可能 → Azure サービス間通信では Key Vault にシークレットを格納する必要がない
- **Key Vault が必要なケース**: Microsoft Entra 認証をサポートしないサービスへの接続、外部 API キー、証明書、暗号キーの管理
- 出典: [What are managed identities?](https://learn.microsoft.com/en-us/entra/identity/managed-identities-azure-resources/overview)

### Python 環境
- 実行環境: Python 3.13.5
- openpyxl 最低要件: Python 3.8+
- **前提条件には「Python 3.10 以上」を記載**（3.8 は EOL 間近のため余裕を持つ）
- 使用箇所: `convert-excel-to-json.py`（Step 15）のみ

## 3. 修正方針一覧

### 高優先度（3件）

| # | 指摘 | 対応 |
|---|------|------|
| 1 | 版数 CLAUDE.md で v1.4 と記載 | CLAUDE.md 側を「1.0」に修正。手順書の版数は 1.0 のまま |
| 10 | 「13.5節」が存在しない | 正しい参照先（Step 11）に修正 |
| 25 | Step 12 手書き JSON と scripts/ の二重管理 | Step 12 を `scripts/*.json` ファイル参照に書き換え |

### 中優先度（15件）

| # | 指摘 | 対応 |
|---|------|------|
| 2+3 | 所要時間「別途」＋過少見積もり | Step 1〜12: 約3〜4時間、Step 13〜15: 約1〜2時間に修正 |
| 5 | Azure CLI 最低バージョン | 2.60 → 2.64 に更新 |
| 6 | VS Code バージョン | 「1.90以降（M365 Agents Toolkit v6対応版）」に変更 |
| 8 | API キー無効化の警告不足 | `--disable-local-auth` 注記に Step 12 REST API との関係を警告追加 |
| 9 | Key Vault の必要性不明確 | Step 5 を削除。本番移行考慮事項に Key Vault の用途・Managed ID との関係を記載 |
| 11 | Week 概念の突然の登場 | 該当記述を削除 |
| 12 | 「設定済み」の誤り | 「Step 13 で設定する」に修正 |
| 13+14 | ネットワーク・本番移行 | 付録に「本番移行時の主要考慮事項」を5項目程度で簡潔に記載 |
| 17 | maxFailedItems: -1 | 「PoC 用設定。本番では上限値を設定すること」注記追加 |
| 19 | AZURE_SUBSCRIPTION_ID | **太字**で手動記入必須を強調 |
| 20 | ignore ファイルの役割不明確 | `.webappignore`（ローカル）と `.deployignore`（zip デプロイ）の役割を注記 |
| 23 | Step 11 と Step 13 の設定順序 | 「SPO 関連のみ Step 11 で先行設定」と注記 |
| 24 | LLM 仮値の不明確さ | 「任意の文字列で可（例: `dummy`）」と明記 |
| 26+27 | 前提条件の欠落 | rag-local セットアップ済み + Python 3.10 以上 + pip を前提条件に追加 |
| 28 | fieldMappings の説明不足 | scenarios/faqs の構造差異を注記 |

### 低優先度（5件）

| # | 対応 |
|---|------|
| 15 | Cosmos DB ロール定義 ID に注釈 |
| 18 | Basic プラン同時 Indexer 数の注記 |
| 29 | CLI 基本・ポータル参考の方針を文書冒頭に明記 |
| 30 | generateClientSecret を Tips ブロック化 |
| 31+32 | コスト概算の注記修正 |

### Step 番号の振り直し

Key Vault（旧 Step 5）削除により、Step 6〜15 を Step 5〜14 に繰り上げ。
全体は **14ステップ** になる。

---

## 4. 影響範囲

- `docs/導入手順書.md`: 全面修正
- `CLAUDE.md`: 版数記載の修正（v1.4 → 1.0）
- scripts/ JSON ファイル: 変更なし（参照されるだけ）
