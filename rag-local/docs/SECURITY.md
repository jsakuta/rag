# セキュリティガイド

このドキュメントでは、RAG-Localシステムのセキュリティ設定と認証情報管理について説明します。

## 目次

- [認証情報管理](#認証情報管理)
- [環境変数の保護](#環境変数の保護)
- [Gitリポジトリのセキュリティ](#gitリポジトリのセキュリティ)
- [Azure Key Vault連携](#azure-key-vault連携)
- [ネットワークセキュリティ](#ネットワークセキュリティ)
- [アクセス制御](#アクセス制御)
- [セキュリティベストプラクティス](#セキュリティベストプラクティス)

---

## 認証情報管理

### 機密情報ファイル一覧

以下のファイルには機密情報が含まれており、厳重な管理が必要です:

| ファイル | 内容 | 管理方法 |
|---------|------|----------|
| `gemini_credentials.json` | Google Cloud サービスアカウントキー | `.gitignore` に登録、定期ローテーション |
| `.env` | API キー、エンドポイントURL | `.gitignore` に登録、本番環境はKey Vault使用 |
| `azure_credentials.json` | Azure 認証情報（オプション） | `.gitignore` に登録 |

### Google Cloud サービスアカウント

#### 最小権限の原則

サービスアカウントには必要最小限の権限のみを付与してください:

**必須権限:**
- `Vertex AI User` - Gemini API使用
- `AI Platform Admin` - モデル管理（必要な場合のみ）

**付与しないでください:**
- `Owner` - 過度な権限
- `Editor` - 不要な変更権限
- `Storage Admin` - ストレージへの書き込み権限（不要な場合）

#### キーのローテーション

```bash
# 1. Google Cloud Consoleで新しいキーを作成
# 2. 新しいキーをテスト環境で検証
# 3. 本番環境に新しいキーをデプロイ
# 4. 古いキーを無効化
# 5. 古いキーファイルを安全に削除

# ファイルの安全な削除（Windows）
sdelete -p 7 old_gemini_credentials.json

# ファイルの安全な削除（Linux/Mac）
shred -vfz -n 7 old_gemini_credentials.json
```

推奨ローテーション期間: **90日**

### Azure OpenAI API キー

#### キーの管理

```env
# .env
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
```

#### キーのローテーション

1. Azure Portal で新しいキーを再生成
2. `.env` ファイルを更新
3. アプリケーションを再起動
4. 古いキーを削除

推奨ローテーション期間: **60-90日**

---

## 環境変数の保護

### .env ファイルの管理

#### 開発環境

```bash
# .env.example から .env を作成
cp .env.example .env

# .env を編集（機密情報を入力）
nano .env

# パーミッションを設定（Linux/Mac）
chmod 600 .env

# パーミッションを確認
ls -l .env
# -rw------- 1 user user 1024 Jan 01 12:00 .env
```

#### 本番環境

**推奨: Azure Key Vault を使用**（後述）

直接 `.env` ファイルを使用する場合:
```bash
# 暗号化されたストレージに配置
# アプリケーションユーザーのみ読み取り可能に設定
chown app_user:app_group .env
chmod 400 .env
```

### 環境変数の検証

起動時に必須環境変数が設定されているか確認:

```python
# config.py で実装済み
required_vars = [
    "DEFAULT_LLM_PROVIDER",
    "DEFAULT_LLM_MODEL",
    "DEFAULT_EMBEDDING_PROVIDER",
    "DEFAULT_EMBEDDING_MODEL",
]

for var in required_vars:
    if not os.getenv(var):
        raise ValueError(f"Required environment variable {var} is not set")
```

---

## Gitリポジトリのセキュリティ

### .gitignore 設定

**重要: 以下のファイルは絶対にGitにコミットしないでください**

```text
# .gitignore

# 認証情報
.env
*.env
gemini_credentials.json
azure_credentials.json

# データファイル
data/

# ログファイル
logs/
*.log

# Python
__pycache__/
*.pyc
.venv/
venv/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db
```

### Git履歴からの機密情報削除

誤ってコミットしてしまった場合:

```bash
# BFG Repo-Cleaner を使用（推奨）
bfg --delete-files gemini_credentials.json
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# または git filter-branch を使用
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch gemini_credentials.json" \
  --prune-empty --tag-name-filter cat -- --all
```

**注意:** 既にプッシュした場合は、キーを無効化して再生成してください。

### Pre-commit フック

機密情報のコミットを防ぐ:

```bash
# .git/hooks/pre-commit を作成
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash

# 機密情報パターンのチェック
if git diff --cached --name-only | grep -E '\.(env|json)$' | grep -v '.example'; then
    echo "Error: Attempting to commit sensitive files"
    exit 1
fi

# API キーパターンのチェック
if git diff --cached | grep -E '(api[_-]?key|password|secret|token).*='; then
    echo "Error: Attempting to commit API keys or secrets"
    exit 1
fi

exit 0
EOF

chmod +x .git/hooks/pre-commit
```

---

## Azure Key Vault連携

### セットアップ

#### 1. Azure Key Vault の作成

```bash
# Azure CLIでKey Vaultを作成
az keyvault create \
  --name rag-local-vault \
  --resource-group rag-local-rg \
  --location japaneast
```

#### 2. シークレットの登録

```bash
# Google Cloud 認証情報
az keyvault secret set \
  --vault-name rag-local-vault \
  --name gemini-credentials \
  --file gemini_credentials.json

# Azure OpenAI API Key
az keyvault secret set \
  --vault-name rag-local-vault \
  --name azure-openai-api-key \
  --value "your-api-key"
```

#### 3. アプリケーションの設定

```env
# .env
AZURE_KEY_VAULT_URL=https://rag-local-vault.vault.azure.net/
AZURE_KEY_VAULT_SCOPES=https://www.googleapis.com/auth/cloud-platform
```

#### 4. コード例

```python
# src/utils/utils.py で実装済み
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

def get_secret_from_key_vault(secret_name: str) -> str:
    """Azure Key Vault からシークレットを取得"""
    vault_url = os.getenv("AZURE_KEY_VAULT_URL")
    credential = DefaultAzureCredential()
    client = SecretClient(vault_url=vault_url, credential=credential)
    secret = client.get_secret(secret_name)
    return secret.value
```

### アクセス制御

#### マネージドIDの設定

```bash
# システムマネージドIDを有効化
az webapp identity assign \
  --name rag-local-app \
  --resource-group rag-local-rg

# Key Vault アクセスポリシーを設定
az keyvault set-policy \
  --name rag-local-vault \
  --object-id <managed-identity-object-id> \
  --secret-permissions get list
```

---

## ネットワークセキュリティ

### HTTPS通信

全てのAPI通信はHTTPSを使用:

```python
# 環境変数で強制
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/  # HTTPは不可
```

### プロキシ設定

企業ネットワーク内での使用:

```env
# .env
HTTP_PROXY=http://proxy.example.com:8080
HTTPS_PROXY=http://proxy.example.com:8080
NO_PROXY=localhost,127.0.0.1
```

### ファイアウォール設定

#### 送信トラフィック許可リスト

| サービス | ドメイン | ポート |
|---------|---------|--------|
| Vertex AI | *.googleapis.com | 443 |
| Azure OpenAI | *.openai.azure.com | 443 |
| Azure Key Vault | *.vault.azure.net | 443 |
| Anthropic | api.anthropic.com | 443 |
| OpenAI | api.openai.com | 443 |

---

## アクセス制御

### ファイルパーミッション

#### Linux/Mac

```bash
# プロジェクトディレクトリ
chmod 755 rag-local/

# 認証情報ファイル（所有者のみ読み取り）
chmod 400 gemini_credentials.json
chmod 400 .env

# スクリプトファイル（実行権限）
chmod 755 scripts/*.py

# ログディレクトリ
chmod 755 logs/
chmod 644 logs/*.log
```

#### Windows

```powershell
# 認証情報ファイルのアクセス制御
icacls gemini_credentials.json /inheritance:r
icacls gemini_credentials.json /grant:r "%USERNAME%:R"

# .envファイルのアクセス制御
icacls .env /inheritance:r
icacls .env /grant:r "%USERNAME%:R"
```

### Docker コンテナのセキュリティ

```dockerfile
# Dockerfile
FROM python:3.11-slim

# 非rootユーザーで実行
RUN useradd -m -u 1000 appuser
USER appuser

# 機密情報は環境変数経由
ENV GEMINI_CREDENTIALS_PATH=/app/secrets/gemini_credentials.json

# 読み取り専用でマウント
# docker run -v $(pwd)/gemini_credentials.json:/app/secrets/gemini_credentials.json:ro
```

実行例:
```bash
docker run --rm \
  -v $(pwd)/gemini_credentials.json:/app/secrets/gemini_credentials.json:ro \
  -e GEMINI_CREDENTIALS_PATH=/app/secrets/gemini_credentials.json \
  --user 1000:1000 \
  rag-local:latest
```

---

## セキュリティベストプラクティス

### チェックリスト

#### 開発環境

- [ ] `.gitignore` に機密情報ファイルを登録
- [ ] `.env.example` のみをコミット（値は空）
- [ ] 認証ファイルのパーミッション設定（400または600）
- [ ] Pre-commit フックの設定
- [ ] 定期的なキーローテーション（90日）

#### 本番環境

- [ ] Azure Key Vault の使用
- [ ] マネージドIDによる認証
- [ ] HTTPS通信の強制
- [ ] ファイアウォール設定
- [ ] ログの暗号化保存
- [ ] 定期的なセキュリティ監査
- [ ] アクセスログの監視

### コードレビュー時の確認事項

- [ ] ハードコードされたAPIキーがないか
- [ ] 認証情報をログ出力していないか
- [ ] エラーメッセージに機密情報が含まれていないか
- [ ] 一時ファイルに認証情報を書き込んでいないか

```python
# NG: ハードコード
API_KEY = "sk-1234567890abcdef"

# OK: 環境変数
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

# NG: ログ出力
logger.info(f"Using API key: {api_key}")

# OK: マスク
logger.info(f"Using API key: {api_key[:8]}...")
```

### 監査ログ

重要な操作をログに記録:

```python
# src/utils/logger.py で実装
logger.info(f"API request - provider: {provider}, user: {user_id}, timestamp: {timestamp}")
logger.warning(f"Authentication failed - IP: {ip_address}, attempts: {count}")
```

---

## インシデント対応

### 認証情報漏洩時の対応

1. **即座に無効化**
   - Google Cloud Console でサービスアカウントキーを無効化
   - Azure Portal で API キーを再生成

2. **影響範囲の確認**
   - アクセスログを確認
   - 不正利用の有無を調査

3. **新しいキーの発行**
   - 新しいサービスアカウントキーを作成
   - 全環境で更新

4. **報告**
   - セキュリティチームに報告
   - インシデントレポートを作成

---

## 関連ドキュメント

- [docs/GOOGLE_CLOUD_AUTH.md](./GOOGLE_CLOUD_AUTH.md) - Google Cloud 認証設定
- [docs/CONFIGURATION.md](./CONFIGURATION.md) - 環境変数設定
- [docs/TROUBLESHOOTING.md](./TROUBLESHOOTING.md) - トラブルシューティング
- [README.md](../README.md) - プロジェクト概要

---

## セキュリティポリシー

脆弱性を発見した場合は、GitHub Issues ではなく直接セキュリティチームに報告してください。
