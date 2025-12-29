# Gemini API セットアップガイド

## 概要
このガイドは、別環境でGemini APIを使用するためのセットアップ手順を説明します。

## 必要なファイル
- `gemini_credentials.json` - 認証情報ファイル

## セットアップ手順

### 1. 認証情報ファイルの配置
```
gemini_credentials.json を安全な場所に配置してください
```

### 2. 環境変数の設定

#### Windows (PowerShell)
```powershell
$env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\gemini_credentials.json"
```

#### Windows (コマンドプロンプト)
```cmd
set GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\gemini_credentials.json
```

#### Linux/Mac
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/gemini_credentials.json"
```

### 3. Pythonでの使用例

#### 基本的な使用方法
```python
import google.generativeai as genai
from google.oauth2 import service_account

# 認証情報の読み込み
credentials = service_account.Credentials.from_service_account_file(
    'gemini_credentials.json',
    scopes=['https://www.googleapis.com/auth/cloud-platform']
)

# Gemini APIの設定
genai.configure(credentials=credentials)

# モデルの初期化
model = genai.GenerativeModel('gemini-2.0-flash-001')

# テキスト生成
response = model.generate_content("こんにちは、世界！")
print(response.text)
```

#### Vertex AIでの使用例
```python
import vertexai
from google.oauth2 import service_account

# 認証情報の読み込み
credentials = service_account.Credentials.from_service_account_file(
    'gemini_credentials.json',
    scopes=['https://www.googleapis.com/auth/cloud-platform']
)

# Vertex AIの初期化
vertexai.init(
    project="pj-cbk001",
    location="us-central1",
    credentials=credentials
)

# モデルの初期化
model = vertexai.language_models.TextGenerationModel.from_pretrained("gemini-2.0-flash-001")

# テキスト生成
response = model.predict("こんにちは、世界！")
print(response.text)
```

### 4. 必要なライブラリのインストール

```bash
pip install google-generativeai
pip install google-cloud-aiplatform
pip install google-auth
```

### 5. 設定ファイル例 (settings.json)

```json
{
  "api_keys": {
    "google": "your_google_api_key_here"
  },
  "vertex_ai": {
    "project_id": "pj-cbk001",
    "location": "us-central1"
  },
  "default_models": {
    "vertex-gemini": "gemini-2.0-flash-001"
  }
}
```

## セキュリティ注意事項

⚠️ **重要**
- `gemini_credentials.json` は機密情報です
- 外部に漏洩しないよう適切に管理してください
- Gitリポジトリにコミットしないでください
- 適切なアクセス権限を設定してください

## トラブルシューティング

### よくある問題

1. **認証エラー**
   - 環境変数が正しく設定されているか確認
   - ファイルパスが正しいか確認

2. **権限エラー**
   - サービスアカウントに適切な権限が付与されているか確認

3. **API制限エラー**
   - 使用量制限に達していないか確認
   - 適切なクォータが設定されているか確認

## サポート

問題が発生した場合は、以下を確認してください：
- 認証情報の有効性
- ネットワーク接続
- APIキーの権限設定 