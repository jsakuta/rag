# RAG_yokin → rag-streamlit

> **注意**: 本プロジェクトは将来的に `rag-streamlit` にリネームされます。

## 他プロジェクトとの違い
- **特徴**: Streamlit UIによる対話的検索、リアルタイム検索、デモ・プレゼンテーションに最適
- **代替プロジェクト**:
  - バッチ処理が必要な場合: [rag_v1.0 (rag-batch)](../rag_v1.0/)
  - 最新技術を使いたい場合: [rag_v2.1 (rag-gemini)](../rag_v2.1/)
  - プロジェクト全体の比較: [ルートREADME](../README.md)

---

## 概要

- RAG（Retrieval Augmented Generation）を活用した預金商品情報検索システム
- **主な機能**：
  - 預金商品に関する質問に対して参照データから最適な回答を生成
  - 大規模データセットからの高精度な情報検索
  - ベクトル検索によるセマンティック類似性に基づく回答生成
- **技術スタック**:
  - Python 3.7+
  - SentenceTransformer (multilingual-e5-base)
  - OpenAI/Anthropic/Google APIクライアント
  - ベクトルデータベースとキャッシング
- **デモ画面**:
  ![デモ画面](./docs/images/demo.png)
  ※画像は実際のUIをセットアップ後に追加してください

## 目次

- [システム要件](#システム要件)
- [セットアップ](#セットアップ)
- [使用方法](#使用方法)
- [設定パラメータ](#設定パラメータ)
- [開発](#開発)
- [トラブルシューティング](#トラブルシューティング)
- [セキュリティ](#セキュリティ)
- [GitHubへのプッシュ方法](#GitHubへのプッシュ方法)

## 1. システム要件

### 1.1 必要な環境

- OS: Windows, macOS, Linux
- Python 3.7+
- 十分なRAM (処理データ量に応じて4GB以上推奨)
- インターネット接続 (LLM APIにアクセスするため)

### 1.2 依存関係

- 外部サービス依存:
  - OpenAI API (GPT-4等)
  - Anthropic API (Claude)
  - Google API (Gemini)
- ライブラリ依存:
  - sentence-transformers
  - pandas
  - numpy
  - logging
  - pytorch
  - 他の依存ライブラリ
- インストールコマンド:

```bash
pip install -r requirements.txt
```

## 2. セットアップ

### 2.1 システムコンポーネント

| ファイル/ディレクトリ | 説明 | 用途 |
| --- | --- | --- |
| 📁 reference/ | 参照データとベクトルキャッシュを格納 | 検索対象データ |
| 📁 reference/vector_cache/ | 計算済みベクトルのキャッシュ | 高速化のためのキャッシュ |
| 📄 utils.py | ユーティリティ関数 | ロガー等の共通機能 |
| 📄 processor.py | データ処理機能 | 入力データの前処理 |
| 📄 search.py | 検索機能 | ベクトル検索実装 |
| 📄 ui.py | ユーザーインターフェース | フロントエンド表示 |
| 📄 main.py | メインアプリケーション | エントリーポイント |
| 📄 config.py | 設定ファイル | 環境設定とパラメータ |
| 📄 .env | 環境変数ファイル | APIキー等の機密情報 |
| 📄 Dockerfile | Dockerコンテナ構成 | コンテナ化のための設定 |

### 2.2 環境変数の設定

`.env`ファイルを作成し、必要なAPIキーを設定します:

```env
# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key

# Anthropic API Key (if using Claude)
ANTHROPIC_API_KEY=your_anthropic_api_key

# Google API Key (if using Gemini)
GOOGLE_API_KEY=your_google_api_key
```

### 2.3 初期設定

1. リポジトリをクローン:

```bash
git clone [リポジトリURL]
cd RAG_yokin
```

2. 仮想環境を作成し、依存関係をインストール:

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

3. 参照データを`reference/`ディレクトリに配置

## 3. 使用方法

### 3.1 基本的な使い方

1. アプリケーションを起動:

```bash
python main.py
```

2. 入力データ（質問またはクエリ）を準備:
   - Excelファイル形式の場合は`input_data.xlsx`に保存
   - テキスト形式の場合は対応するディレクトリに配置

3. 参照データを確認:
   - `reference_data.xlsx`が正しく配置されていることを確認

4. 検索処理を実行し、結果を確認

### 3.2 検索設定のカスタマイズ

検索パラメータは`config.py`で調整可能です:

```python
# 類似度閾値の設定例
SIMILARITY_THRESHOLD = 0.75

# 返却する結果の最大数
MAX_RESULTS = 5
```

## 4. 設定パラメータ

| パラメータ名 | 説明 | デフォルト値 | 設定可能な値 |
| --- | --- | --- | --- |
| MODEL_NAME | 使用する埋め込みモデル | "intfloat/multilingual-e5-base" | SentenceTransformerの任意のモデル |
| CACHE_DIR | ベクトルキャッシュの保存場所 | "reference/vector_cache" | 任意のディレクトリパス |
| SIMILARITY_THRESHOLD | 類似度の閾値 | 0.75 | 0.0〜1.0の浮動小数点 |
| MAX_RESULTS | 返却する最大結果数 | 5 | 任意の正の整数 |
| LOG_LEVEL | ログレベル | "INFO" | "DEBUG", "INFO", "WARNING", "ERROR" |

## 5. 開発

### 5.1 開発環境構築

```bash
git clone [リポジトリURL]
cd RAG_yokin
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### 5.2 コーディング規約

- PEP 8に準拠したコードスタイルを使用
- docstringはGoogle形式で記述
- 関数・メソッドには適切なコメントを付与
- コミットメッセージは変更内容を明確に記述
- プルリクエストはテンプレートに従って作成

## 6. トラブルシューティング

### 6.1 よくある問題と解決方法

| 問題 | 原因 | 解決方法 |
| --- | --- | --- |
| APIキーエラー | .envファイルが正しく設定されていない | .envファイルの内容とフォーマットを確認 |
| メモリエラー | 大きなデータセットを処理する際のメモリ不足 | バッチサイズを小さくするか、メモリを増設 |
| ベクトル変換エラー | 不適切な入力データ形式 | データ形式を確認し、前処理を適切に実行 |

### 6.2 ログの確認

ログファイルは`app.log`に保存されます。エラーが発生した場合は、このファイルを確認して原因を特定してください。

## 7. セキュリティ

### ⚠️ 重要な注意事項

- `.env`ファイルは絶対にGitHubにプッシュしないでください
- APIキーは厳重に管理し、公開リポジトリに含めないでください
- 機密情報を含むデータセットは適切に保護してください

### APIキーの管理

- APIキーは環境変数またはシークレット管理サービスを使用して管理
- 本番環境ではAPIキーの定期的なローテーションを実施
- 最小権限の原則に従ってAPIキーの権限を設定

## 8. GitHubへのプッシュ方法

input、output、referenceフォルダを除外してGitHubにプッシュするには、`.gitignore`ファイルを使用します。

### 8.1 .gitignoreの設定

以下の内容の`.gitignore`ファイルを作成します:

```
# 特定のフォルダを除外
/input/
/output/
/reference/

# グローバルに除外するパターン
*.log
*.pyc
__pycache__/
.env
*.env
*.env.*
.venv
venv/
env/
.DS_Store
```

これにより、input、output、referenceフォルダ以外のすべてのファイルとフォルダがGitリポジトリに追加されます。

### 8.2 プッシュの手順

```bash
# リポジトリの初期化（初回のみ）
git init

# ファイルの追加
git add .

# コミット
git commit -m "Initial commit without input, output, reference folders"

# リモートリポジトリの追加（初回のみ）
git remote add origin [リポジトリURL]

# プッシュ
git push -u origin main
```

### 8.3 既にプッシュされているフォルダの削除

すでにGitHubにプッシュされているinput、output、referenceフォルダを削除するには、以下の手順を実行します：

```bash
# ファイルをローカルに残したまま、Gitの追跡対象から削除
git rm -r --cached input/ output/ reference/

# 変更をコミット
git commit -m "Remove input, output, reference folders from repository"

# GitHubにプッシュ
git push origin main
```

この操作により、ローカルのファイルはそのまま保持されますが、GitHubリポジトリからは削除されます。また、`.gitignore`ファイルに既に設定されているため、今後これらのフォルダが誤ってプッシュされることはありません。

操作後、GitHubリポジトリを確認して、フォルダが正常に削除されたことを確認してください。
