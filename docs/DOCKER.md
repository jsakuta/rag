# Docker デプロイガイド

RAG システムの Docker デプロイに関する共通ガイドです。

---

## 概要

Docker を使用することで、環境の違いに依存せず一貫した動作を実現できます。

### Docker を使うメリット

| メリット | 説明 |
|---------|------|
| **環境の再現性** | 開発・本番で同じ環境を使用可能 |
| **依存関係の分離** | システムの Python と競合しない |
| **スケーラビリティ** | コンテナオーケストレーション対応 |
| **デプロイの簡素化** | イメージを配布するだけで完了 |

---

## 共通設定

### Dockerfile の基本構成

各プロジェクトの Dockerfile は以下の構成を基本としています：

```dockerfile
# ベースイメージ: Python 3.11 軽量版
FROM python:3.11-slim

# ビルドに必要なツールをインストール
# - build-essential: C/C++ コンパイラ（一部パッケージのビルドに必要）
# - gcc: GNU C コンパイラ
# - curl: ヘルスチェック用
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 依存パッケージをインストール
# 先にコピーすることでキャッシュを活用
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションをコピー
WORKDIR /app
COPY . .

# 環境変数の設定
# PYTHONPATH: モジュール検索パスに /app を追加
# PYTHONUNBUFFERED: 出力をバッファリングしない（ログ即時表示）
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Streamlit のデフォルトポート
EXPOSE 8501

# ヘルスチェック
# Streamlit の内部ヘルスエンドポイントを確認
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# データ永続化用ボリューム
# これらのディレクトリはホストからマウント可能
VOLUME ["/app/input", "/app/reference", "/app/output", "/app/prompt"]

# エントリーポイント
ENTRYPOINT ["python"]
CMD ["main.py"]
```

### Dockerfile の各命令の解説

| 命令 | 説明 |
|------|------|
| `FROM` | ベースイメージを指定。`slim` 版は軽量で推奨 |
| `RUN` | コンテナ内でコマンドを実行 |
| `COPY` | ホストからコンテナにファイルをコピー |
| `WORKDIR` | 作業ディレクトリを設定 |
| `ENV` | 環境変数を設定 |
| `EXPOSE` | 公開するポートを宣言（実際の公開は `-p` オプション） |
| `HEALTHCHECK` | コンテナのヘルスチェック方法を定義 |
| `VOLUME` | マウントポイントを宣言 |
| `ENTRYPOINT` | コンテナ起動時に実行するコマンド |
| `CMD` | ENTRYPOINT への引数またはデフォルトコマンド |

---

## プロジェクト別デプロイ

### rag-batch

バッチ処理に特化したプロジェクト。Excel ファイルの一括処理に最適です。

#### ビルド

```bash
cd rag-batch

# イメージをビルド
# -t: タグ名を指定
docker build -t rag-batch:latest .

# ビルド時にキャッシュを使わない場合
docker build --no-cache -t rag-batch:latest .

# ビルドログを詳細に表示
docker build --progress=plain -t rag-batch:latest .
```

#### バッチモード実行

Excel ファイルを一括処理します。

```bash
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/reference:/app/reference \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/prompt:/app/prompt \
  -e ANTHROPIC_API_KEY=your_key \
  rag-batch:latest main.py
```

**オプション解説：**

| オプション | 説明 |
|-----------|------|
| `--rm` | 終了後にコンテナを自動削除 |
| `-v` | ボリュームマウント（ホスト:コンテナ） |
| `-e` | 環境変数を設定 |

#### インタラクティブモード実行

Streamlit UI で対話的に質問応答できます。

```bash
docker run -p 8501:8501 \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/reference:/app/reference \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/prompt:/app/prompt \
  -e ANTHROPIC_API_KEY=your_key \
  rag-batch:latest main.py interactive
```

**オプション解説：**

| オプション | 説明 |
|-----------|------|
| `-p 8501:8501` | ポートフォワーディング（ホスト:コンテナ） |

ブラウザで http://localhost:8501 にアクセス。

---

### rag-gemini

Google Gemini Embedding と ChromaDB を使用する最新版。

#### ビルド

```bash
cd rag-gemini
docker build -t rag-gemini:latest .
```

#### バッチモード実行

```bash
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/reference:/app/reference \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/gemini_credentials.json:/app/gemini_credentials.json:ro \
  -e GEMINI_CREDENTIALS_PATH=/app/gemini_credentials.json \
  -e GEMINI_PROJECT_ID=your-project-id \
  rag-gemini:latest main.py
```

**オプション解説：**

| オプション | 説明 |
|-----------|------|
| `:ro` | 読み取り専用でマウント（セキュリティ向上） |

#### インタラクティブモード実行

```bash
docker run -p 8501:8501 \
  -v $(pwd)/reference:/app/reference \
  -v $(pwd)/gemini_credentials.json:/app/gemini_credentials.json:ro \
  -e GEMINI_CREDENTIALS_PATH=/app/gemini_credentials.json \
  -e GEMINI_PROJECT_ID=your-project-id \
  rag-gemini:latest bash -c "streamlit run ui/chat.py --server.address 0.0.0.0"
```

**ポイント：**
- `--server.address 0.0.0.0`: コンテナ外からのアクセスを許可
- `bash -c`: 複数コマンドを実行する場合に使用

---

### rag-streamlit

Streamlit UI を備えた対話型システム。

#### ビルド

```bash
cd rag-streamlit
docker build -t rag-streamlit:latest .
```

#### 対話モード実行

```bash
docker run -p 8501:8501 \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/reference:/app/reference \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/prompt:/app/prompt \
  --env-file .env \
  rag-streamlit:latest main.py interactive
```

**オプション解説：**

| オプション | 説明 |
|-----------|------|
| `--env-file .env` | ファイルから環境変数を読み込み |

#### バッチモード実行

```bash
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/reference:/app/reference \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/prompt:/app/prompt \
  --env-file .env \
  rag-streamlit:latest main.py
```

---

## 共通のボリュームマウント

すべてのプロジェクトで共通のディレクトリ構成を使用します。

| ボリューム | 用途 | 必須 |
|-----------|------|------|
| `/app/input` | 入力 Excel ファイル | ○（バッチ） |
| `/app/reference` | 参照データ + ベクトルキャッシュ | ○ |
| `/app/output` | 出力ファイル | ○（バッチ） |
| `/app/prompt` | LLM プロンプトテンプレート | △ |

### ディレクトリ構成の準備

```bash
# 必要なディレクトリを作成
mkdir -p input reference output prompt

# 参照データを配置
cp your_reference.xlsx reference/

# 入力データを配置
cp your_input.xlsx input/

# プロンプトテンプレートを配置
cp your_prompt.txt prompt/
```

---

## 環境変数の管理

### 方法1: コマンドラインで指定

```bash
docker run \
  -e ANTHROPIC_API_KEY=sk-ant-xxx \
  -e OPENAI_API_KEY=sk-xxx \
  ...
```

### 方法2: .env ファイルを使用（推奨）

```bash
# .env ファイルを作成
cat > .env << 'EOF'
ANTHROPIC_API_KEY=sk-ant-xxx
OPENAI_API_KEY=sk-xxx
GEMINI_PROJECT_ID=your-project-id
EOF

# .env ファイルを使用して実行
docker run --env-file .env ...
```

### 方法3: Docker Compose を使用

```yaml
# docker-compose.yml
version: '3.8'
services:
  rag:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./input:/app/input
      - ./reference:/app/reference
      - ./output:/app/output
    env_file:
      - .env
```

```bash
# 起動
docker-compose up

# バックグラウンドで起動
docker-compose up -d

# 停止
docker-compose down
```

---

## トラブルシューティング

### ポート競合

**症状：**
```
Error: bind: address already in use
```

**解決策：**

```bash
# 使用中のポートを確認
# Linux/Mac
lsof -i :8501

# Windows
netstat -ano | findstr :8501

# 別ポートで起動
docker run -p 8502:8501 ...
```

### ボリュームの権限問題

**症状：**
```
PermissionError: [Errno 13] Permission denied
```

**解決策：**

```bash
# ホスト側のディレクトリに書き込み権限を付与
chmod -R 777 input output reference

# または特定のユーザーで実行
docker run --user $(id -u):$(id -g) ...
```

### ヘルスチェック失敗

**症状：**
```
Container is unhealthy
```

**解決策：**

```bash
# コンテナログを確認
docker logs <container_id>

# コンテナ内でシェルを起動してデバッグ
docker exec -it <container_id> bash

# ヘルスチェックエンドポイントを手動確認
curl http://localhost:8501/_stcore/health
```

### メモリ不足

**症状：**
```
Killed
Container exited with code 137
```

**解決策：**

```bash
# メモリ制限を増やす
docker run --memory=4g ...

# または Docker Desktop の設定でメモリを増やす
# Settings → Resources → Memory
```

### イメージのビルドが遅い

**解決策：**

```bash
# ビルドキャッシュを活用
# requirements.txt を先にコピーして変更がなければキャッシュを使用

# .dockerignore を作成して不要ファイルを除外
cat > .dockerignore << 'EOF'
__pycache__
*.pyc
.venv
venv
.git
.env
*.log
input/
output/
reference/
EOF
```

### コンテナが起動しない

**デバッグ手順：**

```bash
# 1. ビルドログを確認
docker build --progress=plain -t rag:debug .

# 2. コンテナを対話モードで起動
docker run -it --entrypoint bash rag:debug

# 3. コンテナ内で手動実行
python main.py

# 4. 依存関係を確認
pip list
```

---

## 本番環境向け設定

### マルチステージビルド

イメージサイズを削減するためのベストプラクティス：

```dockerfile
# ビルドステージ
FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# 実行ステージ
FROM python:3.11-slim AS runner

WORKDIR /app

# ビルドステージから依存パッケージのみコピー
COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python"]
CMD ["main.py"]
```

### セキュリティ強化

```dockerfile
# 非 root ユーザーで実行
RUN useradd -m -u 1000 appuser
USER appuser

# 読み取り専用ファイルシステム（可能な場合）
# docker run --read-only ...

# Capabilities を制限
# docker run --cap-drop=ALL ...
```

### リソース制限

```bash
# CPU とメモリを制限
docker run \
  --cpus=2 \
  --memory=4g \
  --memory-swap=4g \
  ...
```

---

## Docker Compose 設定例

複数サービスを管理する場合：

```yaml
# docker-compose.yml
version: '3.8'

services:
  rag-gemini:
    build:
      context: ./rag-gemini
      dockerfile: Dockerfile
    ports:
      - "8501:8501"
    volumes:
      - ./data/reference:/app/reference
      - ./data/input:/app/input
      - ./data/output:/app/output
      - ./gemini_credentials.json:/app/gemini_credentials.json:ro
    env_file:
      - .env
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  rag-batch:
    build:
      context: ./rag-batch
      dockerfile: Dockerfile
    volumes:
      - ./data/reference:/app/reference
      - ./data/input:/app/input
      - ./data/output:/app/output
    env_file:
      - .env
    profiles:
      - batch  # docker-compose --profile batch up で起動

volumes:
  reference:
  input:
  output:
```

**使用方法：**

```bash
# すべてのサービスを起動
docker-compose up -d

# 特定のサービスのみ起動
docker-compose up rag-gemini

# バッチ処理を実行
docker-compose --profile batch run rag-batch

# ログを確認
docker-compose logs -f

# 停止
docker-compose down
```

---

## よく使うコマンド集

```bash
# イメージ一覧
docker images

# コンテナ一覧（実行中）
docker ps

# コンテナ一覧（すべて）
docker ps -a

# コンテナのログ
docker logs <container_id>

# コンテナ内でシェル起動
docker exec -it <container_id> bash

# 停止中のコンテナを削除
docker container prune

# 未使用イメージを削除
docker image prune

# すべての未使用リソースを削除
docker system prune -a
```
