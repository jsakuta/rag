# ベースイメージとしてPython 3.11を使用（現在の.venvと合わせる）
FROM python:3.11-slim

# システムパッケージの更新とビルドツールのインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリを設定
WORKDIR /app

# Poetry の設定（依存関係管理の改善）
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VIRTUALENVS_CREATE=false
ENV PATH="$POETRY_HOME/bin:$PATH"

# 必要なファイルをコピー
COPY requirements.txt .
COPY ./prompt ./prompt
COPY ./config.py .
COPY ./main.py .
COPY ./processor.py .
COPY ./search.py .
COPY ./ui.py .
COPY ./utils.py .

# 依存関係のインストール
RUN pip install --no-cache-dir -r requirements.txt

# 必要なディレクトリを作成
RUN mkdir -p input reference output

# 環境変数を設定
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# ポート設定（Streamlit用）
EXPOSE 8501

# ヘルスチェック
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501 || exit 1

# ボリュームの設定
VOLUME ["/app/input", "/app/reference", "/app/output", "/app/prompt"]

# 実行コマンドを設定
ENTRYPOINT ["python"]
CMD ["main.py"]
