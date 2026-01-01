# トラブルシューティング

RAG システム共通のトラブルシューティングガイドです。

---

## よくある問題と解決策

### API 関連

#### API キーエラー

**症状:**
```
Error: Invalid API key provided
AuthenticationError: Incorrect API key
```

**原因:**
- `.env` ファイルが存在しない
- API キーが正しく設定されていない
- API キーの有効期限が切れている

**解決策:**

1. `.env` ファイルの存在確認
   ```bash
   ls -la .env
   ```

2. `.env.example` からコピーして作成
   ```bash
   cp .env.example .env
   ```

3. API キーを設定
   ```bash
   # .env ファイルを編集
   ANTHROPIC_API_KEY=sk-ant-your_key_here
   OPENAI_API_KEY=sk-your_key_here
   ```

4. キーの形式を確認
   - Anthropic: `sk-ant-` で始まる
   - OpenAI: `sk-` で始まる
   - Google: プロジェクト ID または API キー

---

#### レート制限エラー

**症状:**
```
RateLimitError: Rate limit exceeded
Error 429: Too Many Requests
```

**原因:**
- 短時間に大量の API 呼び出しを実行
- API プランの制限に達した

**解決策:**

1. バッチサイズを縮小
   ```python
   # config.py
   BATCH_SIZE = 5  # デフォルトの 10 から縮小
   ```

2. リクエスト間に待機時間を追加
   ```python
   import time
   time.sleep(1)  # 1秒待機
   ```

3. 指数バックオフを実装
   ```python
   import time
   for attempt in range(5):
       try:
           response = api_call()
           break
       except RateLimitError:
           wait_time = 2 ** attempt
           time.sleep(wait_time)
   ```

4. API プランのアップグレードを検討

---

#### 認証エラー（Gemini / Google Cloud）

**症状:**
```
DefaultCredentialsError: Could not automatically determine credentials
PermissionDenied: 403 The caller does not have permission
```

**原因:**
- `gemini_credentials.json` が存在しない
- サービスアカウントの権限が不足
- プロジェクト ID が未設定

**解決策:**

1. 認証ファイルの配置確認
   ```bash
   ls -la gemini_credentials.json
   ```

2. 環境変数の設定
   ```bash
   # .env
   GEMINI_CREDENTIALS_PATH=gemini_credentials.json
   GEMINI_PROJECT_ID=your-project-id
   ```

3. サービスアカウントの権限確認
   - Google Cloud Console → IAM & Admin → Service Accounts
   - 必要な権限:
     - `Vertex AI User`
     - `AI Platform Admin`

4. 認証のテスト
   ```bash
   gcloud auth application-default login
   ```

---

### データ関連

#### 列名エラー

**症状:**
```
KeyError: '問合せ内容'
ValueError: Column '回答' not found
```

**原因:**
- 参照ファイルの列名が期待と異なる
- 全角/半角の違い
- 前後のスペース

**解決策:**

1. 期待される列名を確認
   ```
   必須列: 「問合せ内容」「回答」
   ```

2. Excel ファイルを開いて列名を確認
   - 前後のスペースを削除
   - 全角に統一

3. コードで列名を柔軟に処理
   ```python
   # 列名を正規化
   df.columns = df.columns.str.strip()
   ```

4. 列名のマッピングを追加
   ```python
   COLUMN_MAPPING = {
       '質問': '問合せ内容',
       'Question': '問合せ内容',
       'Answer': '回答'
   }
   df = df.rename(columns=COLUMN_MAPPING)
   ```

---

#### メモリエラー

**症状:**
```
MemoryError: Unable to allocate memory
Killed (out of memory)
```

**原因:**
- 大量のデータを一度に処理
- ベクトルキャッシュが大きすぎる
- システムメモリ不足

**解決策:**

1. データを分割して処理
   ```python
   # 1000行ずつ処理
   chunk_size = 1000
   for i in range(0, len(df), chunk_size):
       process(df[i:i+chunk_size])
   ```

2. 不要な変数を削除
   ```python
   del large_variable
   import gc
   gc.collect()
   ```

3. 64bit Python を使用
   ```bash
   python --version  # 確認
   ```

4. スワップ領域を増やす（Linux）
   ```bash
   sudo fallocate -l 8G /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

---

#### 文字化け

**症状:**
```
文字が「????」や「□」で表示される
UnicodeDecodeError: 'utf-8' codec can't decode byte
```

**原因:**
- ファイルが UTF-8 以外でエンコードされている
- BOM（Byte Order Mark）の問題

**解決策:**

1. ファイルのエンコーディングを確認
   ```bash
   file -i your_file.xlsx
   ```

2. UTF-8 で保存し直す
   - Excel: 名前を付けて保存 → CSV UTF-8

3. Python で読み込み時にエンコーディング指定
   ```python
   # エンコーディングを指定
   df = pd.read_excel('file.xlsx', encoding='utf-8')

   # または自動検出
   import chardet
   with open('file.csv', 'rb') as f:
       encoding = chardet.detect(f.read())['encoding']
   df = pd.read_csv('file.csv', encoding=encoding)
   ```

---

### モデル関連

#### 初回起動が遅い

**症状:**
- 初回実行時に 5-10 分かかる
- 「Downloading model...」と表示される

**原因:**
- 埋め込みモデルのダウンロード中
- Hugging Face からモデルを取得

**解決策:**

1. 事前にモデルをダウンロード
   ```python
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer('intfloat/multilingual-e5-base')
   ```

2. モデルキャッシュの確認
   ```bash
   ls ~/.cache/huggingface/
   ```

3. オフライン環境では事前にモデルを配置
   ```bash
   export TRANSFORMERS_OFFLINE=1
   export HF_DATASETS_OFFLINE=1
   ```

---

#### GPU が使用されない

**症状:**
- 処理が非常に遅い
- `torch.cuda.is_available()` が `False`

**原因:**
- PyTorch の CPU 版がインストールされている
- CUDA ドライバがインストールされていない

**解決策:**

1. GPU の確認
   ```bash
   nvidia-smi
   ```

2. PyTorch CUDA 版のインストール
   ```bash
   # CUDA 11.8 の場合
   pip uninstall torch
   pip install torch --index-url https://download.pytorch.org/whl/cu118

   # CUDA 12.1 の場合
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```

3. インストール確認
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"Device: {torch.cuda.get_device_name(0)}")
   ```

---

### キャッシュ・DB 関連

#### キャッシュエラー

**症状:**
```
FileNotFoundError: No such file or directory: 'reference/vector_cache/'
JSONDecodeError: Expecting value
```

**原因:**
- キャッシュディレクトリが存在しない
- キャッシュファイルが破損

**解決策:**

1. ディレクトリの作成
   ```bash
   mkdir -p reference/vector_cache
   ```

2. キャッシュのクリア
   ```bash
   rm reference/vector_cache/cache*.json
   ```

3. 権限の確認
   ```bash
   chmod -R 755 reference/
   ```

---

#### ChromaDB エラー

**症状:**
```
sqlite3.OperationalError: database is locked
ChromaDB: Collection not found
```

**原因:**
- vector_db/ ディレクトリの破損
- 複数プロセスからの同時アクセス
- ディスク容量不足

**解決策:**

1. DB の再生成
   ```bash
   rm -rf reference/vector_db/
   # 再実行時に自動生成される
   ```

2. 単一プロセスでの実行を確認
   ```bash
   # 他のプロセスがないか確認
   ps aux | grep python
   ```

3. ディスク容量の確認
   ```bash
   df -h
   ```

---

### Streamlit 関連

#### ポート使用中

**症状:**
```
OSError: [Errno 98] Address already in use
Streamlit cannot start because port 8501 is already in use
```

**原因:**
- 別の Streamlit インスタンスが起動中
- 前回のプロセスが残っている

**解決策:**

1. 使用中のプロセスを確認・終了
   ```bash
   # Linux/Mac
   lsof -i :8501
   kill -9 <PID>

   # Windows
   netstat -ano | findstr :8501
   taskkill /PID <PID> /F
   ```

2. 別ポートで起動
   ```bash
   streamlit run ui/chat.py --server.port 8502
   ```

---

#### No prompt files found

**症状:**
```
FileNotFoundError: No prompt files found in prompt/
ValueError: Prompt directory is empty
```

**原因:**
- prompt/ ディレクトリが空
- プロンプトファイルが存在しない

**解決策:**

1. プロンプトファイルの配置
   ```bash
   mkdir -p prompt
   cp examples/summarize_v1.0.txt prompt/
   ```

2. デフォルトプロンプトの作成
   ```bash
   cat > prompt/default.txt << 'EOF'
   以下の参考情報を元に、質問に回答してください。

   # 参考情報
   {context}

   # 質問
   {question}

   # 回答
   EOF
   ```

---

## ログの確認方法

### リアルタイムログ

```bash
# ログファイルをリアルタイム表示
tail -f logs/app.log

# 最新100行を表示
tail -100 logs/app.log
```

### エラーのみ表示

```bash
# ERROR レベルのみ
grep ERROR logs/app.log

# WARNING 以上
grep -E "(WARNING|ERROR|CRITICAL)" logs/app.log
```

### デバッグレベル有効化

```bash
# 環境変数で設定
export LOG_LEVEL=DEBUG
python main.py

# または config.py で設定
LOG_LEVEL = "DEBUG"
```

### ログローテーション

```python
# logging 設定例
import logging
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(
    'logs/app.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
```

---

## キャッシュのクリア

### rag-batch / rag-streamlit（JSON キャッシュ）

```bash
# 全キャッシュをクリア
rm reference/vector_cache/cache*.json

# 特定のキャッシュのみ
rm reference/vector_cache/cache_embeddings.json
```

### rag-gemini（ChromaDB）

```bash
# ChromaDB を完全にクリア
rm -rf reference/vector_db/

# コレクション単位でクリア（Python）
import chromadb
client = chromadb.PersistentClient(path="reference/vector_db")
client.delete_collection("qa_collection")
```

### Python キャッシュ

```bash
# __pycache__ を削除
find . -type d -name "__pycache__" -exec rm -rf {} +

# .pyc ファイルを削除
find . -type f -name "*.pyc" -delete
```

---

## パフォーマンス改善

### メモリ使用量削減

```python
# データ型の最適化
df['score'] = df['score'].astype('float32')  # float64 → float32

# 不要な列を削除
df = df[['問合せ内容', '回答']]

# チャンク処理
for chunk in pd.read_excel('large_file.xlsx', chunksize=1000):
    process(chunk)
```

### 検索速度向上

```python
# top_k を調整
config.TOP_K = 3  # デフォルト 4 から削減

# キャッシュを有効活用
config.USE_CACHE = True
config.CACHE_TTL = 3600  # 1時間
```

---

## プロジェクト固有の問題

プロジェクト固有の問題については、各プロジェクトの README を参照してください：

- [rag-gemini/README.md](../rag-gemini/README.md#トラブルシューティング)
- [rag-batch/README.md](../rag-batch/README.md#トラブルシューティング)
- [rag-streamlit/README.md](../rag-streamlit/README.md#トラブルシューティング)
