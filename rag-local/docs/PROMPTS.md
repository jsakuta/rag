# プロンプトファイル一覧

このディレクトリには、LLMに送信するプロンプトテンプレートが格納されています。

## AI使用箇所マップ

```
入力
  │
  ├─ ベクトル化 ─────────────────────────────────── [Embedding Model]
  │   └─ azure_openai: text-embedding-3-large
  │   └─ vertex_ai: gemini-embedding-001
  │
  ├─ クエリ拡張 ─────────────────────────────────── [LLM]
  │   └─ prompt/summarize_v1.0.txt
  │   └─ 呼び出し元: src/core/searcher.py, src/core/search/query_enhancer.py
  │
  ├─ 検索 ───────────────────────────────────────── [ベクトルDB]
  │   └─ ChromaDB
  │
  └─ 関連性判定 ─────────────────────────────────── [LLM]
      └─ prompt/judgment_support.txt
      └─ 呼び出し元: src/core/judgment_support.py
```

## プロンプトファイル詳細

| ファイル | 用途 | 呼び出し元 | AIモデル |
|---------|------|-----------|---------|
| `summarize_v1.0.txt` | 検索クエリ拡張 | `searcher.py`, `query_enhancer.py` | LLM (gemini-2.5-flash-lite) |
| `judgment_support.txt` | 関連性判定（関連あり/要確認/無関係） | `judgment_support.py` | LLM (gemini-2.5-flash-lite) |

## 各プロンプトの説明

### summarize_v1.0.txt（クエリ拡張）

ユーザーの質問文を、ベクトル検索に最適化された検索クエリに変換します。

**入力**: ユーザーの質問文
**出力**: 検索クエリ（キーワード列挙形式）

```
検索クエリ: 現金処理 誤操作 WAVE 700 PRO 有高調整 手続き
```

### judgment_support.txt（関連性判定）

改定内容と検索結果（既存FAQ）の関連性を3段階で判定します。

**入力**: 改定内容、検索結果（質問・回答）
**出力**:
```
関連性: 関連あり / 要確認 / 明らかに無関係
根拠: 判定理由（1-2文）
```

## 設定

使用するLLMモデルは環境変数で設定します：

```
DEFAULT_LLM_PROVIDER=gemini
DEFAULT_LLM_MODEL=gemini-2.5-flash-lite
```

詳細は `.env.example` を参照してください。
