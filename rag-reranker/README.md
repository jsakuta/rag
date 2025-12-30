# rag-reranker

> ⚠️ **DEPRECATED**: このプロジェクトは**メンテナンス終了**しています。新規利用は推奨されません。

## 代替プロジェクト

- **バッチ処理**: [rag-batch](../rag-batch/)
- **最新技術**: [rag-gemini](../rag-gemini/)
- **UI版**: [rag-streamlit](../rag-streamlit/)

## 概要

Cross-Encoderリランキングを使用したRAG実装。Azure Table Storageとの連携機能を含む。

## 主な機能

- ✅ **Cross-Encoderリランキング**: `hotchpotch/japanese-reranker-cross-encoder-large-v1`
- ✅ Azure Table Storageへのデータ保存
- ✅ 複数LLMプロバイダー対応（OpenAI, Anthropic, Google）
- ✅ PDF/Excel両対応
- ✅ 対話型CLIインターフェース

## 技術スタック

- Python 3.7+
- SentenceTransformer (`intfloat/multilingual-e5-base`)
- Azure Table Storage
- OpenAI/Anthropic/Google API

## システムコンポーネント

| ファイル | 説明 |
|---------|------|
| `bot.py` | メインボット実装 |
| `azure_rag.py` | Azure連携機能 |
| `LoanAssistantBot.py` | ローンアシスタントボット |
| `LoanAssistantBot_Batch.py` | バッチ処理版 |
| `text_analyzer.py` | テキスト分析ユーティリティ |
| `PDF.py` | PDF処理 |
| `main.py` | エントリーポイント |
| `config.py` | 設定管理 |

## セットアップ（参考）

```bash
cd old

# 仮想環境作成
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存関係インストール
pip install -r requirements.txt

# 環境変数設定
cp .env.example .env
# .envファイルを編集

# 実行
python main.py
```

## 環境変数

`.env.example`を参照してください。

必要な環境変数:
- `OPENAI_API_KEY`: OpenAI APIキー
- `ANTHROPIC_API_KEY`: Anthropic APIキー
- `GOOGLE_API_KEY`: Google APIキー
- `AZURE_STORAGE_CONNECTION_STRING`: Azure Storage接続文字列
- `AZURE_TABLE_NAME`: Azure Tableテーブル名

## 非推奨理由

1. **古いコード構造**: メンテナンスが困難
2. **機能の改善**: 新プロジェクト（rag-batch, rag-gemini）で機能が大幅に改善
3. **Azure依存**: Azure Table Storageへの依存が強く、柔軟性に欠ける
4. **スケーラビリティ**: 大規模データに対応していない

## マイグレーションガイド

### rag_v1.0 (rag-batch)への移行

```bash
cd ../rag_v1.0

# 類似の機能:
# - ハイブリッド検索（ベクトル + キーワード）
# - バッチ処理
# - Excel入出力

# 主な違い:
# - リランキング機能なし → 必要な場合はoldを参照
# - Azure依存なし → JSONキャッシュ使用
```

### rag_v2.1 (rag-gemini)への移行

```bash
cd ../rag_v2.1

# 類似の機能:
# - ハイブリッド検索
# - LLM拡張検索
# - 高精度検索

# 主な違い:
# - Gemini埋め込みモデル使用
# - ChromaDB永続化
# - 動的DB管理システム
```

## ライセンス

MIT License

## 参考資料

- [LangChain Documentation](https://python.langchain.com/)
- [SentenceTransformers](https://www.sbert.net/)
- [Azure Table Storage](https://azure.microsoft.com/services/storage/tables/)
