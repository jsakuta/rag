# RAG Q&A System Collection

> 段階的に進化した4つのRAG（Retrieval-Augmented Generation）Q&Aシステムのコレクション

## 📚 Projects Overview

| プロジェクト | 新名称（予定） | 用途 | 主な技術 | 推奨ユースケース |
|------------|-------------|------|---------|----------------|
| [old](./old/) | rag-reranker | レガシー版 | Reranker, Azure | 非推奨（参照のみ） |
| [RAG_yokin](./RAG_yokin/) | rag-streamlit | UI版 | Streamlit, E5 | 対話的検索UI |
| [rag_v1.0](./rag_v1.0/) | rag-batch | バッチ処理版 | ハイブリッド検索, Excel | 大量データ一括処理 |
| [rag_v2.1](./rag_v2.1/) | rag-gemini | Gemini統合版 | Vertex AI Gemini, ChromaDB | 高精度検索、最新技術 |

## 🎯 Which Project Should I Use?

### 🎨 For Interactive UI
→ **RAG_yokin (rag-streamlit)**
- Streamlitベースの対話的WebUI
- リアルタイム検索
- デモ・プレゼンテーションに最適

### 🏭 For Batch Processing
→ **rag_v1.0 (rag-batch)**
- Excel一括処理に最適化
- ハイブリッド検索（ベクトル + キーワード）
- Factory Pattern採用

### 🏢 For Enterprise & Latest Tech
→ **rag_v2.1 (rag-gemini)**
- Vertex AI Gemini埋め込みモデル
- ChromaDB永続化ベクトルDB
- LLM拡張検索（デュアルモード）
- 動的DB管理システム

### 🔰 For Legacy Reference
→ **old (rag-reranker)**
- **非推奨**: メンテナンス終了
- Cross-Encoderリランキング実装
- 参照用のみ

## 🌳 Evolution Tree

```
old (rag-reranker) - 基礎実装
 ├─→ RAG_yokin (rag-streamlit) - UI特化派生
 └─→ rag_v1.0 (rag-batch) - バッチ特化派生
      └─→ rag_v2.1 (rag-gemini) - エンタープライズ進化
```

## 🚀 Quick Start

### Common Prerequisites
- Python 3.7以上
- 仮想環境の作成推奨

### Setup Steps
```bash
# 1. リポジトリクローン
git clone <repository-url>
cd rag

# 2. プロジェクト選択（例: rag_v2.1）
cd rag_v2.1

# 3. 仮想環境作成
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 4. 依存関係インストール
pip install -r requirements.txt

# 5. 環境変数設定
cp .env.example .env
# .envファイルを編集してAPIキー等を設定

# 6. 実行
python main.py  # バッチモード
# または
python main.py interactive  # インタラクティブモード
```

## 📊 Feature Comparison

| 機能 | old | RAG_yokin | rag_v1.0 | rag_v2.1 |
|------|-----|-----------|----------|----------|
| **埋め込みモデル** | E5-base | E5-base | E5-base | **Gemini** |
| **ベクトルDB** | Pickle | JSON | JSON | **ChromaDB** |
| **リランキング** | ✅ Cross-Encoder | ❌ | ❌ | ❌ |
| **UI** | CLI | **Streamlit** | Streamlit | Streamlit |
| **動的DB管理** | ❌ | ❌ | ❌ | ✅ |
| **LLM拡張検索** | ❌ | ❌ | ✅ | ✅ (2モード) |
| **階層データ対応** | ✅ PDF | ❌ | ❌ | ✅ Excel |
| **本番運用** | ❌ | △ | ✅ | ✅ |

## 📖 Documentation

各プロジェクトの詳細は、各ディレクトリ内のREADME.mdを参照してください。

- [old/README.md](./old/README.md) - レガシー版（非推奨）
- [RAG_yokin/README.md](./RAG_yokin/README.md) - Streamlit UI版
- [rag_v1.0/README.md](./rag_v1.0/README.md) - バッチ処理版
- [rag_v2.1/README.md](./rag_v2.1/README.md) - Gemini統合版

## 🤝 Contributing

プロジェクトへの貢献を歓迎します。

## 📝 License

各プロジェクトのライセンスについては、個別のREADME.mdを参照してください。

## 📞 Contact

- GitHub: [@Jsakuta](https://github.com/Jsakuta)
- Repository: https://github.com/Jsakuta/RAG_yokin

## 🔄 Migration Plan

このリポジトリは段階的にリファクタリング中です：

- **Phase 1**: ドキュメント整備（完了予定）
- **Phase 2**: フォルダ構造統一（src/ディレクトリ化）
- **Phase 3**: プロジェクトリネーム
  - old → rag-reranker
  - RAG_yokin → rag-streamlit
  - rag_v1.0 → rag-batch
  - rag_v2.1 → rag-gemini

詳細は [C:\Users\SakutaJunki(作田隼樹)\.claude\plans\serene-gathering-rain.md](file://C:\Users\SakutaJunki(作田隼樹)\.claude\plans\serene-gathering-rain.md) を参照してください。
