# レビューコンテキスト

## 変更の背景

- 2025-12-30にフォルダのリネーム・移動を実施
- rag-geminiは改善済みだが、他のフォルダの確認が必要だった
- 調査の結果、rag-batch/config.py にインポートパスの問題を発見

## 変更内容

### 1. rag-batch/config.py のインポートパス修正

**変更前**: `from utils.logger import setup_logger`
**変更後**: `from src.utils.logger import setup_logger`

**理由**:
- main.py や他のファイルは `src.utils.logger` を使用している
- config.py だけが `utils.logger` となっており、インポートパスの統一性がなかった
- 実行時に `ModuleNotFoundError` が発生する可能性があった

### 2. rag-batch/input/ フォルダの作成

**理由**:
- `input_handler.py:13` で `self.input_dir = os.path.join(config.base_dir, "input")` として参照
- フォルダが存在しないと `FileNotFoundError` が発生する

## 影響範囲

- rag-batch のみ
- 他のフォルダ（rag-streamlit, rag-reranker, rag-gemini）には影響なし

## 確認済み事項

- rag-streamlit: 構造OK、インポートOK
- rag-reranker: 構造OK（Deprecated）
- rag-gemini: 改善済み
