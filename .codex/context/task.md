# タスク: フォルダ移動の修正と動作確認

## 目的

rag-gemini改善後、他のフォルダ（rag-batch, rag-streamlit, rag-reranker）が正しく移動されているか確認し、必要な修正を行う。

## 背景

- 2025-12-30にフォルダのリネーム・移動を実施
- _archive/ に旧フォルダを保存、トップレベルに新名称でコピー
- rag-geminiは改善済み

## 発見された問題点

### 問題1: rag-batch/config.py のインポートパスが間違い

- **ファイル**: `rag-batch/config.py`
- **行**: 6
- **現在のコード**: `from utils.logger import setup_logger`
- **正しいコード**: `from src.utils.logger import setup_logger`

### 問題2: rag-batch/input/ フォルダがない

- **場所**: `rag-batch/input/`
- **対応**: 空フォルダを作成
- **理由**: `input_handler.py:13` で `self.input_dir = os.path.join(config.base_dir, "input")` として参照される

## 変更対象ファイル

### rag-batch/config.py

**変更内容**: インポートパスを `src.utils.logger` に修正

**現在のコード** (行6):
```python
from utils.logger import setup_logger
```

**修正後のコード**:
```python
from src.utils.logger import setup_logger
```

### rag-batch/input/

**変更内容**: 空フォルダを作成

## 動作確認対象

1. **rag-streamlit**: 問題なし → 動作確認のみ
2. **rag-batch**: 修正後に動作確認
3. **rag-reranker**: Deprecated → 参考確認

## 完了条件

- [ ] rag-batch/config.py のインポートパス修正完了
- [ ] rag-batch/input/ フォルダ作成完了
- [ ] rag-streamlit: `python main.py` が起動エラーなし
- [ ] rag-batch: `python main.py` が起動エラーなし
- [ ] rag-reranker: `python main.py` が起動エラーなし（Deprecated）

## 制約

- 既存の動作に影響を与えない
- インポートパスは `src.*` 形式に統一
