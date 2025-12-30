# タスク: rag-gemini 実行可能化修正

## 目的
rag-geminiプロジェクトを実行可能な状態にする。現在、依存関係の不整合とDB選択ロジックの問題により実行不可。

## 実装要件
1. requirements.txtのLangChain関連コメントを解除して依存関係を有効化
2. input_handler.pyの各InputHandlerクラスにcurrent_file属性を追加
3. reference/scenario/フォルダを作成
4. README.mdのパス名とコマンドを実コードに合わせて修正

---

## 変更対象ファイル

### 1. requirements.txt
**変更内容**: Line 6-10のコメントを解除してLangChain依存関係を有効化

**現在のコード**:
```
# LangChain関連を削除（タグレス対応）
# langchain>=0.1.0
# langchain-anthropic>=0.0.1
# langchain-openai>=0.0.1
# langchain-google-genai>=0.0.1
```

**実装すべきコード**:
```
# LangChain関連（LLM拡張検索機能用）
langchain>=0.1.0
langchain-anthropic>=0.0.1
langchain-openai>=0.0.1
langchain-google-genai>=0.0.1
```

---

### 2. src/handlers/input_handler.py
**変更内容**: ExcelInputHandlerとMultiFolderInputHandlerのload_data()メソッドでself.current_fileを設定

**ExcelInputHandler.load_data() - 現在のコード** (Line 58-74):
```python
def load_data(self) -> list:
    input_file = self._get_latest_file(self.input_dir, "*.xlsx")
    logger.info(f"Processing input file: {os.path.basename(input_file)}")
    input_df = pd.read_excel(input_file)
```

**ExcelInputHandler.load_data() - 実装すべきコード**:
```python
def load_data(self) -> list:
    input_file = self._get_latest_file(self.input_dir, "*.xlsx")
    self.current_file = os.path.basename(input_file)  # 追加: DB選択用
    logger.info(f"Processing input file: {self.current_file}")
    input_df = pd.read_excel(input_file)
```

**MultiFolderInputHandler.load_data() - 現在のコード** (Line 318-335):
```python
def load_data(self) -> list:
    input_file = self._get_latest_file(self.input_dir, "*.xlsx")
    logger.info(f"Processing input file: {os.path.basename(input_file)}")
    input_df = pd.read_excel(input_file)
```

**MultiFolderInputHandler.load_data() - 実装すべきコード**:
```python
def load_data(self) -> list:
    input_file = self._get_latest_file(self.input_dir, "*.xlsx")
    self.current_file = os.path.basename(input_file)  # 追加: DB選択用
    logger.info(f"Processing input file: {self.current_file}")
    input_df = pd.read_excel(input_file)
```

---

### 3. reference/scenario/ フォルダ
**変更内容**: 空のフォルダを作成（将来のシナリオデータ用）

---

### 4. README.md
**変更内容**: パス名とコマンドを実コードに合わせて更新

| 変更前 | 変更後 |
|--------|--------|
| reference/マージシナリオ/ | reference/scenario/ |
| reference/履歴データ/ | reference/faq_data/ |
| reference/vector_cache/ | reference/vector_db/ |
| streamlit run chat.py | streamlit run ui/chat.py |

---

## 依存関係
- searcher.py: LangChainのChatAnthropic, ChatOpenAIをimport（Line 12-16）
- processor.py: input_handler.current_fileを参照（Line 47）
- dynamic_db_manager.py: reference/scenario, reference/faq_dataを参照（Line 28-29）

## 制約
- 既存の動作を壊さないこと
- LLM拡張検索はconfig.pyでsearch_mode="llm_enhanced"の時のみ使用

## 完了条件
- [ ] requirements.txtのLangChain依存関係が有効化されている
- [ ] ExcelInputHandler.load_data()でself.current_fileが設定される
- [ ] MultiFolderInputHandler.load_data()でself.current_fileが設定される
- [ ] reference/scenario/フォルダが存在する
- [ ] README.mdのパス名が実コードと一致している
- [ ] pip install -r requirements.txtが成功する
