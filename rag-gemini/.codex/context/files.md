# 関連ファイル一覧

## 1. requirements.txt (修正対象)

```
pandas>=2.0.0
numpy>=1.24.0
sentence-transformers>=2.2.0
scikit-learn>=1.3.0
torch>=2.0.0
# LangChain関連を削除（タグレス対応）
# langchain>=0.1.0
# langchain-anthropic>=0.0.1
# langchain-openai>=0.0.1
# langchain-google-genai>=0.0.1
google-generativeai>=0.3.0
google-cloud-aiplatform>=1.35.0
google-auth>=2.17.0
...
```

---

## 2. src/handlers/input_handler.py (修正対象)

### ExcelInputHandler.load_data() (Line 58-74)
```python
class ExcelInputHandler(InputHandler):
    def load_data(self) -> list:
        input_file = self._get_latest_file(self.input_dir, "*.xlsx")
        logger.info(f"Processing input file: {os.path.basename(input_file)}")
        input_df = pd.read_excel(input_file)

        # 列名チェックとデータ抽出
        number_col, query_col, answer_col = self._get_column_names(input_df)
        valid_input_df = input_df.dropna(subset=[query_col])

        data = []
        for _, row in valid_input_df.iterrows():
            data.append({
                "number": str(row[number_col]),
                "query": str(row[query_col]),
                "answer": str(row[answer_col]) if answer_col and pd.notna(row[answer_col]) else ""
            })
        return data
```

### MultiFolderInputHandler.load_data() (Line 318-335)
```python
class MultiFolderInputHandler(InputHandler):
    """複数フォルダから参照データを読み込むハンドラー"""
    
    def load_data(self) -> list:
        # 入力データの読み込み（従来通り）
        input_file = self._get_latest_file(self.input_dir, "*.xlsx")
        logger.info(f"Processing input file: {os.path.basename(input_file)}")
        input_df = pd.read_excel(input_file)

        # 列名チェックとデータ抽出
        number_col, query_col, answer_col = self._get_column_names(input_df)
        valid_input_df = input_df.dropna(subset=[query_col])

        data = []
        for _, row in valid_input_df.iterrows():
            data.append({
                "number": str(row[number_col]),
                "query": str(row[query_col]),
                "answer": str(row[answer_col]) if answer_col and pd.notna(row[answer_col]) else ""
            })
        return data
```

---

## 3. src/core/processor.py (参照元)

### Line 47付近
```python
# 入力ファイル名を取得（動的DB選択用）
input_file = getattr(self.input_handler, 'current_file', None)

results = self.searcher.search(query_number, query_text, original_answer, input_file)
```

---

## 4. src/core/searcher.py (依存関係)

### Line 12-16 (LangChain import)
```python
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
import vertexai
from google.oauth2 import service_account
from langchain.schema import HumanMessage, SystemMessage
```

---

## 5. src/utils/dynamic_db_manager.py (参照)

### Line 28-29
```python
self.reference_faq_path = os.path.join(config.base_dir, "reference", "faq_data")
self.reference_scenario_path = os.path.join(config.base_dir, "reference", "scenario")
```

---

## 6. README.md (修正対象)

### 修正が必要な箇所

Line 118-120:
```markdown
#### マージ版シナリオ（階層構造Excel）
- **配置場所**: `reference/マージシナリオ/`
```
→ `reference/scenario/` に変更

Line 167-169:
```markdown
#### 履歴データ（従来形式）
- **配置場所**: `reference/履歴データ/`
```
→ `reference/faq_data/` に変更

Line 120:
```markdown
- **ベクトルキャッシュディレクトリ**: `reference/vector_cache/`
```
→ `reference/vector_db/` に変更

Line 198-199:
```markdown
   streamlit run chat.py
```
→ `streamlit run ui/chat.py` に変更
