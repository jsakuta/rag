# FAQデータDB読み込みバグ修正計画

## 問題の概要

`DynamicDBManager._vectorize_data()` が `latest_faq` を引数として受け取っているが、内部で使用されておらず、業務分野別DB構築時にFAQデータがベクトル化されない。

## 影響範囲

- `reference/faq_data/` 配下のFAQファイルがベクトルDBに登録されない
- 検索時に `search_source='history_data'` を指定しても結果が返らない可能性

## 修正対象ファイル

1. `src/utils/dynamic_db_manager.py`

## 修正内容

### 1. `_prepare_reference_data_for_vectorization()` の修正 (行830-864)

**現状:**
```python
def _prepare_reference_data_for_vectorization(self, latest_scenario: Optional[str] = None) -> dict:
```

**修正後:**
```python
def _prepare_reference_data_for_vectorization(
    self,
    latest_scenario: Optional[str] = None,
    latest_faq: Optional[str] = None
) -> dict:
```

- `latest_faq` パラメータを追加
- シナリオとFAQ両方を読み込んで統合するロジックを実装

### 2. `_vectorize_data()` の修正 (行621)

**現状:**
```python
reference_data = self._prepare_reference_data_for_vectorization(latest_scenario)
```

**修正後:**
```python
reference_data = self._prepare_reference_data_for_vectorization(latest_scenario, latest_faq)
```

## 詳細な修正ロジック

`_prepare_reference_data_for_vectorization()` を以下のロジックに変更:

```python
def _prepare_reference_data_for_vectorization(
    self,
    latest_scenario: Optional[str] = None,
    latest_faq: Optional[str] = None
) -> dict:
    """動的DB管理システム用の参照データ準備（業務分野フィルタリング対応）

    Args:
        latest_scenario: 読み込むシナリオファイル名
        latest_faq: 読み込むFAQファイル名

    Returns:
        dict: 参照データ（combined_texts, metadatas）
    """
    logger.info(f"参照データ準備開始 (シナリオ: {latest_scenario}, FAQ: {latest_faq})")

    # どちらも指定されていない場合は従来の動作
    if not latest_scenario and not latest_faq:
        from src.handlers.input_handler import MultiFolderInputHandler
        input_handler = MultiFolderInputHandler(self.config)
        return input_handler.load_reference_data()

    all_queries = []
    all_answers = []
    all_metadatas = []

    # シナリオデータの読み込み
    if latest_scenario:
        from src.handlers.input_handler import HierarchicalExcelInputHandler
        scenario_path = os.path.join(self.reference_scenario_path, latest_scenario)
        if os.path.exists(scenario_path):
            handler = HierarchicalExcelInputHandler(self.config, scenario_path)
            scenario_data = handler.load_reference_data()
            all_queries.extend(scenario_data['queries'])
            all_answers.extend(scenario_data['answers'])
            all_metadatas.extend(scenario_data['metadatas'])
            logger.info(f"シナリオデータ読み込み完了: {len(scenario_data['queries'])}件")

    # FAQデータの読み込み
    if latest_faq:
        from src.handlers.input_handler import ExcelInputHandler
        faq_path = os.path.join(self.reference_faq_path, latest_faq)
        if os.path.exists(faq_path):
            handler = ExcelInputHandler(self.config)
            handler.reference_dir = self.reference_faq_path
            # 特定ファイルを読み込むために_get_latest_fileを回避
            faq_data = self._load_faq_file(faq_path)
            all_queries.extend(faq_data['queries'])
            all_answers.extend(faq_data['answers'])
            all_metadatas.extend(faq_data['metadatas'])
            logger.info(f"FAQデータ読み込み完了: {len(faq_data['queries'])}件")

    # combined_textsを生成
    all_combined_texts = []
    for query, answer, metadata in zip(all_queries, all_answers, all_metadatas):
        hierarchy = metadata.get('hierarchy', '') if metadata else ''
        text_parts = []
        if hierarchy.strip():
            text_parts.append(f"分類: {hierarchy}")
        if query.strip():
            text_parts.append(f"質問: {query}")
        if answer.strip():
            text_parts.append(f"回答: {answer}")
        combined_text = " | ".join(text_parts) if text_parts else ""
        all_combined_texts.append(combined_text)

    logger.info(f"参照データ準備完了: {len(all_combined_texts)}件")

    return {
        'queries': all_queries,
        'answers': all_answers,
        'combined_texts': all_combined_texts,
        'metadatas': all_metadatas
    }
```

### 3. FAQファイル読み込みヘルパーメソッドの追加

`ExcelInputHandler.load_reference_data()` は最新ファイルを自動検出する仕様のため、特定ファイルを読み込むヘルパーメソッドを追加:

```python
def _load_faq_file(self, faq_path: str) -> dict:
    """特定のFAQファイルを読み込み"""
    import pandas as pd

    logger.info(f"FAQファイル読み込み: {faq_path}")
    reference_df = pd.read_excel(faq_path)

    # 列名の検索ロジック（ExcelInputHandlerと同様）
    possible_query_cols = ['分割後質問', '問合せ内容', '質問内容', '問い合わせ', '質問', 'query', 'Query']
    possible_answer_cols = ['分割後回答', '回答', '既存回答', 'answer', 'Answer']
    possible_tag_cols = ['タグ付け', 'タグ', '分類', 'category', 'Category', 'tag', 'Tag']

    query_col = next((c for c in possible_query_cols if c in reference_df.columns), None)
    answer_col = next((c for c in possible_answer_cols if c in reference_df.columns), None)
    tag_col = next((c for c in possible_tag_cols if c in reference_df.columns), None)

    if query_col is None or answer_col is None:
        raise DynamicDBError(f"必須列が見つかりません: {list(reference_df.columns)}")

    # 日付列の検索
    date_col = None
    if query_col in reference_df.columns:
        query_col_index = reference_df.columns.get_loc(query_col)
        if query_col_index > 0:
            date_col = reference_df.columns[query_col_index - 1]

    queries = []
    answers = []
    combined_texts = []
    metadatas = []

    for idx, row in reference_df.iterrows():
        query_text = str(row[query_col]) if pd.notna(row[query_col]) else ""
        answer_text = str(row[answer_col]) if pd.notna(row[answer_col]) else ""
        tag_text = str(row[tag_col]) if tag_col and pd.notna(row[tag_col]) else ""
        date_text = str(row[date_col]) if date_col and pd.notna(row[date_col]) else ""

        # combined_text生成
        text_parts = []
        if query_text.strip():
            text_parts.append(f"質問: {query_text}")
        if answer_text.strip():
            text_parts.append(f"回答: {answer_text}")
        combined_texts.append(" | ".join(text_parts) if text_parts else "")

        queries.append(query_text)
        answers.append(answer_text)

        metadatas.append({
            'tags': [tag_text] if tag_text.strip() else [],
            'date': date_text if date_text.strip() else "",
            'source': 'history_data',
            'row_index': idx
        })

    return {
        'queries': queries,
        'answers': answers,
        'combined_texts': combined_texts,
        'metadatas': metadatas
    }
```

## 検証方法

1. **ユニットテスト**: DBを再構築して `source='history_data'` のドキュメントが含まれることを確認
2. **UI検証**: chat.pyで「FAQのみ」検索を実行し、結果が返ることを確認
3. **ログ確認**: ベクトル化時に「FAQデータ読み込み完了: N件」のログが出力されることを確認

```bash
# DB再構築
python scripts/rebuild_before_scenario_db.py

# DB内容確認
python scripts/check_db_content.py
```

## リスク評価

- **低リスク**: 既存のシナリオ読み込みロジックには影響なし
- **後方互換**: `latest_faq=None` の場合は従来通りの動作
