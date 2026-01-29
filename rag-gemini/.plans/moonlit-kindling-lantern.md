# 事務改定評価システムの設計と整理

## 現状の理解

### DB構造
```
reference/vector_db/
├── general/              # 総則（内部事務等を統合したカテゴリ）
│   ├── azure_openai/
│   └── vertex_ai/
├── deposit/azure_openai/ # 預金業務用
├── rev01smile/           # 事務改定①用（smile-bot）
├── rev02souzoku/         # 事務改定②用（souzoku-bot）
├── rev03naibujimu/       # 事務改定③用（naibujimu-bot）
├── rev03smile/           # 事務改定③用（smile-bot）
├── rev03souzoku/         # 事務改定③用（souzoku-bot）
├── rev03torikaku/        # 事務改定③用（torikaku-bot）
├── rev04naibujimu/       # 事務改定④用
├── rev05smile/           # 事務改定⑤用
├── rev06smile/           # 事務改定⑥用
└── revision_eval/        # ★不要（今回誤って作成したゴミ）
```

### データの関係
- `reference/scenario/rev*_シナリオデータ_*.xlsx` = `データ整理/事務改定前_マージ版シナリオ/*.xlsx`
- 同じデータ（行数一致：555行 etc）、文字数列の有無の違いのみ
- 既存rev*DBは**Azure OpenAIのみ**ベクトル化済み

---

## 設計方針

### 1. 全削除＆再構築
- 既存のrev*DBを全て削除
- Azure OpenAIとVertexAI両方を新規構築

### 2. DB構造の最終形
```
reference/vector_db/
├── general/                  # 総則（既存のまま）
│   ├── azure_openai/
│   └── vertex_ai/
├── rev01smile/               # ★全削除＆再構築
│   ├── azure_openai/
│   └── vertex_ai/
├── rev02souzoku/
│   ├── azure_openai/
│   └── vertex_ai/
├── rev03naibujimu/
│   ├── azure_openai/
│   └── vertex_ai/
├── rev03smile/
│   ├── azure_openai/
│   └── vertex_ai/
├── rev03souzoku/
│   ├── azure_openai/
│   └── vertex_ai/
├── rev03torikaku/
│   ├── azure_openai/
│   └── vertex_ai/
├── rev04naibujimu/
│   ├── azure_openai/
│   └── vertex_ai/
├── rev05smile/
│   ├── azure_openai/
│   └── vertex_ai/
├── rev06smile/
│   ├── azure_openai/
│   └── vertex_ai/
```

### 3. 不要フォルダの削除
- `revision_eval/` を削除（今回の試行で作成されたゴミ）
- 既存のrev*DBも削除（再構築するため）

---

## 実装タスク

### タスク1: 不要フォルダ削除
```bash
rm -rf reference/vector_db/revision_eval/
rm -rf reference/vector_db/rev*
```

### タスク2: DB再構築スクリプト修正
**ファイル**: `scripts/rebuild_before_scenario_db.py`

修正内容：
```python
PROVIDERS = ["azure_openai", "vertex_ai"]

def rebuild_dbs():
    for provider in PROVIDERS:
        logger.info(f"=== {provider} でDB構築開始 ===")

        # プロバイダー別に設定を作成
        config = SearchConfig(base_dir=str(PROJECT_ROOT))
        config.embedding_provider = provider

        # embedding_modelも切り替え
        if provider == "azure_openai":
            config.embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        else:
            config.embedding_model = os.getenv("DEFAULT_EMBEDDING_MODEL")

        with DynamicDBManager(config) as db_manager:
            business_areas = db_manager.analyze_reference_files()
            for area in REVISION_AREAS:
                db_manager.update_business_db(area, business_areas[area])
```

### タスク3: 評価スクリプト修正
**ファイル**: `scripts/evaluate_revisions.py`

修正内容：
```python
# 改定番号 → rev*業務分野のマッピング
REVISION_TO_AREAS = {
    '①': ['rev01smile'],
    '②': ['rev02souzoku'],
    '③': ['rev03naibujimu', 'rev03smile', 'rev03souzoku', 'rev03torikaku'],
    '④': ['rev04naibujimu'],
    '⑤': ['rev05smile'],
    '⑥': ['rev06smile'],
}

def search_revision(revision, query, correct_ids, provider):
    """既存DBを使って検索"""
    areas = REVISION_TO_AREAS[revision]
    all_results = []

    for area in areas:
        db_path = f"reference/vector_db/{area}/{provider}"
        # MetadataVectorDBで既存DBを読み込み
        vector_db = MetadataVectorDB(db_path=db_path, collection_name=area)
        # 検索実行
        results = vector_db.search(query_embedding, n_results=top_k)
        all_results.extend(results)

    return all_results
```

出力形式（改定ごとにシート分割）：
- サマリーシート: 全改定の正解率一覧
- 改定①シート: 改定①の検索結果詳細
- 改定②シート: 改定②の検索結果詳細
- ...
- 改定⑥シート: 改定⑥の検索結果詳細

### タスク4: DB再構築実行
```bash
python scripts/rebuild_before_scenario_db.py
```

### タスク5: 評価実行＆検証
```bash
python scripts/evaluate_revisions.py
```

---

## 修正するファイル

| ファイル | 変更内容 |
|---------|---------|
| `scripts/rebuild_before_scenario_db.py` | 両プロバイダー対応 |
| `scripts/evaluate_revisions.py` | 既存DB使用に変更 |

---

## 検証方法

1. `reference/vector_db/rev*/`配下に`azure_openai/`と`vertex_ai/`が作成されることを確認
2. `update_timestamps.json`に両プロバイダーのエントリが追加されることを確認
3. 評価スクリプトでExcel出力を確認（改定ごとのシート、正解率サマリー）
4. 通常の検索（`python main.py`等）が引き続き動作することを確認
