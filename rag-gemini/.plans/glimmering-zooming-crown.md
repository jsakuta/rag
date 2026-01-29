# RAG-Gemini プロジェクト アーキテクチャレビュー結果

## 結論: 統一されたアーキテクチャで構築されている

ユーザーの懸念である「別々の仕組みとして構築されていないか」について、**統一されている**と評価できます。

---

## 1. 現状の要件・機能整理

### 検索機能（3モード）

| モード | 説明 | エントリーポイント |
|--------|------|-------------------|
| **original** | ベクトル(0.9) + キーワード(0.1) | `Searcher.search()` |
| **llm_enhanced** | LLMクエリ拡張 + ハイブリッド検索 | 同上 |
| **multi_stage** | 原文+LLM並列検索 → OR結合 → 3分類 | 同上 |

**統一性**: 全モードが `Searcher.search()` を単一エントリーポイントとして使用

### データソース（2種類）

| ソース | 説明 | 識別方法 |
|--------|------|---------|
| **scenario** | シナリオデータ | `source` メタデータ |
| **history_data** | 問い合わせ履歴（FAQ） | `source` メタデータ |

**統一性**: 同じ `combined_texts` + `metadatas` 形式で保持、`_build_source_filter()` で統一フィルタリング

### DB管理（業務分野別・プロバイダー別）

```
reference/vector_db/
├── general/          # 総則
│   ├── azure_openai/
│   └── vertex_ai/
├── deposit/          # 預金
├── rev01smile/       # 事務改定①
├── rev02souzoku/     # 事務改定②
├── rev03smile/       # 事務改定③（smile）
├── rev03naibujimu/   # 事務改定③（内部事務）
...
└── update_timestamps.json
```

**統一性**: `DynamicDBManager` が一元管理、入力ファイル名から自動DB選択

---

## 2. アーキテクチャ図

```
main.py (エントリーポイント)
    │
    ▼
Processor (オーケストレーション: 257行)
    ├── InputHandlerFactory   ← 入力読み込み
    │     └── Excel / Hierarchical / MultiFolder
    │
    ├── Searcher (検索エンジン: 903行) ← ★中核モジュール
    │     ├── search_mode分岐
    │     │     ├── original → _execute_vector_search()
    │     │     ├── llm_enhanced → summarize_text() → 同上
    │     │     └── multi_stage → _execute_multi_stage_search()
    │     │
    │     ├── DynamicDBManager (DB管理: 864行)
    │     │     └── 業務分野×プロバイダー別DB
    │     │
    │     └── EmbeddingModel (埋め込み)
    │           ├── GeminiEmbeddingModel
    │           └── AzureEmbeddingModel
    │
    ├── JudgmentSupport (LLM判断支援)
    │
    └── OutputHandlerFactory  ← 結果出力
```

---

## 3. 良い点（強み）

| 観点 | 評価 |
|------|------|
| **機能の統合性** | 全機能が統一されたパイプラインで実装 |
| **デザインパターン** | Factory, Strategy, DIが適切に適用 |
| **セキュリティ** | パストラバーサル防止、ホワイトリスト検証 |
| **パフォーマンス** | キャッシュ、バッチ処理、並列処理 |
| **拡張性** | 新プロバイダー・検索モードの追加が容易 |

---

## 4. 改善すべき点

### Priority 1: 検索結果の型安全化
- **問題**: 結果辞書のキーが文字列リテラルとして散在
- **提案**: `TypedDict` または `dataclass` の導入

### Priority 2: Searcherクラスの責務分離（903行）
- **問題**: 複数責務が1クラスに集中
- **提案**: `VectorSearchEngine`, `KeywordSearchEngine`, `QueryEnhancer`, `MultiStageOrchestrator` に分割

### Priority 3: タイムスタンプ管理の簡素化
- **問題**: 3階層構造が複雑
- **提案**: `DBVersionManager` クラスの導入

### Priority 4: 結合テキスト生成の統一
- **問題**: 同じロジックが複数箇所に存在
- **提案**: `TextCombiner` ユーティリティの抽出

### Priority 5: 設定の外部化
- **問題**: 業務分野マッピングがハードコード
- **提案**: YAML設定ファイルに移動

---

## 5. 事務改定DBの統合状況

| 改定番号 | 内容 | DB名 | 状態 |
|---------|------|------|------|
| ① | スマイル機能変更 | rev01smile | 統合済み |
| ② | 相続少額払い | rev02souzoku | 統合済み |
| ③ | 保険証→資格確認証 | rev03{smile,naibujimu,souzoku,torikaku} | 統合済み |
| ④ | 0円新規開設可能 | rev04naibujimu | 統合済み |
| ⑤ | AML→GPLEX | rev05smile | 統合済み |
| ⑥ | DC→MDC | rev06smile | 統合済み |

**結論**: 既存アーキテクチャに完全統合済み。新改定追加時も同じパターンで対応可能。

---

## 6. 重要ファイル一覧

| ファイル | 行数 | 責務 |
|---------|------|------|
| `src/core/searcher.py` | 903 | 検索ロジック中核 |
| `src/utils/dynamic_db_manager.py` | 864 | DB管理・切替 |
| `src/handlers/input_handler.py` | 620 | 入力処理 |
| `src/handlers/output_handler.py` | 431 | 出力処理 |
| `src/core/processor.py` | 257 | オーケストレーション |
| `config.py` | 216 | 設定管理 |

---

## 7. 推奨アクション

1. **現状維持で問題なし** - アーキテクチャは統一されており、懸念は杞憂
2. **将来的なリファクタリング候補**として上記改善点を検討（緊急性なし）
3. **DBの作成**を進めて問題なし - 既存パターンに従えば自動統合される
