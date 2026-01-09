# Multi-Stage Search Feature Implementation

## 変更の背景・目的

事務改定内容をもとに、問合せ履歴から影響を受ける可能性のあるQAを**漏れなく**列挙する機能を追加。

### 3つの検索モード

| モード | 説明 |
|--------|------|
| `original` | 原文検索（入力テキストをそのまま使用してハイブリッド検索） |
| `llm_enhanced` | LLMクエリ化検索（LLMで検索クエリに変換してハイブリッド検索） |
| `multi_stage` | **両方を実行してOR結合**（漏れ防止のため）★新規 |

### multi_stage検索フロー

```
改定内容 → 原文でハイブリッド検索（threshold≥0.5） → 結果A
        → LLMでクエリ変換 → ハイブリッド検索（threshold≥0.5） → 結果B
        → OR結合（A∪B）
        → 3分類:
           - Original_Only（A-B）: 原文検索のみでヒット
           - LLM_Enhanced_Only（B-A）: LLMクエリ検索のみでヒット
           - Both（A∩B）: 両方でヒット
        → LLM影響分析（根拠・修正案生成）
        → 3シートExcel出力
```

## 変更ファイル一覧

1. **config.py** - 多段階検索設定追加
   - `MULTI_STAGE_THRESHOLD`, `MULTI_STAGE_MAX_RESULTS`
   - `multi_stage_enable_llm_analysis`
   - `search_mode`バリデーション更新

2. **src/core/searcher.py** - 多段階OR検索ロジック追加
   - `_execute_multi_stage_search()`: メイン処理
   - `_execute_hybrid_search_with_threshold()`: しきい値付き検索
   - `_merge_multi_stage_results()`: OR結合＋3分類
   - LLM初期化条件の更新

3. **src/handlers/input_handler.py** - TextInputHandler追加
   - テキストファイル入力対応（改定内容の入力用）
   - ファクトリに`"text"`タイプ追加

4. **src/core/impact_analyzer.py** - 新規作成
   - `ImpactAnalyzer`クラス
   - LLMによる影響分析（根拠・修正案生成）
   - リトライロジック付き

5. **src/handlers/output_handler.py** - 3シート出力機能追加
   - `save_data_multi_stage()`: 3シート分割出力
   - `_format_excel_multi_stage()`: 書式設定
   - シート別色分け

6. **src/core/processor.py** - 統合
   - `ImpactAnalyzer`のインポート・初期化
   - `_process_multi_stage_results()`: 後処理（影響分析+3シート出力）

## レビュー観点

1. **機能の正確性**: 多段階検索のOR結合・3分類が正しく実装されているか
2. **エラーハンドリング**: LLM呼び出し失敗時の処理は適切か
3. **後方互換性**: 既存の`original`/`llm_enhanced`モードに影響はないか
4. **コードの可読性**: 新規メソッドの命名・構造は適切か
5. **パフォーマンス**: 大量データ処理時のボトルネックはないか
