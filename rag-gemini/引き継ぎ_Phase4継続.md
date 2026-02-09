# RAG-Gemini プロジェクト整理作業 - 引き継ぎ文書

**作成日**: 2026-02-09
**作業ブランチ**: `refactor/project-cleanup`
**進捗状況**: Phase 4.2-1まで完了（全体の約70%完了）

---

## 📊 現在の状況

### ✅ 完了したフェーズ

| Phase | 内容 | 状態 | 成果 |
|-------|------|------|------|
| Phase 0 | バックアップ・準備 | ✅ 完了 | 作業ブランチ作成 |
| Phase 1 | セキュリティ対応 | ✅ 完了 | 危険ファイル削除、.gitignore強化 |
| Phase 2 | ファイルクリーンアップ | ✅ 完了 | 23,788ファイル削除、10MB削減 |
| Phase 3 | フォルダ構造整理 | ✅ 完了 | data/配下に統一、825MB削減 |
| Phase 4.1 | テスト基盤構築 | ✅ 完了 | pytest環境、ユニットテスト |
| Phase 4.2-1 | キーワード抽出統一 | ✅ 完了 | KeywordSearchEngineに委譲 |
| Phase 5 | ドキュメント整備 | ✅ 完了 | 8つのドキュメント作成 |

### ⏳ 未完了（次回作業）

| Phase | 内容 | 優先度 | 所要時間 |
|-------|------|--------|---------|
| Phase 4.2-2 | 検索モード設定統一 | 高 | 20分 |
| Phase 4.3 | SearchStrategyパターン導入 | 高 | 45分 |
| Phase 4.4 | DynamicDBManager簡素化 | 中 | 20分 |
| Phase 4.5 | 旧コード削除 | 低 | 15分 |
| Phase 6 | 最終確認・統合テスト | 高 | 30分 |

**推定残り時間**: 約130分（2時間10分）

---

## 🎯 次回セッションでの作業内容

### 1. Phase 4.2-2: 検索モード設定の統一（20分）

**目的**: `enable_query_enhancement`を廃止し、`search_mode`に統一

**作業内容**:
```python
# config.py の修正
@property
def enable_query_enhancement(self) -> bool:
    """Deprecated: search_mode='llm_enhanced'を使用してください"""
    import warnings
    warnings.warn(
        "enable_query_enhancement is deprecated. "
        "Use search_mode='llm_enhanced' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return self.search_mode == "llm_enhanced"
```

**影響ファイル**:
- `config.py`
- `src/core/searcher.py` (437行目付近)

---

### 2. Phase 4.3: SearchStrategyパターン導入（45分）

**目的**: Searcherクラスの責務分離（1021行 → 500行以下）

**手順**:
1. **新規ファイル作成**: `src/core/search/search_strategy.py`
2. **抽象基底クラス作成**: `SearchStrategy`
3. **具体クラス実装**:
   - `OriginalSearchStrategy`: ハイブリッド検索
   - `LLMEnhancedSearchStrategy`: LLM拡張検索
   - `MultiStageSearchStrategy`: 多段階検索
4. **Searcherクラスのリファクタリング**:
   ```python
   class Searcher:
       def __init__(self, config: SearchConfig):
           self.config = config
           self.strategy = self._create_strategy()

       def _create_strategy(self) -> SearchStrategy:
           if self.config.search_mode == "original":
               return OriginalSearchStrategy(...)
           elif self.config.search_mode == "llm_enhanced":
               return LLMEnhancedSearchStrategy(...)
           elif self.config.search_mode == "multi_stage":
               return MultiStageSearchStrategy(...)

       def search(self, ...):
           return self.strategy.search(...)
   ```

**影響ファイル**:
- `src/core/searcher.py` (大幅リファクタリング)
- `src/core/search/search_strategy.py` (新規作成)

---

### 3. Phase 4.4: DynamicDBManager簡素化（20分）

**目的**: タイムスタンプ構造のフラット化

**変更内容**:
```python
# 修正前（3階層）
{
  "general": {
    "vertex_ai": {"faq": 1234567890, "scenario": 1234567890},
    "azure_openai": {"faq": 1234567890, "scenario": 1234567890}
  }
}

# 修正後（フラット）
{
  "general_vertex_ai_faq": 1234567890,
  "general_vertex_ai_scenario": 1234567890,
  "general_azure_openai_faq": 1234567890,
  "general_azure_openai_scenario": 1234567890
}
```

**影響ファイル**:
- `src/utils/dynamic_db_manager.py` (_load_update_timestamps, _save_update_timestamps)

---

### 4. Phase 4.5: 旧コード削除とクリーンアップ（15分）

**削除対象**:
- `Searcher._execute_multi_stage_search()` → MultiStageOrchestrator完全移行後
- Phase 4.3完了後、全テストがパスすることを確認してから削除

**注意**: Deprecation警告を1バージョン期間表示してから削除

---

### 5. Phase 6: 最終確認・統合テスト（30分）

**確認項目**:
1. **動作確認**:
   ```bash
   python main.py preflight --business スマイル
   python scripts/evaluate_revisions.py
   ```

2. **テスト実行**:
   ```bash
   pytest tests/ -v
   ```

3. **パフォーマンステスト**:
   - リファクタリング前後で性能劣化がないか確認

4. **最終コミット**:
   ```bash
   git add .
   git commit -m "refactor: Phase 4 コードリファクタリング完了

   - SearchStrategyパターン導入
   - DynamicDBManager簡素化
   - テストコード基盤完成
   - Searcherクラス500行以下に削減

   Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
   ```

---

## 📁 重要なファイル

### 設定ファイル
- `config.py` (SearchConfigクラス)
- `config/settings.yaml` (検索設定、revision_areas)

### コアファイル
- `src/core/searcher.py` (1021行、リファクタリング対象)
- `src/core/search/keyword_search_engine.py`
- `src/core/search/multi_stage_orchestrator.py`
- `src/utils/dynamic_db_manager.py`

### テストファイル
- `tests/conftest.py` (共通フィクスチャ)
- `tests/unit/test_keyword_search_engine.py`

### ドキュメント
- `整理計画.md` (全体計画)
- `docs/ARCHITECTURE.md` (システム設計)
- `docs/API_REFERENCE.md` (API仕様)

---

## ⚠️ 注意事項

### 1. 後方互換性の維持
- 既存のメソッドはDeprecated警告を表示しながら動作維持
- 段階的な移行を推奨

### 2. テストの重要性
- Phase 4.3実装後、必ずテスト実行
- プレフライトチェックで動作確認

### 3. コミット戦略
- 各サブフェーズごとにコミット
- 問題があればロールバック可能に

### 4. データ整理/は削除しない
- トレーサビリティ確保のため保持
- 事務改定の差分データが含まれる

---

## 🔍 トラブルシューティング

### コレクション名の不一致
- **症状**: DBが見つからない
- **確認**: `config/settings.yaml`のrevision_areasを確認
- **正しい形式**: `rev01_smile` (アンダースコア区切り)

### テスト失敗
- **対処**: `conftest.py`のフィクスチャを確認
- **依存関係**: `requirements-dev.txt`をインストール

### プレフライトエラー
- **確認**: `data/vector_db/`, `data/source/`のパス
- **設定**: `src/utils/dynamic_db_manager.py`の26-28行

---

## 📊 進捗管理

### 完了判定基準

- [ ] Phase 4.2-2完了（enable_query_enhancement廃止）
- [ ] Phase 4.3完了（SearchStrategyパターン導入）
- [ ] Phase 4.4完了（DynamicDBManager簡素化）
- [ ] Phase 4.5完了（旧コード削除）
- [ ] Phase 6完了（統合テスト成功）
- [ ] Searcherクラス500行以下達成
- [ ] 全テストパス
- [ ] プレフライトチェック成功
- [ ] 最終コミット完了

### 成果物

- [ ] `src/core/search/search_strategy.py`（新規）
- [ ] `src/core/searcher.py`（500行以下）
- [ ] `src/utils/dynamic_db_manager.py`（簡素化）
- [ ] `tests/unit/`（追加テスト）
- [ ] 統合テスト実行結果

---

## 📝 参考リンク

- **整理計画**: `整理計画.md`
- **メモリ**: `~/.claude/projects/C--VSCode-rag/memory/MEMORY.md`
- **詳細設計**: 調査エージェント（Task ID: a55c502）のレポート

---

## 💡 推奨される開始コマンド

```bash
# 1. ブランチ確認
git status
git log --oneline -3

# 2. 現在のディレクトリ構造確認
ls data/

# 3. 動作確認
python main.py preflight --business スマイル

# 4. Phase 4.2-2から再開
# config.pyの修正から開始
```

---

**次回セッション開始時**: この文書を読み込んで、Phase 4.2-2から継続してください。
