# 改定番号別検索タイプ設定の実装計画

## 概要
事務改定評価スクリプト（evaluate_revisions.py）で、改定番号ごとに検索タイプ（類似検索/キーワード必須）とベクトル重みを個別に設定できるようにする。

## 現状
- `settings.yaml`の`revision_areas`で改定番号ごとに`vector_weight`は設定可能
- 検索タイプ（`search_type`）は未対応
- ⑤⑥は`vector_weight: 0.0`でキーワード重視になっているが、キーワードフィルタ（必須）とは異なる

## 設定ファイル変更

### config/settings.yaml
```yaml
revision_areas:
  "①":
    areas:
      - rev01smile
    search_type: hybrid        # 類似検索（意味で探す）
    vector_weight: 0.9         # 意味重視
  "②":
    areas:
      - rev02souzoku
    search_type: hybrid
    vector_weight: 0.9
  "③":
    areas:
      - rev03naibujimu
      - rev03smile
      - rev03souzoku
      - rev03torikaku
    search_type: hybrid
    vector_weight: 0.9
  "④":
    areas:
      - rev04naibujimu
    search_type: hybrid
    vector_weight: 0.9
  "⑤":
    areas:
      - rev05smile
    search_type: keyword_filter  # キーワード必須
    # vector_weightは不要（キーワード必須なので無視される）
  "⑥":
    areas:
      - rev06smile
    search_type: keyword_filter  # キーワード必須
```

## コード変更

### 1. scripts/evaluate_revisions.py

#### 変更箇所1: 設定読み込み部分（L76-88）
- `REVISION_SEARCH_TYPES`辞書を追加
- 各改定番号の`search_type`を読み込み（デフォルト: `hybrid`）

#### 変更箇所2: search_revision_multi_stage メソッド（L277-325）
- search_typeに応じて検索方法を分岐
- `keyword_filter`の場合: 新メソッド`_execute_keyword_filter`を呼び出し
- `hybrid`の場合: 既存のMultiStageOrchestratorを使用

#### 変更箇所3: 新メソッド追加 `_execute_keyword_filter`
- KeywordSearchEngineのfilter_by_keywords_from_cacheを使用
- ベクトルDBから全ドキュメントを取得
- キーワードマッチでフィルタリング
- マッチ数順でソート
- 結果をフォーマット

#### 変更箇所4: 評価設定表示（L894-910）
- 改定番号別の検索タイプを表示

### 2. 出力フォーマット変更
- 詳細シートに「検索タイプ」列を追加
- サマリーシートに検索タイプを表示

## 実装ファイル一覧

| ファイル | 変更内容 |
|---------|---------|
| config/settings.yaml | revision_areasに`search_type`を追加 |
| scripts/evaluate_revisions.py | キーワードフィルタ検索の分岐処理を追加 |

## 検証方法
1. `python scripts/evaluate_revisions.py`を実行
2. 出力Excelで以下を確認:
   - ①〜④: 類似検索（ベクトル+キーワードスコア計算）
   - ⑤⑥: キーワード必須（キーワードを含む結果のみ）
3. ログで検索タイプが正しく表示されること
