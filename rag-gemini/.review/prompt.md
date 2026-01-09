# Code Review Request

## 対象
多段階OR検索機能の実装

## レビュー観点

### 1. バグ・ロジックエラー (severity: high)
- 多段階検索のOR結合ロジックは正しいか
- しきい値フィルタリングは正しく動作するか
- 3分類（Both, Original_Only, LLM_Enhanced_Only）のロジックに漏れはないか

### 2. セキュリティ (severity: high)
- LLM API呼び出し時の入力バリデーションは適切か
- ファイルパス操作に脆弱性はないか

### 3. エラーハンドリング (severity: medium)
- LLM呼び出し失敗時のフォールバック処理は適切か
- 空のデータに対する処理は正しいか

### 4. コード品質 (severity: low)
- 命名規則・コーディング規約への準拠
- 冗長なコードや重複はないか
- 型ヒントの一貫性

### 5. パフォーマンス (severity: medium)
- 大量データ処理時のボトルネックはないか
- 不要なループや計算はないか

## 出力形式

```json
{
  "issues": [
    {
      "file": "ファイルパス",
      "line": 行番号,
      "severity": "high|medium|low",
      "category": "bug|security|error-handling|code-quality|performance",
      "description": "問題の説明",
      "suggestion": "修正案"
    }
  ],
  "summary": "全体的な評価（1-2文）"
}
```
