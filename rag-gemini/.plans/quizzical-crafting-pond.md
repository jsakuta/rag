# Vertex AI DB構築計画

## 目的
事務改定用の9つのDBに対して、Vertex AI（gemini-embedding-001）用のベクトルDBを構築し、2つのembeddingモデル（Azure OpenAI + Vertex AI）で検索可能な状態にする。

## 現状

| 改定DB | Azure OpenAI | Vertex AI |
|--------|:------------:|:---------:|
| rev01smile | ✓ | ✓ |
| rev02souzoku | ✓ | **欠落** |
| rev03smile | ✓ | **欠落** |
| rev03naibujimu | ✓ | **欠落** |
| rev03souzoku | ✓ | **欠落** |
| rev03torikaku | ✓ | **欠落** |
| rev04naibujimu | ✓ | **欠落** |
| rev05smile | ✓ | **欠落** |
| rev06smile | ✓ | **欠落** |

## 実行手順

### Step 1: 前提条件の確認
```bash
# 1. Vertex AI認証ファイルの存在確認
ls -la gemini_credentials.json

# 2. 環境変数の確認（.envに設定済みか）
# - GEMINI_PROJECT_ID
# - GEMINI_LOCATION
```

### Step 2: Streamlit UIの停止
実行中のStreamlitプロセスを停止する（DBファイルのロック回避）

### Step 3: スクリプト実行
```bash
python scripts/rebuild_before_scenario_db.py
```

**注意**: このスクリプトは以下を実行する：
1. 既存のrev* DBディレクトリを削除
2. タイムスタンプファイルをリセット
3. Azure OpenAIとVertex AI両方で全9DBを再構築

### Step 4: 結果検証
```bash
# 各revXXディレクトリにazure_openaiとvertex_aiの両方が存在するか確認
ls reference/vector_db/rev*/
```

期待する結果:
```
reference/vector_db/
├── rev01smile/
│   ├── azure_openai/
│   └── vertex_ai/
├── rev02souzoku/
│   ├── azure_openai/
│   └── vertex_ai/
... (同様に全9DB)
```

## 対象ファイル
- `scripts/rebuild_before_scenario_db.py` - 再構築スクリプト（既存、変更不要）

## 注意事項
- 再構築は既存DBを**すべて削除**してから行う
- Azure OpenAIのDBも再構築されるため、多少時間がかかる
- Vertex AI APIのレート制限に注意（エラー時はリトライが自動実行される）
