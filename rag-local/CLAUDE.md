# RAG-Local プロジェクト メモリ

## プロジェクト概要
シナリオボットの事務改定差分を管理し、RAG検索システムで正解IDを特定するプロジェクト。

## ドキュメント構成

最新のドキュメントは `docs/` ディレクトリに整理されています:

| ドキュメント | 説明 |
|-------------|------|
| [README.md](../README.md) | プロジェクト概要とクイックスタート |
| [docs/ANSWER_SUPPORT.md](../docs/ANSWER_SUPPORT.md) | 回答支援AI（類似回答検索・DB構築） |
| [docs/REVISION_OPS.md](../docs/REVISION_OPS.md) | 改定影響調査システム |
| [docs/CONFIGURATION.md](../docs/CONFIGURATION.md) | 環境変数と設定オプション詳細 |
| [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) | システムアーキテクチャ・API仕様・プロンプト |
| [docs/TROUBLESHOOTING.md](../docs/TROUBLESHOOTING.md) | トラブルシューティング |

---

## 事務改定差分ファイル構成

### フォルダ構造
```
reference/                          # .gitignore対象（git管理外）
├── 改定内容/                       # 改定内容の説明 (revXX_*.md)
├── 改定シナリオ/
│   ├── rev01_スマイル機能変更/
│   │   ├── 差分.md                 # 統一フォーマットの差分ファイル
│   │   ├── 修正前/
│   │   ├── 修正後/
│   │   └── 参考資料/               # 協議書・通達 (PDF/DOCX/PPTX)
│   ├── rev02_相続少額払い/
│   ├── rev03_保険証→資格確認証/
│   ├── rev04_0円新規開設可能/
│   ├── rev05_AMLフィルター→GPLEX/
│   └── rev06_DC→MDC/
├── マージ版シナリオ/
│   ├── 改定前/                     # 改定評価用マージ版
│   └── 最新/                       # 最新版マージ版
├── 問い合わせ履歴/
└── シナリオボットメンテナンス管理台帳.xlsx
```

### 正解IDフォーマット
```
{ボット名}_{Excel行番号}
例: smile-bot_129, naibujimu-bot_641
```

### ボット名対応表
| ボット名 | 対象 |
|---------|------|
| smile-bot | スマイルタブレット |
| naibujimu-bot | 内部事務 |
| souzoku-bot | 相続 |
| torikaku-bot | 取引時確認 |

---

## 事務改定差分.md 統一フォーマット

```markdown
# X番号_タイトル - 事務改定差分

## 📋 メンテナンス管理台帳との照合

**台帳No.XXの記載**:
- ボット名: XXX
- 大分類: XXX
- 変更箇所: 行番号X, Y, Z

**変更行一覧（メンテ台帳 vs 実際の差分）**:

| 台帳記載行 | Excel行 | 実際の差分 | 状態 |
|-----------|---------|-----------|------|
| X | Y | あり | ✓ 一致 |

## ボット名-bot

### ファイル: シナリオ_XXX.xlsx

**カテゴリ**: Lv1=XXX
**変更前シナリオExcelでの範囲**: 行X～行Y

**黄色ハイライト行（変更前シナリオExcel）**: N行
- 行番号: X, Y, Z

---
変更箇所 N: **カテゴリ内行X** (Excel行Y)
質問遷移: A → B → C

**LvN**:
- 変更前: `...`
- 変更後: `...`

**合計 N 行に変更あり**
```

---

## 台帳番号と改定内容の対応

| 番号 | 台帳No. | 内容 | 対象ボット |
|-----|--------|------|----------|
| ① | 20 | スマイル機能変更 | smile-bot |
| ② | 21 | 相続少額払い | souzoku-bot |
| ③ | 25-30, 35-36 | 保険証→資格確認証 | smile-bot, naibujimu-bot, souzoku-bot, torikaku-bot |
| ④ | 37 | 0円新規開設可能 | naibujimu-bot |
| ⑤ | 41-42 | AML→GPLEX | smile-bot |
| ⑥ | 43-45 | DC→MDC | smile-bot |

---

## 変更前シナリオDB生成フロー

### 現行の処理フロー
```
1. マージ版シナリオ (reference/マージ版シナリオ/最新/マージ版シナリオ_XXX-bot.xlsx)
        ↓
    + 修正前カテゴリファイル (手動でカテゴリを置換)
        ↓ [自動化スクリプトなし - 手動作成]

2. 変更前シナリオ (reference/変更前シナリオ/X.../X変更前シナリオ_XXX-bot.xlsx)
        ↓
   prepare_before_scenario.py (文字数列削除・リネーム)
        ↓
3. data/source/scenarios/revisions/revXXボット_シナリオデータ_YYYYMMDD.xlsx
        ↓
   build_db.py --revisions-only → DynamicDBManager
        ↓
4. data/vector_db/revisions/revXX_bot/ (ベクトルDB)
```

### 重要なスクリプト
- `scripts/prepare_before_scenario.py`: 変更前シナリオの前処理（列削除・リネーム）
- `scripts/build_db.py`: DB構築（回答支援AI（類似回答検索）用 + 改定別、統合スクリプト）
- `scripts/generate_correct_ids.py`: 正解ID対応表生成

---

## 既知の問題

### 空行問題
**原因**: 修正前カテゴリファイルの末尾に空行が含まれている
- 例: `修正前/喪失/シナリオ_スマイルタブレット_喪失_20250731.xlsx` の行132が空行
- この空行が変更前シナリオにマージされ、Excel行365として残存

**対応案**:
1. `prepare_before_scenario.py` に空行フィルタリング追加: `df = df[df['Lv1'].notna()]`
2. 元ファイルを手動で修正

### 行番号の注意点
- **台帳記載行**: メンテナンス管理台帳に記載された行番号（カテゴリ内の行番号）
- **Excel行番号**: 変更前シナリオExcelでの実際の行番号（ヘッダー行1 + データ行）
- 計算式: `Excel行番号 = カテゴリ開始行 + カテゴリ内行番号 - 1`

---

## 正解ID抽出ロジック (generate_correct_ids.py)

### 抽出パターン
1. ボット名: `## smile-bot` 形式のセクションヘッダー
2. 行番号リスト: `- 行番号: 129, 185` または `行番号: X, Y, Z`
3. 変更箇所: `**カテゴリ内行X** (Excel行Y)` → Excel行Yを抽出

### 出力
- ファイル: `data/input/multi_stage_input.xlsx`
- 列: 番号, 改定内容, 正解ID

---

## コード構成（最新）

### ディレクトリ構造
```
rag-local/
├── apps/                         # アプリケーション
│   ├── answer-support/           # 回答支援AI（類似回答検索）
│   │   ├── main.py               # バッチ処理エントリーポイント
│   │   └── ui/
│   │       └── chat.py           # Streamlit UI
│   └── revision-ops/             # 運用保守効率化AI（改定影響調査）
│       ├── run_eval.py           # バッチExcel出力
│       └── ui/
│           └── ops_ui.py         # 改定影響調査 Streamlit UI
│
├── config.py                     # 設定管理
├── requirements.txt              # 依存パッケージ
├── .env.example                  # 環境変数テンプレート
│
├── docs/                         # ドキュメント
│   ├── ANSWER_SUPPORT.md         # 回答支援AI詳細
│   ├── REVISION_OPS.md           # 改定影響調査詳細
│   ├── CONFIGURATION.md          # 設定リファレンス
│   ├── ARCHITECTURE.md           # アーキテクチャ・API仕様
│   ├── TROUBLESHOOTING.md        # トラブルシューティング
│   └── plans/                    # 設計書アーカイブ
│
├── src/                          # 共有コアライブラリ
│   ├── core/                     # コアロジック
│   │   ├── processor.py          # データ処理エンジン
│   │   ├── searcher.py           # 検索統合（Processor から使用）
│   │   ├── judgment_support.py   # LLM判断支援
│   │   └── search/               # 検索エンジン
│   │       ├── search_strategy.py           # 検索戦略切替
│   │       ├── multi_stage_orchestrator.py  # 多段階検索
│   │       ├── query_enhancer.py            # クエリ拡張
│   │       ├── vector_search_engine.py      # ベクトル検索
│   │       ├── keyword_search_engine.py     # キーワード検索
│   │       ├── chromadb_keyword_search.py   # ChromaDBキーワード検索
│   │       └── text_combiner.py             # テキスト結合
│   │
│   ├── types/                    # 型定義
│   │   └── search_types.py       # 検索関連の型定義
│   │
│   ├── handlers/                 # 入出力処理
│   │   ├── input_handler.py      # 入力処理
│   │   └── output_handler.py     # 出力処理
│   │
│   └── utils/                    # ユーティリティ
│       ├── dynamic_db_manager.py # DB管理
│       ├── vector_db.py          # ChromaDB ラッパー
│       ├── base_embedding.py     # 埋め込みモデル基底
│       ├── gemini_embedding.py   # Gemini埋め込み
│       ├── azure_embedding.py    # Azure埋め込み
│       ├── auth.py               # Google Cloud認証
│       ├── business_area_translator.py  # 業務領域変換
│       └── logger.py             # ログ設定
│
├── prompt/                       # プロンプト
│   ├── summarize_v1.0.txt        # クエリ拡張
│   └── judgment_support.txt      # 関連性判定
│
├── scripts/                      # ユーティリティスクリプト
│   ├── build_db.py               # DB構築（回答支援AI（類似回答検索）+ 改定別 統合）
│   ├── generate_correct_ids.py   # 正解ID生成
│   ├── prepare_before_scenario.py # データ前処理
│   ├── check_db_content.py       # DB内容確認
│   └── create_handover_package.py # 引き継ぎパッケージ作成
│
├── data/                         # データディレクトリ
│   ├── vector_db/                # ベクトルDB
│   │   ├── update_timestamps.json
│   │   ├── naibujimu/            # 内部事務DB
│   │   ├── smile/                # スマイルDB
│   │   ├── rev01_smile/          # 改定DB
│   │   ├── rev02_souzoku/
│   │   └── ... (rev03-06)
│   ├── source/
│   │   ├── scenarios/            # シナリオExcel
│   │   │   ├── latest/
│   │   │   └── revisions/
│   │   └── faq/                  # FAQデータ
│   │       └── latest/
│   ├── input/                    # 入力ファイル
│   └── output/                   # 出力ファイル
│       ├── latest/
│       └── archive/
│
├── tests/                        # テスト
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
└── logs/                         # ログファイル
```

### 主要モジュールの役割

#### コアモジュール
- **processor.py**: データ処理の統合管理
- **judgment_support.py**: LLMによる関連性判定
- **multi_stage_orchestrator.py**: 多段階ハイブリッド検索
- **query_enhancer.py**: LLMクエリ拡張

#### データベース管理
- **dynamic_db_manager.py**: 業務領域別DB管理、タイムスタンプ検証
- **vector_db.py**: ChromaDB操作ラッパー


#### 埋め込みモデル
- **base_embedding.py**: 抽象基底クラス
- **gemini_embedding.py**: VertexAI Gemini埋め込み
- **azure_embedding.py**: Azure OpenAI埋め込み

詳細は [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) を参照してください。

---

## 環境変数設定

### 必須環境変数
```env
# GCP認証方式（local: ローカルJSONファイル / key_vault: Azure Key Vault）
CREDENTIAL_SOURCE=local

# LLM設定（Geminiのみサポート）
DEFAULT_LLM_PROVIDER=gemini
DEFAULT_LLM_MODEL=gemini-2.5-flash-lite

# 埋め込みモデル設定（モデルはプロバイダーから自動解決）
DEFAULT_EMBEDDING_PROVIDER=azure_openai

# Google Cloud
GEMINI_PROJECT_ID=your-project-id
GEMINI_LOCATION=us-central1

# Azure OpenAI
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large
AZURE_OPENAI_API_VERSION=2024-12-01-preview
```

詳細は [docs/CONFIGURATION.md](../docs/CONFIGURATION.md) を参照してください。

---

## 技術的な注意事項

- **検索モード**: `search_mode` パラメータで制御（`enable_query_enhancement`は廃止済み）
- **検索実行**: `SearchStrategy`パターン（`src/core/search/search_strategy.py`）で4戦略クラスを切替
- **タイムスタンプ**: フラット形式（旧3階層から自動移行対応済み）
- **テスト**: `pytest` で実行（`tests/`, `pytest.ini`, `requirements-dev.txt`）
- **コレクション命名**: `rev{XX}_{bot}` 形式（例: `rev01_smile`, `rev02_souzoku`）

