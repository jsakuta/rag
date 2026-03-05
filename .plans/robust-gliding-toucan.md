# 運用保守効率化AI UI操作ガイド（Word）作成計画

## Context

`REVISION_OPS.md` にはUI版（ops_ui.py）の操作手順がテキストのみで記載されており、スクリーンショットがない。引き継ぎ資料として、吹き出し付きスクリーンショットで操作手順を視覚的に説明するWord文書を作成する。

## パイプライン概要

```
Phase 1: Streamlit起動 + agent-browserでスクリーンショット撮影（13枚）
Phase 2: Pillow スクリプトでアノテーション（番号付き丸、ハイライト矩形、テキストラベル）
Phase 3: docx-js でWord文書組み立て
```

## 作業ディレクトリ

```
rag-local/docs/guide/
├── screenshots/           # 生のスクリーンショット
├── annotated/             # アノテーション済みPNG
├── annotate.py            # Pillow アノテーションスクリプト
├── annotation_config.json # 各スクリーンショットの注釈座標
├── step_descriptions.json # 各ステップの説明テキスト
├── build_guide.js         # docx-js 文書組立
├── package.json           # { "dependencies": { "docx": "^8" } }
└── ops_ui_guide.docx      # 最終出力
```

## Phase 1: スクリーンショット撮影

### 前提
- `cd rag-local && .venv/Scripts/python.exe -m streamlit run apps/revision-ops/ui/ops_ui.py`
- プリウォーム待機 ~2分（Vertex AI初期化）
- URL: `http://localhost:8501`

### スクリーンショット一覧（13枚）

| # | ID | 状態 | 内容 | モード |
|---|-----|------|------|--------|
| 1 | `01_launch` | 初期表示 | 全体レイアウト（サイドバー+メイン） | 共通 |
| 2 | `02_eval_sidebar` | 評価モードサイドバー | 全設定項目が見える状態 | 評価 |
| 3 | `03_eval_revision_select` | セレクトボックス展開 | 改定番号の選択肢一覧 | 評価 |
| 4 | `04_eval_revision_selected` | 改定番号選択後 | 正解ID件数+対象エリア表示 | 評価 |
| 5 | `05_eval_query_input` | クエリ入力 | テキスト入力欄にサンプルクエリ | 評価 |
| 6 | `06_eval_results` | 検索結果表示 | LLM強化クエリ+タブ+結果カード | 評価 |
| 7 | `07_eval_results_detail` | 結果カード詳細 | スコア、正解バッジ、Q&A内容 | 評価 |
| 8 | `08_impact_mode_switch` | モード切替 | 影響調査モードに切替後 | 影響調査 |
| 9 | `09_impact_sidebar` | 影響調査サイドバー | カテゴリ+ソース+設定 | 影響調査 |
| 10 | `10_impact_query` | クエリ入力 | 影響調査用サンプルクエリ | 影響調査 |
| 11 | `11_impact_results` | 影響調査結果 | 結果一覧（正解バッジなし） | 影響調査 |
| 12 | `12_save_history` | 履歴保存 | 保存成功メッセージ | 共通 |
| 13 | `13_keyword_search` | キーワード検索選択 | 全件返却キャプション表示 | 共通 |

### 撮影手順
1. `agent-browser open http://localhost:8501` → 2分待機
2. `agent-browser snapshot -i` で要素ref取得
3. 各画面で操作 → `agent-browser screenshot <path>` で撮影
4. Streamlit selectboxは click → option click の2段階操作

## Phase 2: Pillow アノテーション

### `annotate.py` — アノテーション機能
- **番号付き丸**: 赤い円 + 白数字（Arial 20pt）— UI要素の位置を示す
- **ハイライト矩形**: 半透明黄色オーバーレイ — 注目領域を強調
- **テキストラベル**: 暗色背景 + 白文字（Meiryo UI 14pt）— 説明テキスト
- **矢印**: 赤い線 + 三角矢じり — 関係性を示す

### フォント
- 数字: `C:/Windows/Fonts/arial.ttf`
- 日本語: `C:/Windows/Fonts/meiryo.ttc`

### 座標設定
- 撮影後にスクリーンショットを確認し `annotation_config.json` に座標を記入
- ビューポート固定サイズ（1280x720）で統一

## Phase 3: Word文書組み立て

### 文書構成

```
表紙: 運用保守効率化AI（改定影響調査）操作ガイド

1. はじめに
   1.1 本ガイドについて
   1.2 前提条件（.env、DB構築済み等）
   1.3 アプリ起動方法

2. 画面構成 [Screenshot 01]
   2.1 全体レイアウト
   2.2 サイドバー
   2.3 メインエリア

3. 評価モード
   3.1 改定番号の選択 [Screenshot 02, 03, 04]
   3.2 検索設定の調整 [Screenshot 02 detail]
   3.3 検索の実行 [Screenshot 05]
   3.4 結果の確認 [Screenshot 06, 07]

4. 影響調査モード
   4.1 モード切替 [Screenshot 08]
   4.2 カテゴリ・ソース選択 [Screenshot 09]
   4.3 検索の実行 [Screenshot 10]
   4.4 結果の確認 [Screenshot 11]

5. 共通操作
   5.1 キーワード検索 [Screenshot 13]
   5.2 チャット履歴の保存 [Screenshot 12]

6. 注意事項
```

### 各ステップのレイアウト

```
[見出し] ステップ名

[アノテーション付きスクリーンショット — 幅600pt中央寄せ]

[テーブル: 番号 | 操作 | 説明]
① | ○○を選択 | ○○の説明…
② | △△をクリック | △△の説明…
```

### 技術スタック
- `npm install docx`（docx-js v8）
- `node build_guide.js` で .docx 生成
- ImageRun で PNG 埋め込み、Table で手順表

## 重要ファイル

| ファイル | 用途 |
|---------|------|
| `rag-local/apps/revision-ops/ui/ops_ui.py` | 撮影対象のStreamlit UI |
| `rag-local/ui/shared.py` | 共有UIコンポーネント（CSS、カード書式） |
| `~/.claude/skills/agent-browser/SKILL.md` | agent-browserコマンドリファレンス |
| `~/.claude/skills/docx/docx-js.md` | docx-js APIリファレンス |
| `rag-local/docs/REVISION_OPS.md` | 既存ドキュメント（用語・コンテキスト参照） |

## 実行順序

1. **作業ディレクトリ作成**: `rag-local/docs/guide/screenshots/`, `annotated/`
2. **Streamlit起動**: バックグラウンドで `streamlit run` 実行
3. **agent-browserで撮影**: 13枚を順番に撮影（評価モード→影響調査モード）
4. **座標計測 + annotation_config.json 作成**
5. **annotate.py 作成・実行**: アノテーション済みPNG生成
6. **step_descriptions.json 作成**: 各ステップの説明テキスト
7. **build_guide.js 作成**: docx-js で文書組立
8. **npm install + node build_guide.js**: Word文書生成
9. **検証**: 出力docxを開いて画像・テキスト・レイアウト確認

## 検証方法
- 生成された `ops_ui_guide.docx` をWordで開き、全画像が正しく表示されることを確認
- アノテーションの番号とテーブルの番号が対応していることを確認
- 日本語テキストが文字化けしていないことを確認
