# rag-batch

## 他プロジェクトとの違い
- **特徴**: Excel一括バッチ処理に最適化、ハイブリッド検索（ベクトル + キーワード）、Factory Pattern採用
- **代替プロジェクト**:
  - 対話的UIが必要な場合: [rag-streamlit](../rag-streamlit/)
  - 最新技術を使いたい場合: [rag-gemini](../rag-gemini/)
  - プロジェクト全体の比較: [ルートREADME](../README.md)

---

[![Python](https://img.shields.io/badge/Python-3.7+-blue)](https://www.python.org/)
[![Sentence Transformers](https://img.shields.io/badge/SentenceTransformers-2.2.0+-blue)](https://www.sbert.net/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1.0+-blue)](https://python.langchain.com/)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 概要
本システムは、大規模言語モデル(LLM)とベクトル検索、キーワード検索を組み合わせたハイブリッド検索システムです。質問文から類似度の高い参照データを検索し、高精度な回答を生成・提供します。

### 主な機能
- **ハイブリッド検索**:
    - ベクトル類似度検索 (SentenceTransformers)
    - キーワード類似度検索 (SudachiPy)
    - 検索スコアの重み付け統合
- **2つの実行モード**:
    - バッチ処理モード: Excel一括処理
    - インタラクティブモード: StreamlitベースのWeb UI
- **高度な最適化**:
    - ベクトルキャッシング (JSON形式)
    - 進捗表示 (tqdmによるプログレスバー)
    - 詳細なロギング

### 技術スタック
- Python 3.7+
- pandas
- Sentence Transformers
- Streamlit
- xlsxwriter
- tqdm
- langchain, langchain-anthropic, langchain-openai
- sudachipy, sudachidict-core
- python-dotenv, openpyxl, pymupdf4llm

- **デモ画面 (インタラクティブモード)**:
  ![Streamlit UI](./docs/images/streamlit_ui.png)
  *(注意: この画像はダミーです。実際には、`docs/images/` ディレクトリに `streamlit_ui.png` という名前のスクリーンショットを配置してください)*

## 目次

1. [システム要件](#システム要件)
2. [セットアップ](#セットアップ)
3. [使用方法](#使用方法)
    - [3.1 入力ファイル要件](#31-入力ファイル要件)
    - [3.2 バッチモード実行手順](#32-バッチモード実行手順)
    - [3.3 インタラクティブモード実行手順](#33-インタラクティブモード実行手順)
4. [設定パラメータ](#設定パラメータ)
5. [入出力形式の変更](#入出力形式の変更)
6. [開発](#開発)
    - [6.1 開発環境構築](#61-開発環境構築)
    - [6.2 コーディング規約](#62-コーディング規約)
7. [トラブルシューティング](#トラブルシューティング)
8. [セキュリティ](#セキュリティ)
9. [ライセンス](#ライセンス)
10. [バグ報告・機能要望](#バグ報告機能要望)

## 1. システム要件

### 1.1 必要な環境

- OS: Windows, macOS, Linux
- Python 3.7以上

### 1.2 依存関係

- pandas
- sentence-transformers
- streamlit
- xlsxwriter
- python-dotenv
- tqdm
- langchain
- langchain-anthropic (LLMプロバイダーとしてAnthropicを使用する場合)
- langchain-openai (LLMプロバイダーとしてOpenAIを使用する場合)
- openpyxl
- sudachipy
- sudachidict-core
- pymupdf4llm

以下のコマンドで依存関係をインストールできます:

```bash
pip install -r requirements.txt
```

## 2. セットアップ

### 2.1 システムコンポーネント

| ファイル/ディレクトリ | 説明 | 用途 |
| --- | --- | --- |
| 📁 input/ | 入力ファイル用ディレクトリ | 質問データを含むExcelファイルを配置 |
| 📁 reference/ | 参照データ用ディレクトリ | 検索対象となる参照用Excelファイルを配置 |
| 📁 reference/vector_cache/ | ベクトルキャッシュディレクトリ | 計算済みの参照データのベクトルをJSON形式で保存 |
| 📁 prompt/ | プロンプトテンプレートディレクトリ | LLM用のプロンプトテンプレートファイル (.txt) を保存 |
| 📁 output/ | 出力ファイル用ディレクトリ | 検索結果のExcelファイルが出力される |
| 📁 logs/ | ログファイル用ディレクトリ | アプリケーションログ (app.log) の保存先 |
| 📄 config.py | 設定管理モジュール | システム全体の設定 (検索パラメータ, LLM設定など) をPythonコードで管理 |
| 📄 main.py | エントリーポイント | プログラムの開始点 (バッチモード or インタラクティブモードの選択) |
| 📄 processor.py | データ処理モジュール | メインの処理ロジック (入力読み込み, 検索, 結果出力) を実装 |
| 📄 searcher.py | 検索エンジンモジュール | ハイブリッド検索 (ベクトル検索 + キーワード検索) のコア機能を実装 |
| 📄 input_handler.py | 入力処理モジュール | 入力ファイルの読み込み (現在はExcelのみ) を担当 |
| 📄 output_handler.py | 出力処理モジュール | 結果の出力 (現在はExcelのみ) を担当 |
| 📄 chat.py | Streamlit UIモジュール | インタラクティブモードのWeb UIを提供 |
| 📄 .env | 環境変数ファイル (オプション) | APIキーなどの機密情報を管理 (利用は任意) |
| 📄 requirements.txt | 依存パッケージリスト | 必要なPythonパッケージを記載 |
| 📄 utils/logger.py | ロガー設定 | ログレベルや出力形式を設定 |
| 📄 README.md | このファイル | プロジェクトの説明、使用方法、開発ガイドなど |

### 2.2 環境変数の設定 (オプション)

LLMプロバイダーとしてOpenAIまたはAnthropicを使用し、APIキーが必要な場合は、.envファイルを作成し、以下のようにAPIキーを設定します:

```env
# .env ファイルの内容 (例)
ANTHROPIC_API_KEY=your_anthropic_api_key
OPENAI_API_KEY=your_openai_api_key
```

重要: .env ファイルは .gitignore に追加されており、Gitリポジトリには含まれません。機密情報が誤って公開されるのを防ぐためです。

## 3. 使用方法

### 3.1 入力ファイル要件

| ファイルタイプ | 場所 | 必須列 | 形式 |
|--------------|------|--------|------|
| 質問データ | input/ | 1列目: 番号<br>2列目: 質問内容<br>3列目: 回答(任意) | Excel (.xlsx) |
| 参照データ | reference/ | ・問合せ内容<br>・回答 | Excel (.xlsx) |

### 3.2 バッチモード実行手順
1. 入力ファイルの配置
   - 質問データを `input/` に配置
   - 参照データを `reference/` に配置
2. コマンド実行
   ```bash
   python main.py
   ```
3. 結果確認
   - `output/` に `output_batch_v{ベクトルの重み}_k{キーワードの重み}_{タイムスタンプ}.xlsx` が生成
   - `logs/app.log` に処理ログが出力

### 3.3 インタラクティブモード実行手順
1. 起動コマンド実行（いずれか）
   ```bash
   python main.py interactive
   # または
   streamlit run chat.py
   ```
2. Web UI操作
   - 質問を入力し「送信」をクリック
   - 類似質問と回答が表示
   - 「チャット履歴を保存」で対話履歴をExcel保存（ `output/` に `output_chat_v{ベクトルの重み}_k{キーワードの重み}_{タイムスタンプ}.xlsx` が生成）

## 4. 設定パラメータ

| パラメータ | 説明 | デフォルト値 | 設定可能な値 |
|-----------|------|------------|------------|
| top_k | 類似文書の取得件数 | 4 | 1以上の整数 |
| model_name | 埋め込みモデル | intfloat/multilingual-e5-base | SentenceTransformersの対応モデル |
| vector_weight | ベクトル検索の重み | 0.9 (バッチ)<br>0.7 (UI) | 0.0～1.0 |
| llm_provider | LLMプロバイダ | anthropic | anthropic, openai |
| llm_model | 使用モデル | claude-3-5-sonnet-20241022 | プロバイダの対応モデル |
| base_dir | 基準ディレクトリ | "." | 有効なパス |
| input_type | 入力形式 | excel | excel (※将来的に拡張予定) |
| output_type | 出力形式 | excel | excel (※将来的に拡張予定) |

### 設定方法
1. 直接編集: `config.py` の `SearchConfig` クラスを編集
2. 環境変数: `.env` ファイルで指定（APIキーのみ）
   ```env
   ANTHROPIC_API_KEY=your_api_key
   OPENAI_API_KEY=your_api_key  # OpenAI使用時
   ```

## 5. 入出力形式の変更

現状では、入力・出力ともにExcel形式のみをサポートしています。 将来的には、input_handler.py と output_handler.py を拡張することで、CSVやJSONなど他の形式にも対応可能です。

## 6. 開発

### 6.1 開発環境構築

リポジトリをクローン:

```bash
git clone https://github.com/jsakuta/rag_yokin.git
cd rag_yokin
```

仮想環境を作成し、アクティベート:

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

依存関係をインストール:

```bash
pip install -r requirements.txt
```

### 6.2 コーディング規約

- PEP 8 に準拠します。
- コメント、docstringは日本語で記述します。
- 関数、クラスにはdocstringを付与し、型ヒントを積極的に使用します。
- 長い行は80文字以内で折り返します。
- コミットメッセージも日本語で記述してかまいませんが、具体的かつ簡潔に記述してください。

## 7. トラブルシューティング

### 7.1 よくある問題と解決方法

キャッシュ関連のエラー:

- reference/vector_cache/ ディレクトリが存在することを確認してください。
- キャッシュファイル (reference/vector_cache/cache.json) を削除し、再生成を試みてください。

メモリエラー:

- 非常に大きなExcelファイルを処理する場合、より高性能なマシンを使用するか、将来的に分割処理の実装を検討してください。

### 7.2 ログの確認

アプリケーションのログは logs/app.log に出力されます。

ログレベルは utils/logger.py で変更できます (デフォルトはINFO)。問題が発生した場合は、ログレベルをDEBUGに変更して詳細な情報を確認してください。

## 8. セキュリティ

⚠️ 重要な注意事項:

- APIキーなどの機密情報は .env ファイルで管理し、絶対に Git リポジトリに含めないでください。
- 定期的に依存パッケージを更新し、セキュリティ脆弱性を解消してください。
- 本システムを本番環境で利用する場合は、入力データの検証、アクセス制御、適切な認証・認可の仕組みを導入するなど、セキュリティ対策を講じてください。

## 9. ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は LICENSE ファイルをご覧ください。

## 10. バグ報告・機能要望

GitHub Issues にて受け付けています。バグ報告の際は、以下の情報を含めてください:

- エラーメッセージ
- logs/app.log の内容 (可能な限り)
- 再現手順
- 利用環境 (OS, Pythonバージョンなど)

機能要望も歓迎します。