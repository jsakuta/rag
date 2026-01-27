# Excel色付け＆Markdown比較ファイル生成機能 実装プラン

## 概要

変更前シナリオ生成スクリプトに、以下の機能を追加：
1. **Excel色付け**: 生成された「変更前シナリオ」ファイルで、修正前データで置き換えた行に背景色を設定
2. **Markdown比較ファイル**: 各改定ごとに、修正前と修正後の内容を並べて表示するMDファイルを生成（6改定 = 6ファイル）

## 要件

### 1. Excel色付け
- 対象：`データ整理/変更前シナリオ/改定名/変更前シナリオ_xxx.xlsx`
- 置換した行（修正前データで上書きした範囲）に黄色（または目立つ色）の背景色を設定
- openpyxlライブラリを使用

### 2. Markdown比較ファイル
- 出力先：`データ整理/改定フォルダ名/変更内容比較.md`
- 各改定で6ファイル生成
- 内容：
  - 改定名と概要
  - 各ボットごとのセクション（複数ボット対象の場合）
  - 修正前と修正後の内容を並べて表示（テキスト形式）
  - 差分箇所を強調表示

## 現状分析

### 既存スクリプト（generate_before_scenarios.py）
- `_apply_category_replacement`: カテゴリ単位の置換（first~lastの範囲を計算）
- `_full_replace`: 全置換（torikaku-bot用）
- `run_job`: メイン処理、pandas DataFrameでExcel出力

**課題**：
- 置換した行範囲（first, last）が関数内に閉じており、`run_job`に返されていない
- 現在は`DataFrame.to_excel()`でシンプル出力（スタイルなし）

## 実装アプローチ

### Phase 1: データ構造の拡張

置換情報を記録するためのデータクラスを追加：

```python
@dataclass(frozen=True)
class ReplacementInfo:
    """置換した範囲の情報"""
    first_row: int  # DataFrameのインデックス（0始まり）
    last_row: int
    key_label: str  # "Lv1=カード関連" など
    before_file_name: str
```

### Phase 2: 関数の修正

#### `_apply_category_replacement`の戻り値拡張
```python
def _apply_category_replacement(
    merged_df: pd.DataFrame,
    before_df: pd.DataFrame
) -> tuple[pd.DataFrame, str, ReplacementInfo]:
    # 既存処理 + ReplacementInfoの生成
    first, last = _get_contiguous_slice_indices(mask)
    # ...
    info = ReplacementInfo(
        first_row=first,
        last_row=first + len(replacement) - 1,  # 置換後の範囲
        key_label=key_label,
        before_file_name="..."
    )
    return new_df, key_label, info
```

#### `_full_replace`も同様に拡張
```python
def _full_replace(
    merged_df: pd.DataFrame,
    before_df: pd.DataFrame
) -> tuple[pd.DataFrame, ReplacementInfo]:
    # ...
    info = ReplacementInfo(
        first_row=0,
        last_row=len(replacement) - 1,
        key_label="FULL_REPLACE",
        before_file_name="..."
    )
    return replacement, info
```

#### `run_job`で置換情報を収集
```python
def run_job(...) -> None:
    # ...
    replacement_infos: list[ReplacementInfo] = []

    if job.mode == "full_replace":
        out_df, info = _full_replace(merged_df, before_df)
        replacement_infos.append(info)
    else:
        for f in bot_before_files:
            out_df, key_label, info = _apply_category_replacement(out_df, before_df)
            replacement_infos.append(info)

    # Excel出力（スタイル付き）
    write_excel_with_highlighting(out_path, out_df, replacement_infos)
```

### Phase 3: Excel色付け機能

新規関数を追加：

```python
from openpyxl import load_workbook
from openpyxl.styles import PatternFill

def write_excel_with_highlighting(
    output_path: Path,
    df: pd.DataFrame,
    replacement_infos: list[ReplacementInfo],
) -> None:
    """
    DataFrameをExcelに出力し、置換した行に背景色を設定
    """
    # 1. pandasでExcel出力
    df.to_excel(output_path, index=False, sheet_name="Sheet1")

    # 2. openpyxlで開いてスタイル設定
    wb = load_workbook(output_path)
    ws = wb.active

    # 黄色の背景色
    yellow_fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")

    # 置換した範囲に色を付ける
    for info in replacement_infos:
        # DataFrameのインデックス → Excelの行番号（+2：ヘッダー行 + 0始まり調整）
        excel_first_row = info.first_row + 2
        excel_last_row = info.last_row + 2

        for row_idx in range(excel_first_row, excel_last_row + 1):
            for cell in ws[row_idx]:
                cell.fill = yellow_fill

    wb.save(output_path)
```

### Phase 4: Markdown比較ファイル生成

#### 改定ごとのJob集約

```python
def group_jobs_by_revision(jobs: list[Job]) -> dict[str, list[Job]]:
    """改定フォルダ名でJobをグルーピング"""
    grouped: dict[str, list[Job]] = {}
    for job in jobs:
        if job.revision_dir_name not in grouped:
            grouped[job.revision_dir_name] = []
        grouped[job.revision_dir_name].append(job)
    return grouped
```

#### Markdown生成関数

```python
def generate_comparison_markdown(
    base_dir: Path,
    revision_dir_name: str,
    jobs_for_revision: list[Job],
) -> None:
    """
    改定ごとの比較MDファイルを生成

    出力先: データ整理/{revision_dir_name}/変更内容比較.md
    """
    data_root = base_dir / "データ整理"
    revision_dir = data_root / revision_dir_name
    output_path = revision_dir / "変更内容比較.md"

    lines = []

    # ヘッダー
    lines.append(f"# {revision_dir_name} - 変更内容比較")
    lines.append("")
    lines.append(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # 各ボットごとのセクション
    for job in jobs_for_revision:
        lines.append(f"## {job.bot}")
        lines.append("")

        # 修正前/修正後ファイルを取得
        all_before = _iter_scenario_excels(revision_dir, "修正前")
        all_after = _iter_scenario_excels(revision_dir, "修正後")

        bot_before = _filter_files_for_bot(all_before, job.bot)
        bot_after = _filter_files_for_bot(all_after, job.bot)

        # ファイルペアを作成（ファイル名で対応付け）
        pairs = _match_before_after_pairs(bot_before, bot_after)

        for before_path, after_path in pairs:
            lines.append(f"### ファイル: {before_path.name}")
            lines.append("")

            # Excelを読み込んで比較
            before_df = _read_excel_first_sheet(before_path)
            after_df = _read_excel_first_sheet(after_path)

            # 差分を計算して表示
            diff_lines = _compute_text_diff(before_df, after_df)
            lines.extend(diff_lines)
            lines.append("")

    # ファイル出力
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated comparison MD: {output_path}")
```

#### 差分計算関数

```python
def _match_before_after_pairs(
    before_files: list[Path],
    after_files: list[Path],
) -> list[tuple[Path, Path]]:
    """
    修正前と修正後のファイルをペアリング
    カテゴリ名（カード関連、喪失など）でマッチング
    """
    pairs = []
    for before in before_files:
        # ファイル名からカテゴリを抽出（例：諸届、カード関連など）
        before_name = before.stem

        # 対応する修正後ファイルを探す
        for after in after_files:
            after_name = after.stem
            # カテゴリ名が含まれているかチェック（簡易的なマッチング）
            if _extract_category(before_name) == _extract_category(after_name):
                pairs.append((before, after))
                break

    return pairs


def _extract_category(file_stem: str) -> str:
    """
    ファイル名からカテゴリを抽出
    例: "⑥スマイル機能変更_修正前_シナリオ_スマイルタブレット_諸届_20250718 (1)"
        → "諸届"
    """
    # 日付とカッコを除去
    cleaned = re.sub(r"_\d{8}.*$", "", file_stem)
    # 最後のアンダースコア以降を取得
    parts = cleaned.split("_")
    return parts[-1] if parts else file_stem


def _compute_text_diff(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
) -> list[str]:
    """
    DataFrameの差分をテキスト形式で生成

    出力形式例:
    ---
    変更箇所 1:

    【変更前】 行54, Lv3
    【JCB・DC】 「JCB・DCクレジット・デビット単体型カード再発行」メニューで手続きしてください

    【変更後】 行54, Lv3
    【JCB・MDC】 「JCB・MDCクレジット・デビット単体型カード再発行」メニューで手続きしてください
    ---
    """
    lines = []

    # 行数の違いをチェック
    if len(before_df) != len(after_df):
        lines.append(f"**注意**: 行数が異なります（変更前: {len(before_df)}行、変更後: {len(after_df)}行）")
        lines.append("")

    # 共通する行数まで比較
    min_rows = min(len(before_df), len(after_df))
    lv_columns = [c for c in before_df.columns if c.startswith("Lv")]

    change_count = 0
    for i in range(min_rows):
        for col in lv_columns:
            if col not in before_df.columns or col not in after_df.columns:
                continue

            before_val = str(before_df.iloc[i][col]) if pd.notna(before_df.iloc[i][col]) else ""
            after_val = str(after_df.iloc[i][col]) if pd.notna(after_df.iloc[i][col]) else ""

            if before_val != after_val and before_val and after_val:  # 両方空でない場合のみ
                change_count += 1
                lines.append("---")
                lines.append(f"変更箇所 {change_count}:")
                lines.append("")
                lines.append(f"**【変更前】** 行{i+2}, {col}")
                lines.append(f"```")
                lines.append(before_val[:500])  # 長すぎる場合は切り詰め
                lines.append(f"```")
                lines.append("")
                lines.append(f"**【変更後】** 行{i+2}, {col}")
                lines.append(f"```")
                lines.append(after_val[:500])
                lines.append(f"```")
                lines.append("")

    if change_count == 0:
        lines.append("差分なし（変更前と変更後で内容が一致）")
    else:
        lines.append(f"**合計 {change_count} 箇所の変更**")

    return lines
```

### Phase 5: メイン処理の統合

```python
def main() -> int:
    # ... 引数パース ...

    # Jobを改定ごとにグルーピング
    jobs_by_revision = group_jobs_by_revision(jobs)

    # 各Jobを実行（Excel生成＋色付け）
    for j in jobs:
        run_job(
            base_dir=base_dir,
            output_dir=output_dir,
            job=j,
            write=bool(args.write),
            overwrite=bool(args.overwrite),
        )

    # Markdown比較ファイル生成
    if args.write:  # dry-runでない場合のみ
        for revision_name, revision_jobs in jobs_by_revision.items():
            generate_comparison_markdown(
                base_dir=base_dir,
                revision_dir_name=revision_name,
                jobs_for_revision=revision_jobs,
            )

    return 0
```

## 実装ステップ

1. **データ構造追加**
   - `ReplacementInfo`クラスを定義

2. **既存関数の修正**
   - `_apply_category_replacement`の戻り値に`ReplacementInfo`を追加
   - `_full_replace`も同様に修正
   - `run_job`で置換情報を収集

3. **Excel色付け機能**
   - `write_excel_with_highlighting`関数を実装
   - openpyxlを使ってスタイル設定

4. **Markdown生成機能**
   - `group_jobs_by_revision`を実装
   - `_match_before_after_pairs`を実装
   - `_extract_category`を実装
   - `_compute_text_diff`を実装
   - `generate_comparison_markdown`を実装

5. **メイン処理統合**
   - `main`関数にMarkdown生成処理を追加

6. **テスト**
   - `--write`オプションで実行
   - 生成されたExcelファイルの色付けを確認
   - 生成されたMarkdownファイルの内容を確認

## 出力ファイル

### 1. 色付きExcel（既存出力先）
```
データ整理/変更前シナリオ/
├── ①スマイル機能変更メンテ台帳20/
│   └── 変更前シナリオ_smile-bot.xlsx  ← 置換箇所が黄色
├── ②相続少額払いメンテ台帳21/
│   └── 変更前シナリオ_souzoku-bot.xlsx
...
```

### 2. Markdown比較ファイル（新規）
```
データ整理/
├── ①スマイル機能変更メンテ台帳20/
│   ├── 変更内容比較.md  ← 新規作成
│   ├── 修正前/
│   └── 修正後/
├── ②相続少額払いメンテ台帳21/
│   ├── 変更内容比較.md  ← 新規作成
│   ├── 修正前/
│   └── 修正後/
...
```

## 重要な実装上の注意

### 1. DataFrame行インデックス → Excel行番号の変換
- DataFrameは0始まり、Excelは1始まり
- Excelにはヘッダー行がある（1行目）
- 変換式：`excel_row = df_index + 2`

### 2. 置換範囲の正確な記録
- `_apply_category_replacement`では`pd.concat`で新しいDataFrameを作成
- 置換後の範囲：`first_row`～`first_row + len(replacement) - 1`

### 3. ファイルペアリング
- 修正前と修正後のファイル名が完全一致しない場合がある
- カテゴリ名（諸届、カード関連など）で柔軟にマッチング

### 4. 改定フォルダ名の一貫性
- JOBSの`revision_dir_name`を使用して改定を識別
- 同じ改定で複数ボットがある場合、改定名でグルーピング

## 検証方法

1. **Excel色付けの確認**
   ```bash
   python scripts/generate_before_scenarios.py --write --overwrite --only-revision "①スマイル機能変更メンテ台帳20"
   ```
   - 生成されたExcelファイルを開いて、黄色い行が置換箇所と一致するか確認

2. **Markdown比較ファイルの確認**
   - `データ整理/①スマイル機能変更メンテ台帳20/変更内容比較.md`を開く
   - 変更前と変更後の内容が並んで表示されているか確認
   - 差分箇所が強調されているか確認

3. **全改定の一括生成**
   ```bash
   python scripts/generate_before_scenarios.py --write --overwrite
   ```
   - 6つの改定すべてでExcelとMarkdownが生成されるか確認

## Critical Files

- `c:\VSCode\rag\rag-gemini\scripts\generate_before_scenarios.py` - メインスクリプト（大幅に修正）
- `c:\VSCode\rag\rag-gemini\requirements.txt` - openpyxlが含まれていることを確認済み
