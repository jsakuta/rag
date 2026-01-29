# 事務改定差分.mdの統一フォーマット化 & 正解ID対応表生成

## 概要

1. **事務改定差分.mdを統一フォーマットに修正**
   - 6ファイルすべて同じ体裁に統一
   - カテゴリ内行番号とExcel行番号の両方を記載
   - 省略（「他X行」）を展開

2. **正解ID対応表を生成**
   - 多段階検索の入力ファイル（番号, 改定内容, 正解ID）を自動生成

---

## フェーズ1: 事務改定差分.mdの統一フォーマット化

### 統一フォーマット

```markdown
# ①スマイル機能変更メンテ台帳20 - 事務改定差分

## 📋 メンテナンス管理台帳との照合

**台帳No.20の記載**:
- ボット名: スマイル
- 大分類: 諸届
- 変更箇所: 行番号28, 84, 85
- 変更内容: 28,84:【本人確認】の文言変更。85:「キャッシュカード暗証...」追加

## smile-bot

### ファイル: ①スマイル機能変更_修正前_シナリオ_スマイルタブレット_諸届_20250718.xlsx

**カテゴリ**: Lv1=諸届
**変更前シナリオExcelでの範囲**: 行104～行234

**変更行一覧（メンテ台帳 vs 実際の差分）**:

| 台帳記載行 | Excel行 | 実際の差分 | 状態 |
|-----------|---------|-----------|------|
| 28 | 131 | なし | ⚠️ 不一致 |
| 84 | 187 | なし | ⚠️ 不一致 |
| 85 | 188 | なし | ⚠️ 不一致 |
| - | 129 | あり | 📝 台帳未記載 |
| - | 185 | あり | 📝 台帳未記載 |

**⚠️ 不一致の詳細**:
メンテナンス管理台帳の記載と実際のファイル差分が異なります。
ファイルの差し替えまたは台帳の確認が必要です。

---
変更箇所 1: **カテゴリ内行26** (Excel行129)
...
```

### 変換式

```
カテゴリ内行番号 = Excel行番号 - カテゴリ開始行 + 1
```

例：諸届カテゴリ（範囲: 行104～行234）
- Excel行129 → カテゴリ内行 = 129 - 104 + 1 = **26**

### 重要：データソースの信頼性

| ソース | 信頼性 | 用途 |
|--------|--------|------|
| **メンテ台帳H列の行番号** | ✓ 正解 | 正解IDの生成元 |
| **修正前/後ファイルの実際の差分** | △ 参考 | 不一致がある可能性 |

→ 事務改定差分.mdには**両方を記載**し、不一致があれば明示する

### メンテ台帳H列の行番号形式

```
行番号：1、5、12、13、18～25、30～32
行番号：4、35、83
行番号：54～62、64
```

- カンマ/読点区切り: `1、5、12、13`
- 範囲指定: `18～25`（全角チルダ）
- 混在: `54～62、64`

### 修正対象ファイル（6ファイル）

| # | ファイル | 対応する台帳No. | 現状の問題 |
|---|----------|----------------|-----------|
| ① | ①事務改定差分.md | No.20 | Excel行番号のみ、台帳との照合必要 |
| ② | ②事務改定差分.md | No.21 | 「他20行」省略、修正後ファイル不完全 |
| ③ | ③事務改定差分.md | No.25-30,35-36 | 複数台帳対応、照合必要 |
| ④ | ④事務改定差分.md | No.37 | OK（1行のみ）|
| ⑤ | ⑤事務改定差分.md | No.41-42 | 「他X行」省略 |
| ⑥ | ⑥事務改定差分.md | No.43-45 | 「他8行」省略、⑤と喪失重複 |

---

## フェーズ2: 正解ID対応表生成

### データソース

### 1. 事務改定内容フォルダ
- **パス**: `データ整理/事務改定内容/`
- **ファイル**:
  - `①スマイル機能変更.md` → 改定内容テキスト
  - `②相続少額払い.md`
  - `③保険証→資格確認証.md`
  - `④難易度高_０円新規nn.md`
  - `⑤AMLフィルター→GPLEX.md`
  - `⑥DC→MDC.md`

### 2. 事務改定差分.md（各メンテ台帳フォルダ内）
- **パス例**: `データ整理/①スマイル機能変更メンテ台帳20/①事務改定差分.md`
- **構造**:
  ```markdown
  ## smile-bot
  ### ファイル: ...
  **黄色ハイライト行（変更前シナリオExcel）**: 2行
  - 行番号: 129, 185

  ## naibujimu-bot
  ...
  - 行番号: 96, 99
  ```

### 3. フォルダ名とメンテナンス台帳No.の対応
- `①スマイル機能変更メンテ台帳20` → No.20
- `②相続少額払いメンテ台帳21` → No.21
- `③保険証→資格確認証メンテ台帳25.26.27.28.29.30.35.36` → No.25,26,27,28,29,30,35,36
- `④難易度高_0円新規開設可能メンテ台帳37` → No.37
- `⑤AMLフィルター→GPLEXメンテ台帳41.42` → No.41,42
- `⑥DC→MDCメンテ台帳43.44.45` → No.43,44,45

---

## 正解IDの抽出ルール

事務改定差分.mdから以下を抽出：

1. **ボット名**: `## smile-bot` などのセクションヘッダー
2. **行番号**: `- 行番号: 129, 185` の形式

**正解ID形式**: `{ボット名}_{行番号}`
- 例: `smile-bot_129`, `smile-bot_185`, `naibujimu-bot_96`
- 複数ボットにまたがる場合はすべて含める

---

## 出力形式

### 入力ファイル形式（multi_stage検索用）

| 番号 | 改定内容 | 正解ID |
|------|----------|--------|
| 1 | 個人事業主の場合でもカードによる本人認証を認める。 | smile-bot_129, smile-bot_185 |
| 2 | １．内容\n○健康保険証等は2025年12月2日... | smile-bot_366, naibujimu-bot_96, naibujimu-bot_99, ... |

---

## 実装計画

### スクリプト: `scripts/generate_correct_ids.py`

```python
import re
import os
import pandas as pd
from pathlib import Path

DATA_DIR = Path("データ整理")

def main():
    results = []

    # 事務改定内容フォルダのmdファイルを処理
    revision_dir = DATA_DIR / "事務改定内容"
    for md_file in sorted(revision_dir.glob("*.md")):
        # ①②③...の番号を取得
        prefix = md_file.stem[0]  # "①", "②" など

        # 対応する事務改定差分.mdを探す
        diff_md = find_diff_md(prefix)
        if not diff_md:
            continue

        # 改定内容を読み込み
        revision_content = md_file.read_text(encoding='utf-8').strip()

        # 事務改定差分.mdから正解IDを抽出
        correct_ids = extract_correct_ids(diff_md)

        results.append({
            '番号': prefix,
            '改定内容': revision_content,
            '正解ID': ', '.join(correct_ids)
        })

    # Excel出力
    df = pd.DataFrame(results)
    df.to_excel("input/multi_stage_input.xlsx", index=False)
    print(f"出力完了: {len(results)}件")

def find_diff_md(prefix: str) -> Path | None:
    """プレフィックスに対応する事務改定差分.mdを検索"""
    for folder in DATA_DIR.iterdir():
        if folder.is_dir() and folder.name.startswith(prefix):
            diff_file = folder / f"{prefix}事務改定差分.md"
            if diff_file.exists():
                return diff_file
    return None

def extract_correct_ids(diff_md: Path) -> list[str]:
    """事務改定差分.mdから正解IDを抽出"""
    content = diff_md.read_text(encoding='utf-8')

    correct_ids = []
    current_bot = None

    for line in content.split('\n'):
        # ボット名セクション: ## smile-bot
        if line.startswith('## ') and '-bot' in line:
            current_bot = line[3:].strip()

        # 行番号: - 行番号: 129, 185
        if current_bot and line.strip().startswith('- 行番号:'):
            numbers_str = line.split(':')[1].strip()
            numbers = parse_row_numbers(numbers_str)
            for num in numbers:
                correct_ids.append(f"{current_bot}_{num}")

    return correct_ids

def parse_row_numbers(text: str) -> list[int]:
    """行番号文字列をパース（カンマ区切り対応）"""
    # 全角→半角変換
    text = text.translate(str.maketrans('０１２３４５６７８９', '0123456789'))

    numbers = []
    for part in text.split(','):
        part = part.strip()
        if part.isdigit():
            numbers.append(int(part))

    return numbers

if __name__ == "__main__":
    main()
```

---

## 修正対象ファイル

| ファイル | 内容 |
|----------|------|
| `scripts/generate_correct_ids.py` | 新規作成：対応表生成スクリプト |

---

## 検証方法

1. スクリプト実行
   ```bash
   python scripts/generate_correct_ids.py
   ```

2. 出力ファイル確認
   - `input/multi_stage_input.xlsx` が生成されることを確認
   - 列構成: 番号, 改定内容, 正解ID

3. サンプル検証
   - ①の正解ID: `smile-bot_129, smile-bot_185`
   - ③の正解ID: `smile-bot_366, naibujimu-bot_96, naibujimu-bot_99, ...`
