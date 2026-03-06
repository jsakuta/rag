# アノテーション品質修正（第4ラウンド）

## Context

第3ラウンドのレビューで3点のフィードバック:
1. 運用保守 13_keyword_search: ②「全件返却キャプション」は操作ではないので表から除外し補記にする
2. 運用保守 01_launch: ②モード選択の位置がまだ真ん中(x=100)、黄色ハイライトが広すぎ
3. 回答支援 as_07_chat_history: ステップ説明の内容は合っているがアノテーション(3個)とステップ(2個)が不一致

---

## 修正一覧

### 1. 運用保守 13_keyword_search: ②を表から補記に移動

**現状（step_descriptions.json）**:
```json
"steps": [
  {"num": 1, "action": "キーワード検索を選択", "desc": "..."},
  {"num": 2, "action": "全件返却キャプション", "desc": "キーワード検索ではマッチする全件が..."}
]
```
**問題**: 「全件返却キャプション」は操作(action)ではなく、UIの説明テキストに過ぎない

**修正**:
- `steps` から ② を削除（①のみ残す）
- `text` として表の後に補記を追加

```json
{
  "heading": "5.1 キーワード検索",
  "image": "13_keyword_search",
  "steps": [
    {"num": 1, "action": "キーワード検索を選択", "desc": "検索タイプで「キーワード検索」を選択すると、ベクトル検索をスキップしてキーワード一致のみで検索します。"}
  ],
  "text": "キーワード検索ではマッチする全件が返却されます。候補数の設定は適用されません。用語の単純な置き換え（例: 改定⑤「AMLフィルター」→「GPLEX」）の検出に適しています。"
}
```

**annotation_config.json**: ①のみ（既に修正済み）。変更不要。

### 2. 運用保守 01_launch: ②左寄せ + ハイライト縮小

**現状**:
- ② (100, 163) "モード選択" — サイドバー中央、ラベル付き
- highlight: `{"x1": 0, "y1": 70, "x2": 290, "y2": 700}` — サイドバー全体を黄色で覆っている

**修正**:
- ② x=100→15, label "モード選択"→""（左寄せ、ラベル除去）
- サイドバー全体ハイライト削除（円で要素を指しているので不要。検索フォームのハイライトは残す）

```json
"01_launch": {
  "circles": [
    {"num": 1, "x": 370, "y": 115, "label": "タイトル"},
    {"num": 2, "x": 15, "y": 163, "label": ""},
    {"num": 3, "x": 380, "y": 280, "label": "検索フォーム"}
  ],
  "highlights": [
    {"x1": 370, "y1": 300, "x2": 1190, "y2": 420}
  ]
}
```

### 3. 回答支援 as_07_chat_history: アノテーション数をステップに合わせる

**現状（annotation_config）**: 3個の円
- ① (650, 345) "2回目のクエリ"
- ② (420, 475) "2回目の検索結果"
- ③ (15, 710) ""

**現状（step_descriptions）**: 2ステップ
- ① 複数回の検索
- ② チャット履歴を保存

**問題**: 円が3個、ステップが2個で不一致。ラベルも不一致。

**修正**: 2個に合わせる
- ① 2回目のクエリバブル付近 → 「複数回の検索」を示す
- ② サイドバーの「チャット履歴を保存」ボタン

```json
"as_07_chat_history": {
  "circles": [
    {"num": 1, "x": 650, "y": 345, "label": ""},
    {"num": 2, "x": 15, "y": 710, "label": ""}
  ],
  "highlights": [
    {"x1": 560, "y1": 375, "x2": 1850, "y2": 415}
  ]
}
```

---

## 修正対象ファイル

| ファイル | 変更内容 |
|---------|---------|
| `build_guide.js` | `note` フィールドサポート追加（steps後にテキスト表示） |
| `revision_ops/step_descriptions.json` | 5.1: ②削除 + `note`補記追加 |
| `revision_ops/annotation_config.json` | 01: ②左寄せ+ラベル空, サイドバーハイライト削除 |
| `answer_support/annotation_config.json` | as_07: 3円→2円に削減 |

## 実装手順

1. `build_guide.js` に `note` フィールドサポート追加
2. `revision_ops/step_descriptions.json` 修正（5.1 の②削除 + note追加）
3. `revision_ops/annotation_config.json` 修正（01_launch の②左寄せ+ハイライト縮小）
4. `answer_support/annotation_config.json` 修正（as_07 の3円→2円）
5. `python annotate.py answer_support` → as_07 目視確認
6. `python annotate.py revision_ops` → 01_launch 目視確認
7. `node build_guide.js answer_support` で docx 再ビルド
8. `node build_guide.js revision_ops` で docx 再ビルド

## 検証

- 01_launch: ②が左端(x=15)、サイドバー全体の黄色ハイライトがないこと
- 13_keyword_search: docx内の表が①のみ、その下に補記テキストがあること
- as_07: ①②の2個のみ、①がクエリバブル付近、②がチャット履歴保存ボタン付近

### build_guide.js の note フィールド追加

`buildSubsections()` のレンダリング順序: `text` → `code` → `bullets` → `image` → `steps` → `steps_group`
→ 既存の `text` / `bullets` はすべて steps の**前**に出る。

**対策**: `note` フィールドを追加（表の後に出るテキスト）

**build_guide.js 修正箇所**:
1. `buildSubsections()` (行244-246付近): `sub.steps` の後に `if (sub.note) children.push(bodyText(sub.note));` を追加
2. トップレベルセクション (行185-187付近): 同様に `section.steps` の後に追加
