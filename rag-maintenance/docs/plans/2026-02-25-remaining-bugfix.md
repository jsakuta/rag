# 残存バグ修正 実装計画

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 引き継ぎ前のバグ検証で発見された残存バグ3件を修正し、修正後の不整合がないことを確認する

**Architecture:** 3つの独立した修正（sharepoint folderUrl / extractSafeTopN / キャッシュフォールバック）を順番に実施。各修正は他に影響を与えない。テストスイートが存在しないため `tsc --noEmit` で型整合性を検証する。

**Tech Stack:** TypeScript, M365 Agents SDK, Azure AI Search SDK, Microsoft Graph SDK

---

## 修正対象と修正しない項目

### 修正する（3件）

| # | 場所 | 内容 | 信頼度 |
|---|------|------|--------|
| A | `sharepoint.ts:55` | `parentReference?.webUrl` が Graph API に存在しない → folderUrl 常に空 | 95% |
| B | `agent.ts:847-853` | `extractSafeTopN` が `targetType` を無視 → FAQタブの件数変更が効かない | 90% |
| C | `agent.ts:898-903` | キャッシュミス時のカテゴリフォールバックが全カテゴリ → シナリオ/FAQ混在 | 85% |

### 修正しない（1件 — 設計ノート）

| # | 場所 | 内容 | 判断理由 |
|---|------|------|---------|
| D | `cards.ts:755` | 「検索結果に戻る」に perPage 未送信 | `buildResultCard` の二分探索が page=1 で再実行され正しい結果を返す。`buildResultCard` の戻り値型を変更すると3箇所の呼び出し元に波及し、不整合リスクが修正利益を上回る |

---

## Task 1: sharepoint.ts — folderUrl の導出方法を修正

**Files:**
- Modify: `maintenance-bot/src/sharepoint.ts:52-56`

**問題の詳細:**

Graph API の `PUT /drives/{id}/root:/{path}:/content` レスポンス（DriveItem）の `parentReference` は `ItemReference` 型で、`webUrl` プロパティを持たない。そのため `response.parentReference?.webUrl` は常に `undefined` になり、フォールバック `getFolderWebUrl()` が毎回呼ばれる（余分な API コール）。

**修正方針:**

アップロードレスポンスの `response.webUrl`（ファイルのWebURL）から末尾のファイル名部分を除去してフォルダURLを導出する。`getFolderWebUrl()` フォールバックは保険として残す。

**Step 1: sharepoint.ts を修正**

```typescript
// 修正前（sharepoint.ts:52-56）
return {
  webUrl: response.webUrl,
  filename,
  folderUrl: response.parentReference?.webUrl ?? "",
};

// 修正後
const fileWebUrl: string = response.webUrl ?? "";
const lastSlash = fileWebUrl.lastIndexOf("/");
return {
  webUrl: fileWebUrl,
  filename,
  folderUrl: lastSlash > 0 ? fileWebUrl.substring(0, lastSlash) : "",
};
```

**影響範囲の確認:**

- `SpoUploadResult` 型（`sharepoint.ts:6-10`）: 変更なし
- `agent.ts:438` の `succeeded[0].folderUrl || await getFolderWebUrl()` フォールバック: `folderUrl` が正しく取得できるようになるため `getFolderWebUrl()` は通常呼ばれなくなるが、安全弁として残す
- `cards.ts:914-920` の `if (folderUrl)` ガード: `folderUrl` が空でなくなるため「フォルダを開く」ボタンが安定表示される

**Step 2: コンパイル確認**

Run: `cd maintenance-bot && npx tsc --noEmit`
Expected: エラー0件

**Step 3: コミット**

```bash
git add maintenance-bot/src/sharepoint.ts
git commit -m "fix: SPO folderUrl を webUrl から導出（parentReference.webUrl は Graph API に存在しない）"
```

---

## Task 2: agent.ts — extractSafeTopN で targetType を考慮

**Files:**
- Modify: `maintenance-bot/src/agent.ts:847-853`（関数定義）
- Modify: `maintenance-bot/src/agent.ts:148`（searchSemantic 呼び出し）
- Modify: `maintenance-bot/src/agent.ts:195`（searchKeyword 呼び出し）
- **変更不要**: `maintenance-bot/src/agent.ts:237`（searchPage 呼び出し — 後述）

**問題の詳細:**

Adaptive Card の `ToggleVisibility` は CSS 表示を切り替えるだけで、非表示セクションの Input 値もフォーム送信に含まれる。FAQタブで `topN_faq=50` を選んで検索しても、非表示のシナリオタブの `topN=30`（デフォルト）が先に読まれて `30` が返る。

**修正方針:**

`extractSafeTopN` に `targetType` パラメータを追加（オプショナル）。`targetType === "faq"` なら `topN_faq` を優先、それ以外なら `topN` を優先。`targetType` 未指定時は従来通り `topN` → `topN_faq` の順（searchPage ハンドラーとの後方互換性を維持）。

**Step 1: extractSafeTopN 関数を修正**

```typescript
// 修正前（agent.ts:847-853）
function extractSafeTopN(data: Record<string, unknown>): number {
  // シナリオタブは "topN"、FAQタブは "topN_faq" のIDを使用
  let raw = extractNumber(data, "topN", -1);
  if (raw < 0) raw = extractNumber(data, "topN_faq", DEFAULT_TOP_N);
  if (raw < 0) raw = DEFAULT_TOP_N;
  return Math.min(Math.max(raw, 10), 100);
}

// 修正後
function extractSafeTopN(data: Record<string, unknown>, targetType?: SearchTargetType): number {
  // targetType に応じて優先フィールドを切り替え
  // searchPage からは targetType なしで呼ばれる（pageData に topN が数値で埋め込まれるため問題ない）
  const primaryKey = targetType === "faq" ? "topN_faq" : "topN";
  const fallbackKey = targetType === "faq" ? "topN" : "topN_faq";
  let raw = extractNumber(data, primaryKey, -1);
  if (raw < 0) raw = extractNumber(data, fallbackKey, DEFAULT_TOP_N);
  if (raw < 0) raw = DEFAULT_TOP_N;
  return Math.min(Math.max(raw, 10), 100);
}
```

**Step 2: searchSemantic の呼び出しを修正**

```typescript
// 修正前（agent.ts:148 付近）
const topN = extractSafeTopN(data);

// 修正後
const topN = extractSafeTopN(data, targetType);
```

**Step 3: searchKeyword の呼び出しを修正**

```typescript
// 修正前（agent.ts:195 付近）
const topN = extractSafeTopN(data);

// 修正後
const topN = extractSafeTopN(data, targetType);
```

**変更不要の確認（searchPage ハンドラー）:**

`searchPage` ハンドラー（agent.ts:237）は `extractSafeTopN(data)` を `targetType` なしで呼ぶ。ページ遷移時の data には `pageData.topN`（数値）が含まれており、`extractNumber(data, "topN", -1)` で正しく取得される。`topN_faq` フィールドは pageData に含まれないため、`primaryKey="topN"` で正しく動作する。変更不要。

**Step 4: コンパイル確認**

Run: `cd maintenance-bot && npx tsc --noEmit`
Expected: エラー0件

**Step 5: コミット**

```bash
git add maintenance-bot/src/agent.ts
git commit -m "fix: extractSafeTopN が targetType を考慮してFAQタブの件数変更を正しく反映"
```

---

## Task 3: agent.ts — キャッシュミス時のカテゴリフォールバック修正

**Files:**
- Modify: `maintenance-bot/src/agent.ts:887-904`（extractCategorySelectionsFromPageData）
- Modify: `maintenance-bot/src/agent.ts:240-252`（searchPage ハンドラーのキャッシュ復元ロジック）

**問題の詳細:**

`extractCategorySelectionsFromPageData` のフォールバック（agent.ts:899-903）が全カテゴリ（シナリオ4 + FAQ3）を返す。キャッシュミス（TTL超過・50件溢れ）時に再検索すると、シナリオのみ検索したセッションでFAQも検索されてしまう。

**修正方針:**

2段階で修正:
1. `searchPage` ハンドラーで、キャッシュが存在する場合は **常に** キャッシュの categories を使用する（現在は `!query` 条件のときのみ）
2. `extractCategorySelectionsFromPageData` のフォールバックは空カテゴリにする（防御的プログラミング — キャッシュもパースも失敗した場合に全カテゴリ検索を避ける）

**Step 1: searchPage ハンドラーのキャッシュ復元を拡張**

```typescript
// 修正前（agent.ts:244-252）
// Cache fallback: 「検索結果に戻る」ボタンは searchSessionId + page のみ送信するため
// query が空の場合はキャッシュから検索パラメータを復元する
const cached = searchSessionId ? searchResultCache.get(searchSessionId) : undefined;
if (!query && cached) {
  query = cached.query;
  mode = cached.mode;
  categories = cached.categories;
  topN = cached.topN;
}

// 修正後
// Cache fallback: キャッシュが存在する場合は常にキャッシュの値を優先する
// （「検索結果に戻る」ボタンは最小限のデータのみ送信するため、
//   query/mode/categories/topN はキャッシュから復元する）
const cached = searchSessionId ? searchResultCache.get(searchSessionId) : undefined;
if (cached) {
  if (!query) query = cached.query;
  if (!mode || mode === "semantic") mode = cached.mode;  // デフォルト値でない場合のみ維持
  categories = cached.categories;
  topN = cached.topN;
}
```

**注意**: `mode` の復元条件を変更。`extractField(data, "mode", "semantic")` のデフォルト値が `"semantic"` のため、data に mode が含まれない場合は `"semantic"` が入る。キャッシュの mode が `"keyword"` の場合に上書きされてしまうのを防ぐ。

**しかしこれは問題がある**: `searchPage` ハンドラーでは `mode` は `pageData` に含まれている（cards.ts:409）。通常のページ遷移では data.mode が正しい値を持つ。キャッシュで上書きすると、pageData の mode が無視される。

**修正方針の再検討**: `categories` のみキャッシュから無条件復元し、`query`/`mode`/`topN` は現行のまま（!query 条件時のみ復元）にする。

```typescript
// 最終的な修正
const cached = searchSessionId ? searchResultCache.get(searchSessionId) : undefined;
if (cached) {
  // categories はキャッシュから常に復元（フォールバック値より正確）
  categories = cached.categories;
}
if (!query && cached) {
  query = cached.query;
  mode = cached.mode;
  topN = cached.topN;
}
```

**Step 2: extractCategorySelectionsFromPageData のフォールバックを空に変更**

```typescript
// 修正前（agent.ts:899-903）
// フォールバック: 全カテゴリ
return {
  scenarios: SCENARIO_CATEGORIES.map((c) => c.id),
  faqs: FAQ_CATEGORIES.map((c) => c.id),
};

// 修正後
// フォールバック: 空カテゴリ（キャッシュから復元されることを前提とする）
// キャッシュも存在しない場合は executeSearchPaged で「カテゴリ未選択」エラーになる
return {
  scenarios: [],
  faqs: [],
};
```

**影響範囲の確認:**

- `searchPage` ハンドラー（agent.ts:241）: `extractCategorySelectionsFromPageData` が空を返しても、直後に `cached.categories` で上書きされるため問題なし
- `executeSearchPaged` → `buildResultCard`: categories が空の場合、検索結果カードは正常に「0件」を表示。エラーにはならない
- キャッシュミスかつ `selectedCategories` パース失敗（極めて稀）: `searchByCategories` に空カテゴリが渡る → 検索スキップ → 「該当する候補が見つかりませんでした」表示。全カテゴリ検索より安全

**Step 3: コンパイル確認**

Run: `cd maintenance-bot && npx tsc --noEmit`
Expected: エラー0件

**Step 4: コミット**

```bash
git add maintenance-bot/src/agent.ts
git commit -m "fix: ページ遷移時のカテゴリ復元をキャッシュ優先に変更（キャッシュミス時の全カテゴリ検索を防止）"
```

---

## Task 4: 最終検証 + まとめコミット

**Step 1: TypeScript コンパイル最終確認**

Run: `cd maintenance-bot && npx tsc --noEmit`
Expected: エラー0件

**Step 2: git status で意図しない変更がないか確認**

Run: `git status`
Expected: `sharepoint.ts` と `agent.ts` の2ファイルのみ変更

**Step 3: 変更差分の最終レビュー**

Run: `git diff maintenance-bot/src/sharepoint.ts maintenance-bot/src/agent.ts`

確認ポイント:
- [ ] `sharepoint.ts`: `folderUrl` の導出が `response.webUrl` ベースに変更されている
- [ ] `agent.ts:extractSafeTopN`: `targetType` パラメータ追加、呼び出し元2箇所が更新されている
- [ ] `agent.ts:searchPage`: `categories = cached.categories` がキャッシュヒット時に無条件実行される
- [ ] `agent.ts:extractCategorySelectionsFromPageData`: フォールバックが空カテゴリに変更されている
- [ ] 上記以外の変更がない

---

## 設計ノート: 「検索結果に戻る」ボタンの perPage について

`buildNeedsUpdateCompleteCard`（cards.ts:750-757）の「検索結果に戻る」ボタンは `{ searchSessionId, page: 1 }` のみを送信し、`perPage` を含まない。

**修正しない理由:**

1. `searchPage` → `executeSearchPaged` → `buildResultCard` で `fixedPerPage=undefined` になるが、`buildResultCard` 内の二分探索がページ1の内容に最適な `perPage` を再計算する
2. ページ1に戻るため、計算結果は初回検索時と同等になる
3. `buildResultCard` の戻り値型を `{ card, perPage }` に変更すると3箇所の呼び出し元に波及し、不整合リスクが発生する
4. `CachedSearchResult` に `perPage` フィールドを追加してもページサイズは表示内容依存で変動するため、キャッシュ値が最適とは限らない

将来テストスイートが整備された後に、必要に応じて `CachedSearchResult` に `perPage` を追加する改修を検討する。
