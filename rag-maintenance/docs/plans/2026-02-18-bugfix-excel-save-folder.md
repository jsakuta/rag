# Excel全量出力・要修正保存・フォルダリンク修正 実装計画

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 3つのバグを修正する — Excel出力をカテゴリ全量に変更、要修正保存でキャッシュを統合、フォルダパスリンクを表示

**Architecture:** Excel出力時にAI Searchからカテゴリ全シナリオを取得する関数を追加。saveNeedsUpdateでキャッシュのneedsUpdateIdsを統合。SPOフォルダURLをファイルURLから構築するフォールバックを追加。

**Tech Stack:** TypeScript, @azure/search-documents, ExcelJS, Microsoft Graph API

---

## 前提知識

### データ規模（カテゴリ別シナリオ件数）
| カテゴリID | 名称 | 件数 |
|-----------|------|------|
| smile | スマイル | 555 |
| souzoku | 相続 | 269 |
| naibujimu | 内部事務 | 1,384 |
| torikaku | 取引時確認 | 105 |

AI Search の `top` パラメータ上限は 1,000。`naibujimu` は 1,384 件あるため、ページネーション（`skip` + `top`）が必要。

### テスト方針
プロジェクトにテスト基盤なし（PoC）。F5デバッグ + Teams上の手動E2Eで検証する。各タスク完了後にビルド確認（`tsc --noEmit`）を実施。

### 関連ファイル
- `maintenance-bot/src/agent.ts` — ハンドラ群（saveNeedsUpdate, exportExcel）
- `maintenance-bot/src/cards.ts` — Adaptive Card構築（buildNeedsUpdateCompleteCard, buildExcelExportCompleteCard）
- `maintenance-bot/src/excel.ts` — Excel生成（generateCategoryExcels）
- `maintenance-bot/src/sharepoint.ts` — SPOアップロード（uploadExcelToSharePoint）
- `maintenance-bot/src/config.ts` — カテゴリ定義

---

## Task 1: Bug 2修正 — saveNeedsUpdateでキャッシュのneedsUpdateIdsを統合

**Files:**
- Modify: `maintenance-bot/src/agent.ts:199-226`（saveNeedsUpdateハンドラ）

### 問題
`extractSelectedIds(data, "scenario_")` は現在カードのトグルのみ取得。
ページ遷移時に `syncToggleState` でキャッシュに蓄積された他ページのIDが無視される。

### Step 1: saveNeedsUpdateハンドラを修正

現在のコード（agent.ts:199-226）:
```typescript
agentApp.adaptiveCards.actionExecute(
  "saveNeedsUpdate",
  async (context: TurnContext, _state: TurnState, data: Record<string, unknown>) => {
    const selectedIds = extractSelectedIds(data, "scenario_");
    // ... selectedIdsのみ使用
  }
);
```

修正後:
```typescript
agentApp.adaptiveCards.actionExecute(
  "saveNeedsUpdate",
  async (context: TurnContext, _state: TurnState, data: Record<string, unknown>) => {
    const searchSessionId = extractField(data as Record<string, unknown>, "searchSessionId", "");
    cleanExpiredCache();
    const cached = searchSessionId ? searchResultCache.get(searchSessionId) : undefined;

    // 1. 現在カードのトグル状態をキャッシュに同期（ページ遷移と同じロジック）
    if (cached) {
      syncToggleState(data, cached.needsUpdateIds, "scenario_");
    }

    // 2. キャッシュの全needsUpdateIds + 現在カードのIDを統合
    const currentPageIds = extractSelectedIds(data, "scenario_");
    const allIds = new Set<string>(currentPageIds);
    if (cached) {
      for (const id of cached.needsUpdateIds) {
        allIds.add(id);
      }
    }
    const selectedIds = Array.from(allIds);

    console.log(`[saveNeedsUpdate] selectedIds: ${JSON.stringify(selectedIds)} (current page: ${currentPageIds.length}, cached: ${cached?.needsUpdateIds.size ?? 0})`);
    if (selectedIds.length === 0) {
      return { type: "AdaptiveCard", body: [{ type: "TextBlock", text: "要修正の対象が選択されていません。" }], version: "1.5", $schema: "http://adaptivecards.io/schemas/adaptive-card.json" } as AdaptiveCard;
    }

    const user = context.activity.from?.name ?? "不明";
    const query = extractQuery(data as Record<string, unknown>, context);

    const saved = await saveNeedsUpdate(selectedIds, query, user);

    // キャッシュの needsUpdateIds にマージ（重複なし）
    if (searchSessionId && searchResultCache.has(searchSessionId)) {
      const c = searchResultCache.get(searchSessionId)!;
      for (const id of selectedIds) {
        c.needsUpdateIds.add(id);
      }
      c.timestamp = Date.now();
    }

    return buildNeedsUpdateCompleteCard(saved, user, searchSessionId || undefined);
  }
);
```

ポイント:
- `syncToggleState` を最初に呼び、現在カードのトグル状態をキャッシュに反映
- `extractSelectedIds`（現在カード）+ `cached.needsUpdateIds`（過去ページ）を `Set` で統合
- `searchSessionId` の抽出を先頭に移動（キャッシュ参照に必要）

### Step 2: ビルド確認

```bash
cd maintenance-bot && npx tsc --noEmit
```
Expected: エラーなし

### Step 3: コミット

```bash
git add maintenance-bot/src/agent.ts
git commit -m "fix(FR-014): saveNeedsUpdateでキャッシュのneedsUpdateIdsを統合

ページ遷移で蓄積されたチェック状態と現在カードのトグルを
Setで統合し、全ページの選択を保存するように修正"
```

---

## Task 2: Bug 3修正 — Excel出力をカテゴリ全量に変更

**Files:**
- Modify: `maintenance-bot/src/agent.ts`（exportExcelハンドラ + 新関数追加）

### 問題
`generateCategoryExcels(cached.scenarios, ...)` は検索結果（最大30件）のみ使用。
ユーザーの意図: カテゴリの全シナリオをExcelに出力し、要修正行のみ黄色ハイライト。

### Step 1: カテゴリ全シナリオ取得関数を追加

`agent.ts` の `searchSingle` 関数の後に追加（590行付近）:

```typescript
/**
 * 指定カテゴリの全シナリオを AI Search から取得する（Excel全量出力用）。
 * AI Search の top 上限は 1,000 のため、skip でページネーションする。
 */
async function fetchAllScenariosForCategory(categoryId: string): Promise<SearchResultItem[]> {
  const PAGE_SIZE = 1000;
  const filter = `isDeleted eq false and dataType eq 'scenario' and categoryId eq '${categoryId}'`;
  const allItems: SearchResultItem[] = [];
  let skip = 0;

  while (true) {
    const searchResults = await getSearchClient().search("*", {
      queryType: "simple" as const,
      select: ["id", "dataType", "categoryId", "categoryName", "title", "content", "order"] as string[],
      top: PAGE_SIZE,
      skip,
      filter,
      orderBy: ["order asc"],
    });

    let count = 0;
    for await (const result of searchResults.results) {
      const doc = result.document as Record<string, unknown>;
      allItems.push({
        id: String(doc.id),
        dataType: "scenario",
        categoryId: String(doc.categoryId),
        categoryName: String(doc.categoryName),
        title: String(doc.title),
        content: String(doc.content),
        score: 0,
        order: typeof doc.order === "number" ? doc.order : undefined,
      });
      count++;
    }

    if (count < PAGE_SIZE) break;
    skip += PAGE_SIZE;
  }

  console.log(`[fetchAllScenariosForCategory] ${categoryId}: ${allItems.length} total scenarios`);
  return allItems;
}
```

### Step 2: exportExcelハンドラを修正

現在のコード（agent.ts:229-310）の `generateCategoryExcels` 呼び出し部分を修正:

```typescript
agentApp.adaptiveCards.actionExecute(
  "exportExcel",
  async (_context: TurnContext, _state: TurnState, data: Record<string, unknown>) => {
    const searchSessionId = extractField(data, "searchSessionId", "");
    cleanExpiredCache();

    const cached = searchSessionId ? searchResultCache.get(searchSessionId) : undefined;
    if (!cached) {
      console.warn(`[exportExcel] Cache miss for session: ${searchSessionId}`);
      return buildExcelExportErrorCard({ message: "検索結果の有効期限が切れました。再度検索してください。" });
    }

    try {
      // カテゴリ全シナリオを取得（検索結果ではなくカテゴリ全量）
      const scenarioCategoryIds = cached.categories.scenarios.filter(
        (catId) => VALID_SCENARIO_IDS.has(catId)
      );

      if (scenarioCategoryIds.length === 0) {
        return buildExcelExportErrorCard({
          message: "出力対象のシナリオカテゴリがありません。",
          searchSessionId: searchSessionId || undefined,
        });
      }

      // 並列で全カテゴリのシナリオを取得
      const categoryResults = await Promise.all(
        scenarioCategoryIds.map((catId) => fetchAllScenariosForCategory(catId))
      );
      const allScenarios = categoryResults.flat();

      console.log(`[exportExcel] Total scenarios for Excel: ${allScenarios.length}, needsUpdateIds: ${cached.needsUpdateIds.size}`);

      // カテゴリ別Excel生成（全シナリオ、要修正IDでハイライト）
      const categoryExcels = await generateCategoryExcels(allScenarios, cached.needsUpdateIds);

      // ... 以降の SPOアップロード処理は変更なし（既存コードそのまま）
```

注意:
- `cached.categories.scenarios` から検索対象カテゴリIDを取得
- `fetchAllScenariosForCategory` でカテゴリ全量を取得
- `cached.needsUpdateIds` で要修正行を判定（検索結果ではなくIDベース）
- SPOアップロード以降のコードは変更不要

### Step 3: ビルド確認

```bash
cd maintenance-bot && npx tsc --noEmit
```
Expected: エラーなし

### Step 4: コミット

```bash
git add maintenance-bot/src/agent.ts
git commit -m "feat(FR-015): Excel出力をカテゴリ全量シナリオに変更

検索結果（最大30件）ではなく、カテゴリの全シナリオを
AI Searchから取得してExcelに出力するように変更。
要修正フラグ付きシナリオのみ黄色ハイライト。
naibujimu（1,384件）対応のためskipページネーション実装。"
```

---

## Task 3: Bug 1修正 — フォルダパスリンクのフォールバック

**Files:**
- Modify: `maintenance-bot/src/sharepoint.ts:52-56`

### 問題
Graph API の `response.parentReference?.webUrl` が空文字を返す場合がある。
空文字は falsy なので「フォルダを開く」ボタンが非表示になる。

### Step 1: フォルダURLのフォールバックを追加

`sharepoint.ts` の return 部分を修正:

```typescript
  // フォルダURLをファイルURLから構築（parentReference.webUrl のフォールバック）
  let folderUrl = response.parentReference?.webUrl ?? "";
  if (!folderUrl && response.webUrl) {
    // ファイルURL例: https://site.sharepoint.com/.../folder/file.xlsx
    // → 末尾のファイル名を除去してフォルダURLを構築
    const lastSlash = response.webUrl.lastIndexOf("/");
    if (lastSlash > 0) {
      folderUrl = response.webUrl.substring(0, lastSlash);
    }
  }

  return {
    webUrl: response.webUrl,
    filename,
    folderUrl,
  };
```

### Step 2: ビルド確認

```bash
cd maintenance-bot && npx tsc --noEmit
```
Expected: エラーなし

### Step 3: コミット

```bash
git add maintenance-bot/src/sharepoint.ts
git commit -m "fix(FR-015): SPOフォルダURLのフォールバックを追加

parentReference.webUrlが空の場合、ファイルURLから
フォルダURLを構築するフォールバックを追加"
```

---

## Task 4: E2E動作確認

### Step 1: F5デバッグ起動

Teams Toolkit で F5 デバッグを起動。

### Step 2: 検索テスト

1. 取引時確認カテゴリを選択して意味検索を実行
2. 検索結果が表示されることを確認（`$select` エラーが解消されていること）

### Step 3: 要修正保存テスト（Bug 2検証）

1. 検索結果ページ1で数件チェック → 次のページへ遷移
2. ページ2で数件チェック → 「要修正を保存」をクリック
3. サマリーに**全ページの選択件数**が表示されることを確認
4. コンソールログ `[saveNeedsUpdate] selectedIds: [...] (current page: X, cached: Y)` で統合を確認

### Step 4: Excel出力テスト（Bug 3検証）

1. 要修正保存後の画面で「Excelで出力」をクリック
2. 出力されたExcelを開き、**カテゴリの全シナリオ**が含まれることを確認
3. 取引時確認の場合: 105件すべてが出力されていること
4. 要修正行のみ黄色ハイライトされていること
5. orderでソートされていること（元のExcel行順）

### Step 5: フォルダリンクテスト（Bug 1検証）

1. Excel出力完了カードに「フォルダを開く」ボタンが表示されることを確認
2. クリックしてSharePointフォルダが開くことを確認

---

## 実装順序と依存関係

```
Task 1 (Bug 2: saveNeedsUpdate統合) ← 独立、最小変更
    ↓
Task 2 (Bug 3: Excel全量出力)       ← Task 1完了後が望ましい（needsUpdateIds統合済み前提）
    ↓
Task 3 (Bug 1: フォルダURL)          ← 独立、最小変更
    ↓
Task 4 (E2E検証)                     ← 全修正後
```
