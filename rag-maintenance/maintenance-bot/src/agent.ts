import { ActivityTypes } from "@microsoft/agents-activity";
import {
  AgentApplication,
  MemoryStorage,
  TurnContext,
  TurnState,
  MessageFactory,
  CardFactory,
  type AdaptiveCard,
} from "@microsoft/agents-hosting";
import { SearchClient } from "@azure/search-documents";
import { DefaultAzureCredential } from "@azure/identity";
import config, { SCENARIO_CATEGORIES, FAQ_CATEGORIES, DEFAULT_TOP_N } from "./config";
import {
  buildSearchCard,
  buildResultCard,
  buildDeleteConfirmCard,
  buildDeleteCompleteCard,
  buildNeedsUpdateCompleteCard,
  buildCancelCard,
  SearchResultItem,
  CategorySelection,
} from "./cards";
import { deleteFaqs, saveNeedsUpdate } from "./cosmos";

// --- クライアント遅延初期化 ---
let _searchClient: SearchClient<Record<string, unknown>> | null = null;
function getSearchClient() {
  if (!_searchClient) {
    _searchClient = new SearchClient(
      config.aiSearchEndpoint,
      config.aiSearchIndexName,
      new DefaultAzureCredential()
    );
  }
  return _searchClient;
}

const storage = new MemoryStorage();
export const agentApp = new AgentApplication({ storage });

// --- ウェルカムメッセージ ---
agentApp.onConversationUpdate("membersAdded", async (context: TurnContext) => {
  await context.sendActivity(
    "影響候補検出Botです。改定内容を入力すると、影響を受ける可能性のあるシナリオ・FAQを検索します。"
  );
});

// --- FR-001: テキスト入力 → 統合検索カード ---
agentApp.onActivity(ActivityTypes.Message, async (context: TurnContext) => {
  const query = context.activity.text?.trim();
  if (!query) {
    await context.sendActivity("テキストを入力してください。");
    return;
  }

  // 2,000文字制限 (FR-001)
  if (query.length > 2000) {
    await context.sendActivity(
      "入力テキストが2,000文字を超えています。短くしてから再送信してください。"
    );
    return;
  }

  const card = buildSearchCard(query);
  const activity = MessageFactory.attachment(
    CardFactory.adaptiveCard(card)
  );
  await context.sendActivity(activity);
});

// --- FR-003: 意味検索 (Action.Execute verb: searchSemantic) ---
agentApp.adaptiveCards.actionExecute(
  "searchSemantic",
  async (context: TurnContext, _state: TurnState, data: Record<string, unknown>) => {
    const query = extractQuery(data, context);
    const targetType = extractTargetType(data);
    const categories = extractCategorySelections(data, targetType);
    const topN = extractSafeTopN(data);
    console.log(`[searchSemantic] query: ${query}, targetType: ${targetType}, categories: ${JSON.stringify(categories)}, topN: ${topN}`);
    return await executeSearch(query, "semantic", categories, topN);
  }
);

// --- FR-004: キーワード検索 (Action.Execute verb: searchKeyword) ---
agentApp.adaptiveCards.actionExecute(
  "searchKeyword",
  async (context: TurnContext, _state: TurnState, data: Record<string, unknown>) => {
    const query = extractQuery(data, context);
    const targetType = extractTargetType(data);
    const categories = extractCategorySelections(data, targetType);
    const topN = extractSafeTopN(data);
    console.log(`[searchKeyword] query: ${query}, targetType: ${targetType}, categories: ${JSON.stringify(categories)}, topN: ${topN}`);
    return await executeSearch(query, "keyword", categories, topN);
  }
);

// --- ページ遷移 (Action.Execute verb: searchPage) ---
agentApp.adaptiveCards.actionExecute(
  "searchPage",
  async (context: TurnContext, _state: TurnState, data: Record<string, unknown>) => {
    const query = extractQuery(data, context);
    const mode = extractField(data, "mode", "semantic") === "keyword" ? "keyword" : "semantic";
    const page = extractNumber(data, "page", 1);
    const topN = extractSafeTopN(data);
    const perPage = extractNumber(data, "perPage", -1);

    // ページ遷移時は data に埋め込み済みの selectedCategories を使用
    const categories = extractCategorySelectionsFromPageData(data);
    console.log(`[searchPage] query: ${query}, mode: ${mode}, page: ${page}, topN: ${topN}, perPage: ${perPage}`);
    return await executeSearchPaged(query, mode, page, categories, topN, perPage > 0 ? perPage : undefined);
  }
);

// --- FR-013: FAQ削除確認 (Action.Execute verb: confirmDeleteFaqs) ---
agentApp.adaptiveCards.actionExecute(
  "confirmDeleteFaqs",
  async (_context: TurnContext, _state: TurnState, data: Record<string, string>) => {
    const selectedIds = extractSelectedIds(data, "faq_");
    if (selectedIds.length === 0) {
      return { type: "AdaptiveCard", body: [{ type: "TextBlock", text: "削除対象が選択されていません。" }], version: "1.5", $schema: "http://adaptivecards.io/schemas/adaptive-card.json" } as AdaptiveCard;
    }

    const faqInfos = selectedIds.map((id) => ({
      id,
      title: id,
    }));
    return buildDeleteConfirmCard(faqInfos);
  }
);

// --- FR-013: FAQ削除実行 (Action.Execute verb: executeDeleteFaqs) ---
agentApp.adaptiveCards.actionExecute(
  "executeDeleteFaqs",
  async (context: TurnContext, _state: TurnState, data: { faqIds: string[] }) => {
    const user = context.activity.from?.name ?? "不明";
    const deleted = await deleteFaqs(data.faqIds);
    return buildDeleteCompleteCard(deleted, user);
  }
);

// --- FR-014: シナリオ要修正フラグ保存 (Action.Execute verb: saveNeedsUpdate) ---
agentApp.adaptiveCards.actionExecute(
  "saveNeedsUpdate",
  async (context: TurnContext, _state: TurnState, data: Record<string, string>) => {
    const selectedIds = extractSelectedIds(data, "scenario_");
    if (selectedIds.length === 0) {
      return { type: "AdaptiveCard", body: [{ type: "TextBlock", text: "要修正の対象が選択されていません。" }], version: "1.5", $schema: "http://adaptivecards.io/schemas/adaptive-card.json" } as AdaptiveCard;
    }

    const user = context.activity.from?.name ?? "不明";
    const query = (data as Record<string, string>).query ?? "";
    const saved = await saveNeedsUpdate(selectedIds, query, user);
    return buildNeedsUpdateCompleteCard(saved, user);
  }
);

// --- キャンセル ---
agentApp.adaptiveCards.actionExecute(
  "cancelAction",
  async () => {
    return buildCancelCard();
  }
);

// --- 検索結果型 ---
interface SearchResults {
  scenarios: SearchResultItem[];
  faqs: SearchResultItem[];
}

// --- 検索実行ロジック ---
async function executeSearch(
  query: string,
  mode: "semantic" | "keyword",
  categories: CategorySelection,
  topN: number
): Promise<AdaptiveCard> {
  if (!query) {
    return {
      type: "AdaptiveCard",
      $schema: "http://adaptivecards.io/schemas/adaptive-card.json",
      version: "1.5",
      body: [{ type: "TextBlock", text: "検索クエリが取得できませんでした。もう一度テキストを入力してください。", wrap: true, color: "Attention" }],
    } as AdaptiveCard;
  }

  if (categories.scenarios.length === 0 && categories.faqs.length === 0) {
    return {
      type: "AdaptiveCard",
      $schema: "http://adaptivecards.io/schemas/adaptive-card.json",
      version: "1.5",
      body: [{ type: "TextBlock", text: "カテゴリを1つ以上選択してください。", wrap: true, color: "Attention" }],
    } as AdaptiveCard;
  }

  try {
    const results = await searchByCategories(query, mode, categories, topN);

    if (results.scenarios.length === 0 && results.faqs.length === 0) {
      return {
        type: "AdaptiveCard",
        $schema: "http://adaptivecards.io/schemas/adaptive-card.json",
        version: "1.5",
        body: [{ type: "TextBlock", text: "該当する候補が見つかりませんでした。", wrap: true }],
      } as AdaptiveCard;
    }

    return buildResultCard(query, mode, results.scenarios, results.faqs, 1, categories, topN);
  } catch (err: unknown) {
    return buildSearchErrorCard(err);
  }
}

// --- ページ遷移用の検索実行（再検索 + 指定ページ表示） ---
async function executeSearchPaged(
  query: string,
  mode: "semantic" | "keyword",
  page: number,
  categories: CategorySelection,
  topN: number,
  perPage?: number
): Promise<AdaptiveCard> {
  if (!query) {
    return {
      type: "AdaptiveCard",
      $schema: "http://adaptivecards.io/schemas/adaptive-card.json",
      version: "1.5",
      body: [{ type: "TextBlock", text: "検索クエリが取得できませんでした。", wrap: true, color: "Attention" }],
    } as AdaptiveCard;
  }
  try {
    const results = await searchByCategories(query, mode, categories, topN);

    if (results.scenarios.length === 0 && results.faqs.length === 0) {
      return {
        type: "AdaptiveCard",
        $schema: "http://adaptivecards.io/schemas/adaptive-card.json",
        version: "1.5",
        body: [{ type: "TextBlock", text: "該当する候補が見つかりませんでした。", wrap: true }],
      } as AdaptiveCard;
    }

    return buildResultCard(query, mode, results.scenarios, results.faqs, page, categories, topN, perPage);
  } catch (err: unknown) {
    return buildSearchErrorCard(err);
  }
}

function buildSearchErrorCard(err: unknown): AdaptiveCard {
  console.error("[executeSearch] Error:", err);
  let message: string;
  if (err instanceof Error) {
    message = err.message;
    // Azure SDK固有のエラープロパティを安全に取得
    if ("statusCode" in err) message += ` | Status: ${(err as { statusCode: unknown }).statusCode}`;
    if ("code" in err) message += ` | Code: ${(err as { code: unknown }).code}`;
  } else if (typeof err === "object" && err !== null) {
    message = JSON.stringify(err);
  } else {
    message = String(err);
  }
  return {
    type: "AdaptiveCard",
    $schema: "http://adaptivecards.io/schemas/adaptive-card.json",
    version: "1.5",
    body: [{ type: "TextBlock", text: `検索エラー: ${message || "(詳細はデバッグコンソール参照)"}`, color: "Attention", wrap: true }],
  } as AdaptiveCard;
}

// --- ホワイトリスト ---
const VALID_SCENARIO_IDS: Set<string> = new Set(SCENARIO_CATEGORIES.map((c) => c.id));
const VALID_FAQ_IDS: Set<string> = new Set(FAQ_CATEGORIES.map((c) => c.id));

function deduplicateById(items: SearchResultItem[]): SearchResultItem[] {
  const seen = new Set<string>();
  return items.filter((item) => {
    if (seen.has(item.id)) return false;
    seen.add(item.id);
    return true;
  });
}

// --- カテゴリ別並列検索 ---
async function searchByCategories(
  query: string,
  mode: "semantic" | "keyword",
  categories: CategorySelection,
  topN: number
): Promise<SearchResults> {
  // ホワイトリスト検証済みのカテゴリIDのみ使用
  const validScenarioIds = categories.scenarios.filter((catId) => VALID_SCENARIO_IDS.has(catId));
  const validFaqIds = categories.faqs.filter((catId) => VALID_FAQ_IDS.has(catId));

  // シナリオ: 選択カテゴリ分を並列検索
  const scenarioPromises = validScenarioIds.map((catId) =>
    searchSingle(query, mode, "scenario", catId, topN)
  );
  // FAQ: 選択カテゴリ分を並列検索
  const faqPromises = validFaqIds.map((catId) =>
    searchSingle(query, mode, "faq", catId, topN)
  );

  const [scenarioResults, faqResults] = await Promise.all([
    Promise.allSettled(scenarioPromises),
    Promise.allSettled(faqPromises),
  ]);

  // 成功した結果のみ取得、失敗はログ出力
  const scenarios = scenarioResults
    .filter((r): r is PromiseFulfilledResult<SearchResultItem[]> => r.status === "fulfilled")
    .flatMap((r) => r.value);
  const faqs = faqResults
    .filter((r): r is PromiseFulfilledResult<SearchResultItem[]> => r.status === "fulfilled")
    .flatMap((r) => r.value);

  scenarioResults.forEach((r, i) => {
    if (r.status === "rejected") {
      console.error(`[searchByCategories] scenario/${validScenarioIds[i]} failed:`, r.reason);
    }
  });
  faqResults.forEach((r, i) => {
    if (r.status === "rejected") {
      console.error(`[searchByCategories] faq/${validFaqIds[i]} failed:`, r.reason);
    }
  });

  return {
    scenarios: deduplicateById(scenarios).sort((a, b) => b.score - a.score),
    faqs: deduplicateById(faqs).sort((a, b) => b.score - a.score),
  };
}

// --- 単一カテゴリ検索 ---
async function searchSingle(
  query: string,
  mode: "semantic" | "keyword",
  dataType: "scenario" | "faq",
  categoryId: string,
  topN: number
): Promise<SearchResultItem[]> {
  // categoryId は searchByCategories でホワイトリスト検証済み
  const filter = `isDeleted eq false and dataType eq '${dataType}' and categoryId eq '${categoryId}'`;

  const options = mode === "semantic"
    ? {
        queryType: "simple" as const,
        vectorSearchOptions: {
          queries: [
            {
              kind: "text" as const,
              text: query,
              fields: ["contentVector"],
            },
          ],
        },
        select: ["id", "dataType", "categoryName", "title", "content"] as string[],
        top: topN,
        filter,
      }
    : {
        queryType: "full" as const,
        searchFields: ["title", "content", "keywords"] as string[],
        select: ["id", "dataType", "categoryName", "title", "content"] as string[],
        top: topN,
        filter,
      };

  const searchResults = await getSearchClient().search(query, options);
  const items: SearchResultItem[] = [];
  for await (const result of searchResults.results) {
    const doc = result.document as Record<string, unknown>;
    items.push({
      id: String(doc.id),
      dataType,
      categoryName: String(doc.categoryName),
      title: String(doc.title),
      content: String(doc.content),
      score: result.score ?? 0,
    });
  }

  console.log(`[searchSingle] ${dataType}/${categoryId}: ${items.length} results`);
  return items;
}

// --- ユーティリティ ---

type SearchTargetType = "scenario" | "faq";

/** Action.Execute の data から query を取得（SDKのデータ構造差異を吸収） */
function extractQuery(data: Record<string, unknown>, context: TurnContext): string {
  // 1. data.query（直接渡し）
  if (typeof data?.query === "string" && data.query) return data.query;
  // 2. data.data.query（M365 Agents SDK: dataがAction全体の場合）
  const nested = data?.data as Record<string, unknown> | undefined;
  if (typeof nested?.query === "string" && nested.query) return nested.query;
  // 3. activity.value.action.data.query（Bot Framework標準構造）
  const val = context.activity.value as Record<string, unknown> | undefined;
  const actionData = (val?.action as Record<string, unknown>)?.data as Record<string, unknown> | undefined;
  if (typeof actionData?.query === "string" && actionData.query) return actionData.query;
  // 4. activity.value.data.query
  const valData = val?.data as Record<string, unknown> | undefined;
  if (typeof valData?.query === "string" && valData.query) return valData.query;
  // 5. activity.value.query
  if (typeof val?.query === "string" && val.query) return val.query;
  return "";
}

/** Action.Execute の data からフィールドを取得（data.X ?? data.data.X フォールバック） */
function extractField(data: Record<string, unknown>, field: string, defaultValue: string): string {
  if (typeof data?.[field] === "string" && data[field]) return data[field] as string;
  const nested = data?.data as Record<string, unknown> | undefined;
  if (typeof nested?.[field] === "string" && nested[field]) return nested[field] as string;
  return defaultValue;
}

/** Action.Execute の data から数値フィールドを取得 */
function extractNumber(data: Record<string, unknown>, field: string, defaultValue: number): number {
  if (data?.[field] !== undefined && data[field] !== null) {
    const v = Number(data[field]);
    if (!isNaN(v)) return v;
  }
  const nested = data?.data as Record<string, unknown> | undefined;
  if (nested?.[field] !== undefined && nested[field] !== null) {
    const v = Number(nested[field]);
    if (!isNaN(v)) return v;
  }
  return defaultValue;
}

/** topN をバリデーション付きで取得（10〜100に制限） */
function extractSafeTopN(data: Record<string, unknown>): number {
  // シナリオタブは "topN"、FAQタブは "topN_faq" のIDを使用
  let raw = extractNumber(data, "topN", -1);
  if (raw < 0) raw = extractNumber(data, "topN_faq", DEFAULT_TOP_N);
  if (raw < 0) raw = DEFAULT_TOP_N;
  return Math.min(Math.max(raw, 10), 100);
}

/** data から targetType を取得（"scenario" | "faq"、デフォルト "scenario"） */
function extractTargetType(data: Record<string, unknown>): SearchTargetType {
  const raw = extractField(data, "targetType", "scenario");
  return raw === "faq" ? "faq" : "scenario";
}

/** Input.Toggle の値を boolean として判定（SDK差異を吸収） */
function isToggleOn(value: unknown): boolean {
  return value === "true" || value === true;
}

/** 初回検索時: チェックボックスから CategorySelection を抽出（scat_ / fcat_ プレフィックス） */
function extractCategorySelections(
  data: Record<string, unknown>,
  targetType: SearchTargetType
): CategorySelection {
  const nested = data?.data as Record<string, unknown> | undefined;
  const prefix = targetType === "scenario" ? "scat_" : "fcat_";
  const categories = targetType === "scenario" ? SCENARIO_CATEGORIES : FAQ_CATEGORIES;

  const selectedIds = categories
    .filter((c) => {
      const key = `${prefix}${c.id}`;
      return isToggleOn(data[key]) || isToggleOn(nested?.[key]);
    })
    .map((c) => c.id);

  return targetType === "scenario"
    ? { scenarios: selectedIds, faqs: [] }
    : { scenarios: [], faqs: selectedIds };
}

/** ページ遷移時: data.selectedCategories から CategorySelection を取得 */
function extractCategorySelectionsFromPageData(data: Record<string, unknown>): CategorySelection {
  // data.selectedCategories ?? data.data.selectedCategories フォールバック
  const sel = data?.selectedCategories as CategorySelection | undefined;
  if (sel && Array.isArray(sel.scenarios) && Array.isArray(sel.faqs)) {
    return sel;
  }
  const nested = data?.data as Record<string, unknown> | undefined;
  const nestedSel = nested?.selectedCategories as CategorySelection | undefined;
  if (nestedSel && Array.isArray(nestedSel.scenarios) && Array.isArray(nestedSel.faqs)) {
    return nestedSel;
  }
  // フォールバック: 全カテゴリ
  return {
    scenarios: SCENARIO_CATEGORIES.map((c) => c.id),
    faqs: FAQ_CATEGORIES.map((c) => c.id),
  };
}

function extractSelectedIds(
  data: Record<string, string>,
  prefix: string
): string[] {
  const ids: string[] = [];
  for (const [key, value] of Object.entries(data)) {
    if (key.startsWith(prefix) && isToggleOn(value)) {
      ids.push(key.replace(prefix, ""));
    }
  }
  return ids;
}
