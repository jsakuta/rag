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
import config from "./config";
import {
  buildModeSelectCard,
  buildResultCard,
  buildDeleteConfirmCard,
  buildDeleteCompleteCard,
  buildNeedsUpdateCompleteCard,
  buildCancelCard,
  SearchResultItem,
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

// --- FR-001: テキスト入力 → 検索モード選択カード ---
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

  const card = buildModeSelectCard(query);
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
    console.log("[searchSemantic] data:", JSON.stringify(data));
    console.log("[searchSemantic] query:", query);
    return await executeSearch(query, "semantic");
  }
);

// --- FR-004: キーワード一致検索 (Action.Execute verb: searchKeyword) ---
agentApp.adaptiveCards.actionExecute(
  "searchKeyword",
  async (context: TurnContext, _state: TurnState, data: Record<string, unknown>) => {
    const query = extractQuery(data, context);
    console.log("[searchKeyword] data:", JSON.stringify(data));
    console.log("[searchKeyword] query:", query);
    return await executeSearch(query, "keyword");
  }
);

// --- ページ遷移 (Action.Execute verb: searchPage) ---
agentApp.adaptiveCards.actionExecute(
  "searchPage",
  async (context: TurnContext, _state: TurnState, data: Record<string, unknown>) => {
    const query = extractQuery(data, context);
    const mode = (data.mode as string) === "keyword" ? "keyword" : "semantic";
    const page = Number(data.page) || 1;
    console.log(`[searchPage] query: ${query}, mode: ${mode}, page: ${page}`);
    return await executeSearchPaged(query, mode, page);
  }
);

// --- FR-013: FAQ削除確認 (Action.Execute verb: confirmDeleteFaqs) ---
agentApp.adaptiveCards.actionExecute(
  "confirmDeleteFaqs",
  async (_context: TurnContext, _state: TurnState, data: Record<string, string>) => {
    // Input.Toggle の値は "faq_{id}" = "true"/"false" で送られてくる
    const selectedIds = extractSelectedIds(data, "faq_");
    if (selectedIds.length === 0) {
      return { type: "AdaptiveCard", body: [{ type: "TextBlock", text: "削除対象が選択されていません。" }], version: "1.5", $schema: "http://adaptivecards.io/schemas/adaptive-card.json" } as AdaptiveCard;
    }

    // FAQ情報を取得して確認カードを生成
    const faqInfos = selectedIds.map((id) => ({
      id,
      title: id, // 簡易版: IDのみ表示
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

// --- 検索実行ロジック ---
async function executeSearch(
  query: string,
  mode: "semantic" | "keyword"
): Promise<AdaptiveCard> {
  if (!query) {
    return {
      type: "AdaptiveCard",
      $schema: "http://adaptivecards.io/schemas/adaptive-card.json",
      version: "1.5",
      body: [{ type: "TextBlock", text: "検索クエリが取得できませんでした。もう一度テキストを入力してください。", wrap: true, color: "Attention" }],
    } as AdaptiveCard;
  }
  try {
    let items: SearchResultItem[];

    if (mode === "semantic") {
      items = await searchSemantic(query);
    } else {
      items = await searchKeyword(query);
    }

    if (items.length === 0) {
      return {
        type: "AdaptiveCard",
        $schema: "http://adaptivecards.io/schemas/adaptive-card.json",
        version: "1.5",
        body: [{ type: "TextBlock", text: "該当する候補が見つかりませんでした。", wrap: true }],
      } as AdaptiveCard;
    }

    return buildResultCard(query, mode, items);
  } catch (err: unknown) {
    return buildSearchErrorCard(err);
  }
}

// --- ページ遷移用の検索実行（再検索 + 指定ページ表示） ---
async function executeSearchPaged(
  query: string,
  mode: "semantic" | "keyword",
  page: number
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
    const items = mode === "semantic"
      ? await searchSemantic(query)
      : await searchKeyword(query);

    if (items.length === 0) {
      return {
        type: "AdaptiveCard",
        $schema: "http://adaptivecards.io/schemas/adaptive-card.json",
        version: "1.5",
        body: [{ type: "TextBlock", text: "該当する候補が見つかりませんでした。", wrap: true }],
      } as AdaptiveCard;
    }

    return buildResultCard(query, mode, items, page);
  } catch (err: unknown) {
    return buildSearchErrorCard(err);
  }
}

function buildSearchErrorCard(err: unknown): AdaptiveCard {
  console.error("[executeSearch] Error:", err);
  let message: string;
  if (err instanceof Error) {
    const e = err as unknown as Record<string, unknown>;
    message = [err.message, e.statusCode, e.code, e.details]
      .filter(Boolean)
      .join(" | ");
  } else {
    message = JSON.stringify(err);
  }
  return {
    type: "AdaptiveCard",
    $schema: "http://adaptivecards.io/schemas/adaptive-card.json",
    version: "1.5",
    body: [{ type: "TextBlock", text: `検索エラー: ${message || "(詳細はデバッグコンソール参照)"}`, color: "Attention", wrap: true }],
  } as AdaptiveCard;
}

// --- FR-003: 意味検索（ハイブリッド + Semantic Ranker） ---
async function searchSemantic(query: string): Promise<SearchResultItem[]> {
  const results = await getSearchClient().search(query, {
    queryType: "semantic",
    semanticSearchOptions: {
      configurationName: "semantic-config",
    },
    vectorSearchOptions: {
      queries: [
        {
          kind: "text",
          text: query,
          fields: ["contentVector"],
        },
      ],
    },
    select: ["id", "dataType", "categoryName", "title", "content"],
    top: 20,
    filter: "isDeleted eq false",
  });

  const items: SearchResultItem[] = [];
  for await (const result of results.results) {
    const doc = result.document as Record<string, unknown>;
    const score = result.rerankerScore ?? result.score ?? 0;

    // rerankerScore >= 1.5 でフィルタ (FR-003)
    if (result.rerankerScore !== undefined && result.rerankerScore < 1.5) {
      continue;
    }

    items.push({
      id: String(doc.id),
      dataType: doc.dataType as "scenario" | "faq",
      categoryName: String(doc.categoryName),
      title: String(doc.title),
      content: String(doc.content),
      score,
    });
  }

  return items;
}

// --- FR-004: キーワード一致検索 ---
async function searchKeyword(query: string): Promise<SearchResultItem[]> {
  // search.ismatchscoring で title, content を全文検索
  const results = await getSearchClient().search(query, {
    queryType: "full",
    searchFields: ["title", "content"],
    select: ["id", "dataType", "categoryName", "title", "content"],
    top: 50,
    filter: "isDeleted eq false",
  });

  const items: SearchResultItem[] = [];
  for await (const result of results.results) {
    const doc = result.document as Record<string, unknown>;
    items.push({
      id: String(doc.id),
      dataType: doc.dataType as "scenario" | "faq",
      categoryName: String(doc.categoryName),
      title: String(doc.title),
      content: String(doc.content),
      score: result.score ?? 0,
    });
  }

  return items;
}

// --- ユーティリティ ---

/** Action.Execute の data から query を取得（SDKのデータ構造差異を吸収） */
function extractQuery(data: Record<string, unknown>, context: TurnContext): string {
  // 1. data.query（直接渡し）
  if (typeof data?.query === "string" && data.query) return data.query;
  // 2. activity.value.action.data.query（Bot Framework標準構造）
  const val = context.activity.value as Record<string, unknown> | undefined;
  const actionData = (val?.action as Record<string, unknown>)?.data as Record<string, unknown> | undefined;
  if (typeof actionData?.query === "string" && actionData.query) return actionData.query;
  // 3. activity.value.data.query
  const valData = val?.data as Record<string, unknown> | undefined;
  if (typeof valData?.query === "string" && valData.query) return valData.query;
  // 4. activity.value.query
  if (typeof val?.query === "string" && val.query) return val.query;
  return "";
}

function extractSelectedIds(
  data: Record<string, string>,
  prefix: string
): string[] {
  const ids: string[] = [];
  for (const [key, value] of Object.entries(data)) {
    if (key.startsWith(prefix) && value === "true") {
      ids.push(key.replace(prefix, ""));
    }
  }
  return ids;
}
