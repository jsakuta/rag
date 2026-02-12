import { ActivityTypes } from "@microsoft/agents-activity";
import {
  AgentApplication,
  MemoryStorage,
  TurnContext,
} from "@microsoft/agents-hosting";
import { SearchClient } from "@azure/search-documents";
import { DefaultAzureCredential } from "@azure/identity";
import config from "./config";

// AI Search クライアント（Managed Identity 認証）
const searchClient = new SearchClient(
  config.aiSearchEndpoint,
  config.aiSearchIndexName,
  new DefaultAzureCredential()
);

const storage = new MemoryStorage();
export const agentApp = new AgentApplication({ storage });

// 会話開始時のウェルカムメッセージ
agentApp.onConversationUpdate("membersAdded", async (context: TurnContext) => {
  await context.sendActivity(
    "影響候補検出Botです。改定内容を入力すると、影響を受ける可能性のあるシナリオ・FAQを検索します。"
  );
});

// メッセージ受信 → AI Search ハイブリッド検索
agentApp.onActivity(ActivityTypes.Message, async (context: TurnContext) => {
  const query = context.activity.text;
  if (!query) {
    await context.sendActivity("テキストを入力してください。");
    return;
  }

  await context.sendActivity(`「${query}」で検索中...`);

  try {
    const results = await searchClient.search(query, {
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
      top: 5,
      filter: "isDeleted eq false",
    });

    const items: string[] = [];
    for await (const result of results.results) {
      const doc = result.document as Record<string, unknown>;
      const type = doc.dataType === "scenario" ? "シナリオ" : "FAQ";
      const score = result.score?.toFixed(3) ?? "-";
      items.push(
        `**[${type}] ${doc.title}**\n` +
          `カテゴリ: ${doc.categoryName} | スコア: ${score}\n` +
          `${String(doc.content).substring(0, 200)}...`
      );
    }

    if (items.length === 0) {
      await context.sendActivity("該当する候補が見つかりませんでした。");
    } else {
      await context.sendActivity(
        `**影響候補 ${items.length}件:**\n\n---\n\n${items.join("\n\n---\n\n")}`
      );
    }
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    await context.sendActivity(`検索エラー: ${message}`);
  }
});
