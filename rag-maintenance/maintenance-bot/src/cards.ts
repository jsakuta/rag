/**
 * Adaptive Card ビルダー
 * 要件定義書 FR-001〜FR-005, FR-013, FR-014 準拠
 */
import type { AdaptiveCard } from "@microsoft/agents-hosting";
import { CATEGORIES, TOP_N_OPTIONS } from "./config";

// --- 型定義 ---
export interface SearchResultItem {
  id: string;
  dataType: "scenario" | "faq";
  categoryName: string;
  title: string;
  content: string;
  score: number;
}

// --- FR-001/FR-002: 検索モード選択カード ---
export function buildModeSelectCard(queryText: string): AdaptiveCard {
  return {
    type: "AdaptiveCard",
    $schema: "http://adaptivecards.io/schemas/adaptive-card.json",
    version: "1.5",
    body: [
      {
        type: "TextBlock",
        text: "事務改定 影響候補検出",
        weight: "Bolder",
        size: "Medium",
      },
      {
        type: "TextBlock",
        text: `入力: 「${truncate(queryText, 80)}」`,
        wrap: true,
        size: "Small",
        color: "Accent",
      },
      {
        type: "TextBlock",
        text: "業務分野",
        weight: "Bolder",
        size: "Small",
        spacing: "Medium",
      },
      {
        type: "Input.ChoiceSet",
        id: "categoryId",
        value: "all",
        style: "compact",
        choices: CATEGORIES.map((c) => ({
          title: `${c.name}（${c.description}）`,
          value: c.id,
        })),
      },
      {
        type: "TextBlock",
        text: "表示件数",
        weight: "Bolder",
        size: "Small",
        spacing: "Medium",
      },
      {
        type: "Input.ChoiceSet",
        id: "topN",
        value: "30",
        style: "compact",
        choices: TOP_N_OPTIONS.map((n) => ({
          title: `${n} 件`,
          value: String(n),
        })),
      },
      {
        type: "TextBlock",
        text: "検索モードを選択してください",
        wrap: true,
        spacing: "Medium",
      },
    ],
    actions: [
      {
        type: "Action.Execute",
        title: "ハイブリッド検索",
        verb: "searchSemantic",
        data: { query: queryText },
      },
      {
        type: "Action.Execute",
        title: "キーワード一致検索",
        verb: "searchKeyword",
        data: { query: queryText },
      },
    ],
  };
}

// --- FR-005: 検索結果カード（タブ切り替え + ページネーション） ---
export function buildResultCard(
  queryText: string,
  searchMode: string,
  items: SearchResultItem[],
  page: number = 1,
  categoryId: string = "all",
  topN: number = 30
): AdaptiveCard {
  const allScenarios = items.filter((i) => i.dataType === "scenario");
  const allFaqs = items.filter((i) => i.dataType === "faq");
  const modeLabel = searchMode === "semantic" ? "ハイブリッド検索" : "キーワード一致検索";

  // N ページ対応: 10件/ページ
  const ITEMS_PER_PAGE = 10;
  const totalItems = items.length;
  const totalPages = Math.ceil(totalItems / ITEMS_PER_PAGE);
  const safeCurrentPage = Math.max(1, Math.min(page, totalPages));

  // ページ内のアイテムを抽出
  const startIdx = (safeCurrentPage - 1) * ITEMS_PER_PAGE;
  const endIdx = startIdx + ITEMS_PER_PAGE;
  const pageItems = items.slice(startIdx, endIdx);
  const pageScenarios = pageItems.filter((i) => i.dataType === "scenario");
  const pageFaqs = pageItems.filter((i) => i.dataType === "faq");

  // ページ情報テキスト
  const pageInfo = totalPages > 1
    ? ` (${safeCurrentPage}/${totalPages}ページ)`
    : "";

  // 件数表示
  const scenarioLabel = `シナリオ (${pageScenarios.length}件 / 全${allScenarios.length}件)`;
  const faqLabel = `FAQ (${pageFaqs.length}件 / 全${allFaqs.length}件)`;

  // ページ遷移アクション
  const pageActions: Record<string, unknown>[] = [];
  if (totalPages > 1) {
    if (safeCurrentPage > 1) {
      pageActions.push({
        type: "Action.Execute",
        title: "← 前のページ",
        verb: "searchPage",
        data: { query: queryText, mode: searchMode, page: safeCurrentPage - 1, categoryId, topN },
      });
    }
    if (safeCurrentPage < totalPages) {
      pageActions.push({
        type: "Action.Execute",
        title: "次のページ →",
        verb: "searchPage",
        data: { query: queryText, mode: searchMode, page: safeCurrentPage + 1, categoryId, topN },
      });
    }
  }

  const card: AdaptiveCard = {
    type: "AdaptiveCard",
    $schema: "http://adaptivecards.io/schemas/adaptive-card.json",
    version: "1.5",
    body: [
      {
        type: "TextBlock",
        text: `事務改定 影響候補検出結果${pageInfo}`,
        weight: "Bolder",
        size: "Medium",
      },
      {
        type: "TextBlock",
        text: `入力: 「${truncate(queryText, 60)}」`,
        wrap: true,
        size: "Small",
        color: "Accent",
      },
      {
        type: "TextBlock",
        text: `検索モード: ${modeLabel} | 取得件数: ${totalItems}件`,
        size: "Small",
        isSubtle: true,
      },
      // --- タブ切り替えボタン ---
      {
        type: "ActionSet",
        actions: [
          {
            type: "Action.ToggleVisibility",
            title: scenarioLabel,
            targetElements: [
              { elementId: "scenarioContainer", isVisible: true },
              { elementId: "faqContainer", isVisible: false },
            ],
          },
          {
            type: "Action.ToggleVisibility",
            title: faqLabel,
            targetElements: [
              { elementId: "scenarioContainer", isVisible: false },
              { elementId: "faqContainer", isVisible: true },
            ],
          },
        ],
      },
      // --- シナリオタブ ---
      buildScenarioContainer(pageScenarios, queryText),
      // --- FAQタブ ---
      buildFaqContainer(pageFaqs),
      // --- ページ遷移 ---
      ...(pageActions.length > 0
        ? [{ type: "ActionSet", actions: pageActions }]
        : []),
    ],
  };

  return card;
}

function buildScenarioContainer(
  scenarios: SearchResultItem[],
  queryText: string
) {
  const items: Record<string, unknown>[] = [];

  for (let i = 0; i < scenarios.length; i++) {
    const s = scenarios[i];
    items.push(
      {
        type: "ColumnSet",
        columns: [
          {
            type: "Column",
            width: "auto",
            items: [
              {
                type: "TextBlock",
                text: `${numEmoji(i + 1)} ${s.categoryName}`,
                weight: "Bolder",
                size: "Small",
              },
            ],
          },
          {
            type: "Column",
            width: "stretch",
            items: [
              {
                type: "TextBlock",
                text: `スコア: ${s.score.toFixed(4)}`,
                size: "Small",
                horizontalAlignment: "Right",
                isSubtle: true,
              },
            ],
          },
        ],
      },
      {
        type: "TextBlock",
        text: s.title,
        weight: "Bolder",
        wrap: true,
      },
      {
        type: "TextBlock",
        text: `「${truncate(s.content, 150)}」`,
        wrap: true,
        size: "Small",
        isSubtle: true,
      },
      {
        type: "Input.Toggle",
        id: `scenario_${s.id}`,
        title: "要修正",
        value: "false",
      },
      {
        type: "TextBlock",
        text: "---",
        separator: true,
      }
    );
  }

  if (scenarios.length === 0) {
    items.push({
      type: "TextBlock",
      text: "該当するシナリオはありません",
      isSubtle: true,
      wrap: true,
    });
  }

  return {
    type: "Container",
    id: "scenarioContainer",
    isVisible: true,
    style: "default",
    maxHeight: "400px",
    items: [
      ...items,
      ...(scenarios.length > 0
        ? [
            {
              type: "ActionSet",
              actions: [
                {
                  type: "Action.Execute",
                  title: "要修正を保存",
                  verb: "saveNeedsUpdate",
                  data: { query: queryText },
                  style: "positive",
                },
              ],
            },
          ]
        : []),
    ],
  };
}

function buildFaqContainer(faqs: SearchResultItem[]) {
  const items: Record<string, unknown>[] = [];

  for (let i = 0; i < faqs.length; i++) {
    const f = faqs[i];
    items.push({
      type: "Input.Toggle",
      id: `faq_${f.id}`,
      title: `${f.id} | ${f.categoryName} | ${f.title} | スコア: ${f.score.toFixed(4)}`,
      value: "false",
    });
    items.push({
      type: "TextBlock",
      text: `「${truncate(f.content, 150)}」`,
      wrap: true,
      size: "Small",
      isSubtle: true,
      spacing: "None",
    });
  }

  if (faqs.length === 0) {
    items.push({
      type: "TextBlock",
      text: "該当するFAQはありません",
      isSubtle: true,
      wrap: true,
    });
  }

  return {
    type: "Container",
    id: "faqContainer",
    isVisible: false,
    style: "default",
    maxHeight: "400px",
    items: [
      ...items,
      ...(faqs.length > 0
        ? [
            {
              type: "ActionSet",
              actions: [
                {
                  type: "Action.Execute",
                  title: "選択したFAQを削除",
                  verb: "confirmDeleteFaqs",
                  style: "destructive",
                },
              ],
            },
          ]
        : []),
    ],
  };
}

// --- FR-013: FAQ削除確認カード ---
export function buildDeleteConfirmCard(
  faqs: { id: string; title: string; categoryName?: string }[]
): AdaptiveCard {
  return {
    type: "AdaptiveCard",
    $schema: "http://adaptivecards.io/schemas/adaptive-card.json",
    version: "1.5",
    body: [
      {
        type: "TextBlock",
        text: `以下の${faqs.length}件のFAQを削除しますか？`,
        weight: "Bolder",
        size: "Medium",
        color: "Attention",
      },
      ...faqs.map((f) => ({
        type: "TextBlock",
        text: `- ${f.id}: ${f.categoryName ?? ""} / ${f.title}`,
        wrap: true,
        size: "Small",
      })),
    ],
    actions: [
      {
        type: "Action.Execute",
        title: "削除実行",
        verb: "executeDeleteFaqs",
        data: { faqIds: faqs.map((f) => f.id) },
        style: "destructive",
      },
      {
        type: "Action.Execute",
        title: "キャンセル",
        verb: "cancelAction",
      },
    ],
  };
}

// --- FR-013: FAQ削除完了カード ---
export function buildDeleteCompleteCard(
  deleted: { id: string; title: string }[],
  user: string
): AdaptiveCard {
  const now = formatJST(new Date());
  return {
    type: "AdaptiveCard",
    $schema: "http://adaptivecards.io/schemas/adaptive-card.json",
    version: "1.5",
    body: [
      {
        type: "TextBlock",
        text: `${deleted.length}件のFAQを削除しました`,
        weight: "Bolder",
        size: "Medium",
        color: "Good",
      },
      ...deleted.map((d) => ({
        type: "TextBlock",
        text: `- ${d.id}: ${d.title} → 削除済`,
        wrap: true,
        size: "Small",
      })),
      {
        type: "TextBlock",
        text: `削除者: ${user}  ${now}`,
        size: "Small",
        isSubtle: true,
      },
    ],
  };
}

// --- FR-014: 要修正フラグ保存完了カード ---
export function buildNeedsUpdateCompleteCard(
  saved: { id: string; title: string }[],
  user: string
): AdaptiveCard {
  const now = formatJST(new Date());
  return {
    type: "AdaptiveCard",
    $schema: "http://adaptivecards.io/schemas/adaptive-card.json",
    version: "1.5",
    body: [
      {
        type: "TextBlock",
        text: "要修正フラグを保存しました",
        weight: "Bolder",
        size: "Medium",
        color: "Good",
      },
      ...saved.map((s) => ({
        type: "TextBlock",
        text: `- ${s.title} → 要修正`,
        wrap: true,
        size: "Small",
      })),
      {
        type: "TextBlock",
        text: `記録者: ${user}  ${now}`,
        size: "Small",
        isSubtle: true,
      },
    ],
  };
}

// --- キャンセルカード ---
export function buildCancelCard(): AdaptiveCard {
  return {
    type: "AdaptiveCard",
    $schema: "http://adaptivecards.io/schemas/adaptive-card.json",
    version: "1.5",
    body: [
      {
        type: "TextBlock",
        text: "操作をキャンセルしました。",
        isSubtle: true,
      },
    ],
  };
}

// --- ユーティリティ ---
function truncate(s: string, max: number): string {
  return s.length > max ? s.substring(0, max) + "..." : s;
}

function numEmoji(n: number): string {
  const emojis = ["", "❶", "❷", "❸", "❹", "❺", "❻", "❼", "❽", "❾", "❿"];
  return emojis[n] ?? `(${n})`;
}

function formatJST(d: Date): string {
  return d.toLocaleString("ja-JP", {
    timeZone: "Asia/Tokyo",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}
