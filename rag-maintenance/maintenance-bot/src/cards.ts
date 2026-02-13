/**
 * Adaptive Card ビルダー
 * 要件定義書 FR-001〜FR-005, FR-013, FR-014 準拠
 */
import type { AdaptiveCard } from "@microsoft/agents-hosting";
import { SCENARIO_CATEGORIES, FAQ_CATEGORIES, TOP_N_OPTIONS, ITEMS_PER_PAGE } from "./config";

// --- 型定義 ---
export interface SearchResultItem {
  id: string;
  dataType: "scenario" | "faq";
  categoryName: string;
  title: string;
  content: string;
  score: number;
}

export interface CategorySelection {
  scenarios: string[];  // 例: ["smile", "souzoku"]
  faqs: string[];       // 例: ["smile", "sousoku"]
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
      // --- シナリオ ---
      {
        type: "TextBlock",
        text: "シナリオ",
        weight: "Bolder",
        size: "Small",
        spacing: "Medium",
      },
      ...SCENARIO_CATEGORIES.map((c) => ({
        type: "Input.Toggle",
        id: `cat_s_${c.id}`,
        title: c.name,
        value: "true",
        spacing: "None",
      })),
      // --- FAQ ---
      {
        type: "TextBlock",
        text: "FAQ",
        weight: "Bolder",
        size: "Small",
        spacing: "Medium",
      },
      ...FAQ_CATEGORIES.map((c) => ({
        type: "Input.Toggle",
        id: `cat_f_${c.id}`,
        title: c.name,
        value: "true",
        spacing: "None",
      })),
      // --- 表示件数 ---
      {
        type: "TextBlock",
        text: "各分野の表示件数",
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

// --- FR-005: 検索結果カード（統一ページネーション） ---
export function buildResultCard(
  queryText: string,
  searchMode: string,
  scenarios: SearchResultItem[],
  faqs: SearchResultItem[],
  page: number = 1,
  selectedCategories: CategorySelection,
  topN: number = 30,
  fixedPerPage?: number
): AdaptiveCard {
  const CARD_LIMIT = 28 * 1024;

  // ページ遷移時は前回確定した perPage を使用、初回は ITEMS_PER_PAGE から開始
  let perPage = fixedPerPage ?? ITEMS_PER_PAGE;
  let card = buildResultCardInner(queryText, searchMode, scenarios, faqs, page, selectedCategories, topN, perPage);
  let cardSize = JSON.stringify(card).length;
  console.log(`[buildResultCard] perPage=${perPage}, size=${cardSize} bytes`);

  // 超過時は1件ずつ減らして再構築（最低1件は表示）
  while (cardSize > CARD_LIMIT && perPage > 1) {
    perPage--;
    card = buildResultCardInner(queryText, searchMode, scenarios, faqs, page, selectedCategories, topN, perPage);
    cardSize = JSON.stringify(card).length;
    console.log(`[buildResultCard] reduced perPage=${perPage}, size=${cardSize} bytes`);
  }

  // perPage=1 でも超過する場合はコンテンツを段階的に切り詰めて再構築
  if (cardSize > CARD_LIMIT) {
    console.warn(`[buildResultCard] Card still ${cardSize} bytes with perPage=1 — truncating content`);
    const truncLimits = [500, 300, 150, 80];
    for (const limit of truncLimits) {
      const ts = scenarios.map((s) => ({ ...s, content: truncate(s.content, limit) }));
      const tf = faqs.map((f) => ({ ...f, content: truncate(f.content, limit) }));
      card = buildResultCardInner(queryText, searchMode, ts, tf, page, selectedCategories, topN, 1);
      cardSize = JSON.stringify(card).length;
      if (cardSize <= CARD_LIMIT) {
        console.log(`[buildResultCard] Rebuilt with truncate=${limit}, size=${cardSize} bytes`);
        return card;
      }
    }
    console.error(`[buildResultCard] Card still ${cardSize} bytes after max truncation`);
  }

  return card;
}

function buildResultCardInner(
  queryText: string,
  searchMode: string,
  scenarios: SearchResultItem[],
  faqs: SearchResultItem[],
  page: number,
  selectedCategories: CategorySelection,
  topN: number,
  perPage: number
): AdaptiveCard {
  const modeLabel = searchMode === "semantic" ? "ハイブリッド検索" : "キーワード一致検索";

  // 全結果をスコア順にマージ → 統一ページネーション
  const allItems = [...scenarios, ...faqs].sort((a, b) => b.score - a.score);
  const totalPages = Math.max(1, Math.ceil(allItems.length / perPage));
  const safePage = Math.max(1, Math.min(page, totalPages));
  const start = (safePage - 1) * perPage;
  const pageItems = allItems.slice(start, start + perPage);

  // ページ内のアイテムをタイプ別に分離
  const pageScenarios = pageItems.filter((i) => i.dataType === "scenario");
  const pageFaqs = pageItems.filter((i) => i.dataType === "faq");

  // ページ遷移用データ（perPage を含めてページ間の整合性を維持）
  const pageData = {
    query: queryText,
    mode: searchMode,
    selectedCategories,
    topN,
    perPage,
  };

  // body 構築
  const body: Record<string, unknown>[] = [
    {
      type: "TextBlock",
      text: "事務改定 影響候補検出結果",
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
      text: `検索モード: ${modeLabel}`,
      size: "Small",
      isSubtle: true,
    },
  ];

  // 全体ヘッダー
  if (allItems.length > 0) {
    const pageInfo = totalPages > 1 ? ` [${safePage}/${totalPages}ページ]` : "";
    body.push({
      type: "TextBlock",
      text: `━━ 検索結果 (${pageItems.length}件/全${allItems.length}件)${pageInfo} ━━`,
      weight: "Bolder",
      size: "Small",
      spacing: "Large",
      separator: true,
    });
  }

  // 両タイプ混在フラグ
  const hasBothTypes = pageScenarios.length > 0 && pageFaqs.length > 0;

  // --- シナリオセクション ---
  if (pageScenarios.length > 0) {
    if (hasBothTypes) {
      body.push({
        type: "TextBlock",
        text: "▼ シナリオ",
        weight: "Bolder",
        size: "Small",
        spacing: "Medium",
      });
    }

    for (let i = 0; i < pageScenarios.length; i++) {
      const s = pageScenarios[i];
      const globalRank = start + pageItems.indexOf(s) + 1;
      body.push(
        {
          type: "ColumnSet",
          columns: [
            {
              type: "Column",
              width: "auto",
              items: [
                {
                  type: "TextBlock",
                  text: `${numEmoji(globalRank)} ${s.categoryName}`,
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
          text: `「${s.content}」`,
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
  }

  // --- FAQセクション ---
  if (pageFaqs.length > 0) {
    if (hasBothTypes) {
      body.push({
        type: "TextBlock",
        text: "▼ FAQ",
        weight: "Bolder",
        size: "Small",
        spacing: "Medium",
      });
    }

    for (let i = 0; i < pageFaqs.length; i++) {
      const f = pageFaqs[i];
      body.push(
        {
          type: "Input.Toggle",
          id: `faq_${f.id}`,
          title: `${f.id} | ${f.categoryName} | ${f.title} | スコア: ${f.score.toFixed(4)}`,
          value: "false",
        },
        {
          type: "TextBlock",
          text: `「${f.content}」`,
          wrap: true,
          size: "Small",
          isSubtle: true,
          spacing: "None",
        }
      );
    }
  }

  // --- アクションボタン ---
  const actions: Record<string, unknown>[] = [];

  // 要修正を保存（シナリオがある場合）
  if (pageScenarios.length > 0) {
    actions.push({
      type: "Action.Execute",
      title: "要修正を保存",
      verb: "saveNeedsUpdate",
      data: { query: queryText },
      style: "positive",
    });
  }

  // FAQ削除（FAQがある場合）
  if (pageFaqs.length > 0) {
    actions.push({
      type: "Action.Execute",
      title: "選択したFAQを削除",
      verb: "confirmDeleteFaqs",
      style: "destructive",
    });
  }

  // ページ遷移ボタン
  if (safePage > 1) {
    actions.push({
      type: "Action.Execute",
      title: "← 前ページへ",
      verb: "searchPage",
      data: { ...pageData, page: safePage - 1 },
    });
  }
  if (safePage < totalPages) {
    actions.push({
      type: "Action.Execute",
      title: "次ページへ →",
      verb: "searchPage",
      data: { ...pageData, page: safePage + 1 },
    });
  }

  return {
    type: "AdaptiveCard",
    $schema: "http://adaptivecards.io/schemas/adaptive-card.json",
    version: "1.5",
    body,
    ...(actions.length > 0 ? { actions } : {}),
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
  const emojis: Record<number, string> = {
    1: "❶", 2: "❷", 3: "❸", 4: "❹", 5: "❺",
    6: "❻", 7: "❼", 8: "❽", 9: "❾", 10: "❿",
  };
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
