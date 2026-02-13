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

// --- FR-001/FR-002: 検索モード選択カード（チェックボックス化） ---
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
      // --- シナリオ検索対象 ---
      {
        type: "TextBlock",
        text: "シナリオ検索対象",
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
      // --- FAQ検索対象 ---
      {
        type: "TextBlock",
        text: "FAQ検索対象",
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

// --- FR-005: 検索結果カード（セクション表示・タブ廃止） ---
export function buildResultCard(
  queryText: string,
  searchMode: string,
  scenarios: SearchResultItem[],
  faqs: SearchResultItem[],
  sPage: number = 1,
  fPage: number = 1,
  selectedCategories: CategorySelection,
  topN: number = 30
): AdaptiveCard {
  const modeLabel = searchMode === "semantic" ? "ハイブリッド検索" : "キーワード一致検索";

  // シナリオ ページネーション
  const sTotalPages = Math.max(1, Math.ceil(scenarios.length / ITEMS_PER_PAGE));
  const sSafePage = Math.max(1, Math.min(sPage, sTotalPages));
  const sStart = (sSafePage - 1) * ITEMS_PER_PAGE;
  const pageScenarios = scenarios.slice(sStart, sStart + ITEMS_PER_PAGE);

  // FAQ ページネーション
  const fTotalPages = Math.max(1, Math.ceil(faqs.length / ITEMS_PER_PAGE));
  const fSafePage = Math.max(1, Math.min(fPage, fTotalPages));
  const fStart = (fSafePage - 1) * ITEMS_PER_PAGE;
  const pageFaqs = faqs.slice(fStart, fStart + ITEMS_PER_PAGE);

  // ページ遷移用の共通データ
  const pageData = {
    query: queryText,
    mode: searchMode,
    selectedCategories,
    topN,
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

  // --- シナリオセクション ---
  if (scenarios.length > 0) {
    const sPageInfo = sTotalPages > 1 ? ` [${sSafePage}/${sTotalPages}ページ]` : "";
    body.push({
      type: "TextBlock",
      text: `━━ シナリオ検索結果 (${pageScenarios.length}件/全${scenarios.length}件)${sPageInfo} ━━`,
      weight: "Bolder",
      size: "Small",
      spacing: "Large",
      separator: true,
    });

    for (let i = 0; i < pageScenarios.length; i++) {
      const s = pageScenarios[i];
      const globalIndex = sStart + i + 1;
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
                  text: `${numEmoji(globalIndex)} ${s.categoryName}`,
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
  }

  // --- FAQセクション ---
  if (faqs.length > 0) {
    const fPageInfo = fTotalPages > 1 ? ` [${fSafePage}/${fTotalPages}ページ]` : "";
    body.push({
      type: "TextBlock",
      text: `━━ FAQ検索結果 (${pageFaqs.length}件/全${faqs.length}件)${fPageInfo} ━━`,
      weight: "Bolder",
      size: "Small",
      spacing: "Large",
      separator: true,
    });

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
          text: `「${truncate(f.content, 150)}」`,
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

  // 要修正を保存（シナリオがある場合のみ）
  if (pageScenarios.length > 0) {
    actions.push({
      type: "Action.Execute",
      title: "要修正を保存",
      verb: "saveNeedsUpdate",
      data: { query: queryText },
      style: "positive",
    });
  }

  // FAQ削除（FAQがある場合のみ）
  if (pageFaqs.length > 0) {
    actions.push({
      type: "Action.Execute",
      title: "選択したFAQを削除",
      verb: "confirmDeleteFaqs",
      style: "destructive",
    });
  }

  // ページ遷移ボタン
  if (sSafePage > 1) {
    actions.push({
      type: "Action.Execute",
      title: "シナリオ ← 前",
      verb: "searchPage",
      data: { ...pageData, scenarioPage: sSafePage - 1, faqPage: fSafePage },
    });
  }
  if (sSafePage < sTotalPages) {
    actions.push({
      type: "Action.Execute",
      title: "シナリオ 次 →",
      verb: "searchPage",
      data: { ...pageData, scenarioPage: sSafePage + 1, faqPage: fSafePage },
    });
  }
  if (fSafePage > 1) {
    actions.push({
      type: "Action.Execute",
      title: "FAQ ← 前",
      verb: "searchPage",
      data: { ...pageData, scenarioPage: sSafePage, faqPage: fSafePage - 1 },
    });
  }
  if (fSafePage < fTotalPages) {
    actions.push({
      type: "Action.Execute",
      title: "FAQ 次 →",
      verb: "searchPage",
      data: { ...pageData, scenarioPage: sSafePage, faqPage: fSafePage + 1 },
    });
  }

  const card: AdaptiveCard = {
    type: "AdaptiveCard",
    $schema: "http://adaptivecards.io/schemas/adaptive-card.json",
    version: "1.5",
    body,
    ...(actions.length > 0 ? { actions } : {}),
  };

  // 28KB制限チェック（ITEMS_PER_PAGE=25で通常は~18KB以内）
  const cardSize = JSON.stringify(card).length;
  if (cardSize > 28 * 1024) {
    console.warn(`[buildResultCard] Card size ${cardSize} bytes exceeds 28KB limit`);
  }

  return card;
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
