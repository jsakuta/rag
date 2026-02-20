/**
 * Adaptive Card ビルダー
 * 要件定義書 FR-001〜FR-005, FR-013, FR-014 準拠
 */
import type { AdaptiveCard } from "@microsoft/agents-hosting";
import { SCENARIO_CATEGORIES, FAQ_CATEGORIES, TOP_N_OPTIONS, ITEMS_PER_PAGE, ADAPTIVE_CARD_SIZE_LIMIT } from "./config";

// --- 型定義 ---
export interface SearchResultItem {
  id: string;
  dataType: "scenario" | "faq";
  categoryId: string;
  categoryName: string;
  title: string;
  content: string;
  score: number;
  order?: number;
}

export interface CategorySelection {
  scenarios: string[];  // 例: ["smile", "souzoku"]
  faqs: string[];       // 例: ["smile", "sousoku"]
}

// --- ToggleVisibility ターゲット定義 ---
const TOGGLE_SHOW_SCENARIO = [
  { elementId: "tabScenarioActive", isVisible: true },
  { elementId: "tabScenarioInactive", isVisible: false },
  { elementId: "tabFaqActive", isVisible: false },
  { elementId: "tabFaqInactive", isVisible: true },
  { elementId: "sectionScenario", isVisible: true },
  { elementId: "sectionFaq", isVisible: false },
];

const TOGGLE_SHOW_FAQ = [
  { elementId: "tabScenarioActive", isVisible: false },
  { elementId: "tabScenarioInactive", isVisible: true },
  { elementId: "tabFaqActive", isVisible: true },
  { elementId: "tabFaqInactive", isVisible: false },
  { elementId: "sectionScenario", isVisible: false },
  { elementId: "sectionFaq", isVisible: true },
];

// --- FR-001/FR-002: 統合検索カード（タブUI） ---
export function buildSearchCard(queryText: string): AdaptiveCard {
  return {
    type: "AdaptiveCard",
    $schema: "http://adaptivecards.io/schemas/adaptive-card.json",
    version: "1.5",
    body: [
      // --- ヘッダー ---
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
      // --- タブバー（カード幅2等分、上下中央揃え） ---
      {
        type: "ColumnSet",
        spacing: "Medium",
        separator: true,
        columns: [
          // シナリオタブ（アクティブ状態、初期表示）
          {
            type: "Column",
            id: "tabScenarioActive",
            isVisible: true,
            width: "stretch",
            verticalContentAlignment: "Center",
            selectAction: {
              type: "Action.ToggleVisibility",
              targetElements: TOGGLE_SHOW_SCENARIO,
            },
            items: [
              {
                type: "TextBlock",
                text: "シナリオ",
                weight: "Bolder",
                color: "Accent",
                horizontalAlignment: "Center",
              },
              {
                type: "TextBlock",
                text: "━━━━━━━━━━━━━━",
                color: "Accent",
                size: "Small",
                spacing: "None",
                horizontalAlignment: "Center",
              },
            ],
          },
          // シナリオタブ（非アクティブ状態、初期非表示）
          {
            type: "Column",
            id: "tabScenarioInactive",
            isVisible: false,
            width: "stretch",
            verticalContentAlignment: "Center",
            selectAction: {
              type: "Action.ToggleVisibility",
              targetElements: TOGGLE_SHOW_SCENARIO,
            },
            items: [
              {
                type: "TextBlock",
                text: "シナリオ",
                isSubtle: true,
                horizontalAlignment: "Center",
              },
              // アクティブタブの下線と同じ高さを確保するスペーサー
              {
                type: "TextBlock",
                text: " ",
                size: "Small",
                spacing: "None",
              },
            ],
          },
          // FAQタブ（アクティブ状態、初期非表示・緑系）
          {
            type: "Column",
            id: "tabFaqActive",
            isVisible: false,
            width: "stretch",
            verticalContentAlignment: "Center",
            selectAction: {
              type: "Action.ToggleVisibility",
              targetElements: TOGGLE_SHOW_FAQ,
            },
            items: [
              {
                type: "TextBlock",
                text: "FAQ",
                weight: "Bolder",
                color: "Good",
                horizontalAlignment: "Center",
              },
              {
                type: "TextBlock",
                text: "━━━━━━━━━━━━━━",
                color: "Good",
                size: "Small",
                spacing: "None",
                horizontalAlignment: "Center",
              },
            ],
          },
          // FAQタブ（非アクティブ状態、初期表示）
          {
            type: "Column",
            id: "tabFaqInactive",
            isVisible: true,
            width: "stretch",
            verticalContentAlignment: "Center",
            selectAction: {
              type: "Action.ToggleVisibility",
              targetElements: TOGGLE_SHOW_FAQ,
            },
            items: [
              {
                type: "TextBlock",
                text: "FAQ",
                isSubtle: true,
                horizontalAlignment: "Center",
              },
              // アクティブタブの下線と同じ高さを確保するスペーサー
              {
                type: "TextBlock",
                text: " ",
                size: "Small",
                spacing: "None",
              },
            ],
          },
        ],
      },
      // --- シナリオカテゴリ（初期表示） ---
      {
        type: "Container",
        id: "sectionScenario",
        isVisible: true,
        items: [
          {
            type: "TextBlock",
            text: "シナリオカテゴリ",
            weight: "Bolder",
            size: "Small",
          },
          ...SCENARIO_CATEGORIES.map((c) => ({
            type: "Input.Toggle" as const,
            id: `scat_${c.id}`,
            title: c.name,
            value: "true",
            spacing: "None" as const,
          })),
          // 表示件数
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
          // 検索ボタン
          {
            type: "TextBlock",
            text: "検索モードを選択してください",
            wrap: true,
            spacing: "Medium",
          },
          {
            type: "ActionSet",
            actions: [
              {
                type: "Action.Execute",
                title: "意味検索",
                verb: "searchSemantic",
                data: { query: queryText, targetType: "scenario" },
              },
              {
                type: "Action.Execute",
                title: "キーワード検索",
                verb: "searchKeyword",
                data: { query: queryText, targetType: "scenario" },
              },
            ],
          },
        ],
      },
      // --- FAQカテゴリ（初期非表示） ---
      {
        type: "Container",
        id: "sectionFaq",
        isVisible: false,
        items: [
          {
            type: "TextBlock",
            text: "FAQカテゴリ",
            weight: "Bolder",
            size: "Small",
          },
          ...FAQ_CATEGORIES.map((c) => ({
            type: "Input.Toggle" as const,
            id: `fcat_${c.id}`,
            title: c.name,
            value: "true",
            spacing: "None" as const,
          })),
          // 表示件数
          {
            type: "TextBlock",
            text: "各分野の表示件数",
            weight: "Bolder",
            size: "Small",
            spacing: "Medium",
          },
          {
            type: "Input.ChoiceSet",
            id: "topN_faq",
            value: "30",
            style: "compact",
            choices: TOP_N_OPTIONS.map((n) => ({
              title: `${n} 件`,
              value: String(n),
            })),
          },
          // 検索ボタン
          {
            type: "TextBlock",
            text: "検索モードを選択してください",
            wrap: true,
            spacing: "Medium",
          },
          {
            type: "ActionSet",
            actions: [
              {
                type: "Action.Execute",
                title: "意味検索",
                verb: "searchSemantic",
                data: { query: queryText, targetType: "faq" },
              },
              {
                type: "Action.Execute",
                title: "キーワード検索",
                verb: "searchKeyword",
                data: { query: queryText, targetType: "faq" },
              },
            ],
          },
        ],
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
  fixedPerPage?: number,
  searchSessionId?: string,
  needsUpdateIds?: Set<string>
): AdaptiveCard {
  const CARD_LIMIT = ADAPTIVE_CARD_SIZE_LIMIT;

  // ページ遷移時は前回確定した perPage を使用、初回は実アイテム数から開始
  const totalItems = scenarios.length + faqs.length;
  let perPage = Math.min(fixedPerPage ?? ITEMS_PER_PAGE, totalItems || 1);
  let card = buildResultCardInner(queryText, searchMode, scenarios, faqs, page, selectedCategories, topN, perPage, searchSessionId, needsUpdateIds);
  let cardJson = JSON.stringify(card);
  let cardSize = Buffer.byteLength(cardJson, "utf8");
  console.log(`[buildResultCard] perPage=${perPage}, size=${cardSize} bytes (UTF-8)`);

  // 超過時は二分探索で最大表示件数を特定（O(log n)）
  if (cardSize > CARD_LIMIT && perPage > 1) {
    let lo = 1;
    let hi = perPage - 1;
    while (lo < hi) {
      const mid = Math.ceil((lo + hi) / 2);
      card = buildResultCardInner(queryText, searchMode, scenarios, faqs, page, selectedCategories, topN, mid, searchSessionId, needsUpdateIds);
      cardJson = JSON.stringify(card);
      cardSize = Buffer.byteLength(cardJson, "utf8");
      if (cardSize <= CARD_LIMIT) {
        lo = mid; // mid件は収まる → もっと増やせるか試す
      } else {
        hi = mid - 1; // mid件は超過 → 減らす
      }
    }
    perPage = lo;
    card = buildResultCardInner(queryText, searchMode, scenarios, faqs, page, selectedCategories, topN, perPage, searchSessionId, needsUpdateIds);
    cardJson = JSON.stringify(card);
    cardSize = Buffer.byteLength(cardJson, "utf8");
    console.log(`[buildResultCard] binary search → perPage=${perPage}, size=${cardSize} bytes (UTF-8)`);
  }

  // perPage=1 でも超過する場合はコンテンツを段階的に切り詰めて再構築
  if (cardSize > CARD_LIMIT) {
    console.warn(`[buildResultCard] Card still ${cardSize} bytes with perPage=1 — truncating content`);
    const truncLimits = [500, 300, 150, 80];
    for (const limit of truncLimits) {
      const ts = scenarios.map((s) => ({ ...s, content: truncate(s.content, limit) }));
      const tf = faqs.map((f) => ({ ...f, content: truncate(f.content, limit) }));
      card = buildResultCardInner(queryText, searchMode, ts, tf, page, selectedCategories, topN, 1, searchSessionId, needsUpdateIds);
      cardJson = JSON.stringify(card);
      cardSize = Buffer.byteLength(cardJson, "utf8");
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
  perPage: number,
  searchSessionId?: string,
  needsUpdateIds?: Set<string>
): AdaptiveCard {
  const modeLabel = searchMode === "semantic" ? "意味検索" : "キーワード検索";

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
    searchSessionId,
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
          value: needsUpdateIds?.has(s.id) ? "true" : "false",
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
      const globalRank = start + pageItems.indexOf(f) + 1;
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
                  text: `${numEmoji(globalRank)} ${f.categoryName}`,
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
                  text: `スコア: ${f.score.toFixed(4)}`,
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
          text: f.title,
          weight: "Bolder",
          wrap: true,
        },
        {
          type: "TextBlock",
          text: `「${f.content}」`,
          wrap: true,
          size: "Small",
          isSubtle: true,
        },
        {
          type: "Input.Toggle",
          id: `faq_${f.id}`,
          title: "削除対象",
          value: "false", // FAQ削除は1ページ内操作のみ（ページ間保持は設計スコープ外）
        },
        {
          type: "TextBlock",
          text: "---",
          separator: true,
        }
      );
    }
  }

  // --- ページネーション（上段） ---
  const paginationActions: Record<string, unknown>[] = [];
  if (safePage > 1) {
    paginationActions.push({
      type: "Action.Execute",
      title: "← 前ページへ",
      verb: "searchPage",
      data: { ...pageData, page: safePage - 1 },
    });
  }
  if (safePage < totalPages) {
    paginationActions.push({
      type: "Action.Execute",
      title: "次ページへ →",
      verb: "searchPage",
      data: { ...pageData, page: safePage + 1 },
    });
  }
  if (paginationActions.length > 0) {
    body.push({ type: "ActionSet", actions: paginationActions });
  }

  // --- アクションボタン（下段・左寄せ） ---
  const actionButtons: Record<string, unknown>[] = [];
  if (pageScenarios.length > 0) {
    actionButtons.push({
      type: "Action.Execute",
      title: "要修正を保存",
      verb: "saveNeedsUpdate",
      data: { query: queryText, searchSessionId },
      style: "positive",
    });
  }
  if (pageFaqs.length > 0) {
    actionButtons.push({
      type: "Action.Execute",
      title: "選択したFAQを削除",
      verb: "confirmDeleteFaqs",
      style: "destructive",
    });
  }
  if (actionButtons.length > 0) {
    body.push({ type: "ActionSet", actions: actionButtons });
  }

  return {
    type: "AdaptiveCard",
    $schema: "http://adaptivecards.io/schemas/adaptive-card.json",
    version: "1.5",
    body,
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
  saved: { id: string; title: string; categoryName: string }[],
  user: string,
  searchSessionId?: string
): AdaptiveCard {
  const now = formatJST(new Date());
  const actions: Record<string, unknown>[] = [];

  // FR-015: Excel出力ボタン
  if (searchSessionId) {
    actions.push({
      type: "Action.Execute",
      title: "Excelで出力",
      verb: "exportExcel",
      data: { searchSessionId },
    });
  }

  // 検索結果に戻る
  if (searchSessionId) {
    actions.push({
      type: "Action.Execute",
      title: "検索結果に戻る",
      verb: "searchPage",
      data: { searchSessionId, page: 1 },
    });
  }

  // カテゴリ別にグループ化
  const grouped = new Map<string, { id: string; title: string }[]>();
  for (const s of saved) {
    const key = s.categoryName || "その他";
    if (!grouped.has(key)) grouped.set(key, []);
    grouped.get(key)!.push(s);
  }

  const body: Record<string, unknown>[] = [
    {
      type: "TextBlock",
      text: "要修正フラグを保存しました",
      weight: "Bolder",
      size: "Medium",
      color: "Good",
    },
  ];

  // カテゴリ別セクション
  for (const [catName, items] of grouped) {
    body.push({
      type: "TextBlock",
      text: `▼ ${catName}（${items.length}件）`,
      weight: "Bolder",
      size: "Small",
      spacing: "Medium",
      separator: true,
    });
    for (const s of items) {
      body.push({
        type: "TextBlock",
        text: `- ${s.title} → 要修正`,
        wrap: true,
        size: "Small",
      });
    }
  }

  body.push({
    type: "TextBlock",
    text: `記録者: ${user}  ${now}`,
    size: "Small",
    isSubtle: true,
    spacing: "Medium",
  });

  return {
    type: "AdaptiveCard",
    $schema: "http://adaptivecards.io/schemas/adaptive-card.json",
    version: "1.5",
    body,
    ...(actions.length > 0 ? { actions } : {}),
  };
}

// --- FR-015: Excel出力関連の型定義 ---

export interface ExcelExportFileInfo {
  categoryName: string;
  totalCount: number;
  needsUpdateCount: number;
  webUrl: string;
}

export interface ExcelExportCompleteCardParams {
  files: ExcelExportFileInfo[];
  folderUrl: string;
  searchSessionId?: string;
  failedCategories?: string[];
}

export interface ExcelExportErrorCardParams {
  message: string;
  searchSessionId?: string;
}

// --- FR-015: Excel出力完了カード ---
export function buildExcelExportCompleteCard(
  params: ExcelExportCompleteCardParams
): AdaptiveCard {
  const { files, folderUrl, searchSessionId, failedCategories } = params;
  const now = formatJST(new Date());

  const body: Record<string, unknown>[] = [
    {
      type: "TextBlock",
      text: "Excel出力完了",
      weight: "Bolder",
      size: "Medium",
      color: "Good",
    },
    {
      type: "TextBlock",
      text: `出力日時: ${now}`,
      size: "Small",
      isSubtle: true,
    },
  ];

  // 各ファイル情報を表示
  for (const file of files) {
    body.push(
      {
        type: "ColumnSet",
        separator: true,
        spacing: "Medium",
        columns: [
          {
            type: "Column",
            width: "stretch",
            items: [
              {
                type: "TextBlock",
                text: `${file.categoryName}（${file.totalCount}件 / 要修正: ${file.needsUpdateCount}件）`,
                weight: "Bolder",
                size: "Small",
                wrap: true,
              },
            ],
          },
          {
            type: "Column",
            width: "auto",
            items: [
              {
                type: "ActionSet",
                actions: [
                  {
                    type: "Action.OpenUrl",
                    title: "開く",
                    url: file.webUrl,
                  },
                ],
              },
            ],
          },
        ],
      }
    );
  }

  // 部分失敗警告
  if (failedCategories && failedCategories.length > 0) {
    body.push({
      type: "TextBlock",
      text: `⚠ 以下のカテゴリはアップロードに失敗しました: ${failedCategories.join("、")}`,
      wrap: true,
      color: "Warning",
      spacing: "Medium",
    });
  }

  // アクションボタン
  const actions: Record<string, unknown>[] = [];

  if (folderUrl) {
    actions.push({
      type: "Action.OpenUrl",
      title: "フォルダを開く",
      url: folderUrl,
    });
  }

  if (searchSessionId) {
    actions.push({
      type: "Action.Execute",
      title: "モード選択に戻る",
      verb: "showSearchCard",
      data: { searchSessionId },
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

// --- 検索処理中カード（非同期パターン用） ---
export function buildSearchProcessingCard(queryText: string): AdaptiveCard {
  return {
    type: "AdaptiveCard",
    $schema: "http://adaptivecards.io/schemas/adaptive-card.json",
    version: "1.5",
    body: [
      {
        type: "TextBlock",
        text: "検索中...",
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
        text: "検索結果は完了後にメッセージとして送信されます。再度テキストを入力して検索することもできます。",
        wrap: true,
        size: "Small",
        isSubtle: true,
      },
    ],
  };
}

// --- FR-015: Excel出力処理中カード ---
export function buildExcelProcessingCard(): AdaptiveCard {
  return {
    type: "AdaptiveCard",
    $schema: "http://adaptivecards.io/schemas/adaptive-card.json",
    version: "1.5",
    body: [
      {
        type: "TextBlock",
        text: "Excel出力を開始しました",
        weight: "Bolder",
        size: "Medium",
      },
      {
        type: "TextBlock",
        text: "カテゴリ全量のシナリオを取得し、Excelファイルを生成しています。完了後に結果が送信されます。",
        wrap: true,
        size: "Small",
        isSubtle: true,
      },
    ],
  };
}

// --- FR-015: Excel出力エラーカード ---
export function buildExcelExportErrorCard(
  params: ExcelExportErrorCardParams
): AdaptiveCard {
  const { message, searchSessionId } = params;

  const actions: Record<string, unknown>[] = [];
  if (searchSessionId) {
    actions.push({
      type: "Action.Execute",
      title: "モード選択に戻る",
      verb: "showSearchCard",
      data: { searchSessionId },
    });
  }

  return {
    type: "AdaptiveCard",
    $schema: "http://adaptivecards.io/schemas/adaptive-card.json",
    version: "1.5",
    body: [
      {
        type: "TextBlock",
        text: message,
        wrap: true,
        color: "Attention",
      },
    ],
    ...(actions.length > 0 ? { actions } : {}),
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
