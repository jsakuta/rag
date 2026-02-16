const config = {
  aiSearchEndpoint: process.env.AI_SEARCH_ENDPOINT || "",
  aiSearchIndexName: process.env.AI_SEARCH_INDEX_NAME || "maintenance-search-index",
  cosmosDbEndpoint: process.env.COSMOS_DB_ENDPOINT || "",
  cosmosDbDatabase: process.env.COSMOS_DB_DATABASE || "maintenance-db",
};

export default config;

export const SCENARIO_CATEGORIES = [
  { id: "smile",     name: "スマイル" },
  { id: "souzoku",   name: "相続" },      // 相続業務
  { id: "naibujimu", name: "内部事務" },
  { id: "torikaku",  name: "取引時確認" },
] as const;

export const FAQ_CATEGORIES = [
  { id: "smile",   name: "スマイル" },
  { id: "sousoku", name: "総則" },         // 総則規定 ※ souzoku(相続) とは別
  { id: "yokin",   name: "預金" },
] as const;

export const TOP_N_OPTIONS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] as const;
export const DEFAULT_TOP_N = 30;
export const ITEMS_PER_PAGE = 100;
