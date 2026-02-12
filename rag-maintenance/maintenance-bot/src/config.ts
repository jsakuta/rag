const config = {
  aiSearchEndpoint: process.env.AI_SEARCH_ENDPOINT || "",
  aiSearchIndexName: process.env.AI_SEARCH_INDEX_NAME || "maintenance-search-index",
  cosmosDbEndpoint: process.env.COSMOS_DB_ENDPOINT || "",
  cosmosDbDatabase: process.env.COSMOS_DB_DATABASE || "maintenance-db",
};

export default config;

export const CATEGORIES = [
  { id: "all",       name: "すべて",     description: "全分野横断検索" },
  { id: "smile",     name: "スマイル",   description: "シナリオ + FAQ" },
  { id: "souzoku",   name: "相続",       description: "シナリオ" },
  { id: "naibujimu", name: "内部事務",   description: "シナリオ" },
  { id: "torikaku",  name: "取引時確認", description: "シナリオ" },
  { id: "sousoku",   name: "総則",       description: "FAQ" },
  { id: "yokin",     name: "預金",       description: "FAQ" },
] as const;

export const TOP_N_OPTIONS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150] as const;
