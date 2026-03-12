const config = {
  aiSearchEndpoint: process.env.AI_SEARCH_ENDPOINT || "",
  aiSearchIndexName: process.env.AI_SEARCH_INDEX_NAME || "maintenance-search-index",
  cosmosDbEndpoint: process.env.COSMOS_DB_ENDPOINT || "",
  cosmosDbDatabase: process.env.COSMOS_DB_DATABASE || "maintenance-db",
  spoSiteId: process.env.SPO_SITE_ID || "",
  spoDriveId: process.env.SPO_DRIVE_ID || "",
  spoUploadFolder: process.env.SPO_UPLOAD_FOLDER || "影響候補シナリオ",
};

export default config;

// --- 起動時バリデーション ---
// 必須環境変数が未設定の場合に警告ログを出力する（遅延初期化は維持）
const REQUIRED_VARS = ["AI_SEARCH_ENDPOINT", "COSMOS_DB_ENDPOINT"] as const;
export function validateConfig(): string[] {
  return REQUIRED_VARS.filter((v) => !process.env[v]);
}

// --- アプリ定数 ---
export const MAX_INPUT_LENGTH = 2000;            // FR-001: テキスト入力上限文字数
export const VECTOR_WEIGHT = 4.5;               // AI Search: ベクトル検索の RRF 重み（BM25 は暗黙 1.0）
export const CACHE_TTL_MS = 30 * 60 * 1000;     // 検索結果キャッシュの有効期間（30分）
export const MAX_CACHE_SIZE = 50;               // キャッシュエントリ最大数
export const SEARCH_PAGE_SIZE = 1000;           // AI Search のページネーション上限（SDK制約）
export const ADAPTIVE_CARD_SIZE_LIMIT = 25 * 1024; // Adaptive Card サイズ上限 ≈ 25KB（UTF-8バイト）
export const SPO_SIMPLE_UPLOAD_LIMIT = 4 * 1024 * 1024; // POC 実装上の安全側制限（4MB）

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
