const config = {
  aiSearchEndpoint: process.env.AI_SEARCH_ENDPOINT || "",
  aiSearchIndexName: process.env.AI_SEARCH_INDEX_NAME || "maintenance-search-index",
  cosmosDbEndpoint: process.env.COSMOS_DB_ENDPOINT || "",
  cosmosDbDatabase: process.env.COSMOS_DB_DATABASE || "maintenance-db",
};

export default config;
