import { startServer } from "@microsoft/agents-hosting-express";
import { agentApp } from "./agent";
import { validateConfig } from "./config";

// 起動時に必須環境変数を確認（未設定の場合は警告ログ）
const missing = validateConfig();
if (missing.length > 0) {
  console.warn(`[config] 以下の必須環境変数が未設定です: ${missing.join(", ")}`);
  console.warn("[config] 設定後に再起動してください。");
}

startServer(agentApp);
