/**
 * Cosmos DB テストデータ投入スクリプト
 *
 * 使用法:
 *   npx ts-node scripts/seed-cosmos.ts              # JSON から upsert
 *   npx ts-node scripts/seed-cosmos.ts --clean       # 既存データ全削除後に投入
 *
 * 前提:
 *   scripts/data/scenarios.json と scripts/data/faqs.json が存在すること
 *   （python scripts/convert-excel-to-json.py で生成）
 */
import { CosmosClient, Container } from "@azure/cosmos";
import { DefaultAzureCredential } from "@azure/identity";
import * as fs from "fs";
import * as path from "path";

const ENDPOINT = "https://cosmos-maintenance-poc.documents.azure.com:443/";
const DB_NAME = "maintenance-db";
const BATCH_SIZE = 100;

interface Document {
  id: string;
  categoryId: string;
  [key: string]: unknown;
}

async function deleteAllItems(container: Container, containerName: string): Promise<number> {
  console.log(`  ${containerName}: 既存データを削除中...`);
  let deleted = 0;
  const { resources: items } = await container.items
    .query("SELECT c.id, c.categoryId FROM c")
    .fetchAll();

  for (const item of items) {
    await container.item(item.id, item.categoryId).delete();
    deleted++;
    if (deleted % 100 === 0) {
      process.stdout.write(`\r  ${containerName}: ${deleted}/${items.length}件削除`);
    }
  }
  if (deleted > 0) {
    process.stdout.write(`\r  ${containerName}: ${deleted}件削除完了\n`);
  } else {
    console.log(`  ${containerName}: 削除対象なし`);
  }
  return deleted;
}

async function upsertBatch(
  container: Container,
  docs: Document[],
  containerName: string
): Promise<number> {
  let upserted = 0;
  const total = docs.length;

  for (let i = 0; i < total; i += BATCH_SIZE) {
    const batch = docs.slice(i, i + BATCH_SIZE);
    const promises = batch.map((doc) =>
      container.items.upsert(doc).catch((err) => {
        console.error(`\n  [ERROR] ${doc.id}: ${err.message}`);
        return null;
      })
    );
    const results = await Promise.all(promises);
    upserted += results.filter((r) => r !== null).length;
    process.stdout.write(
      `\r  ${containerName}: ${upserted}/${total}件投入`
    );
  }
  process.stdout.write(`\r  ${containerName}: ${upserted}/${total}件投入完了\n`);
  return upserted;
}

async function main() {
  const args = process.argv.slice(2);
  const cleanMode = args.includes("--clean");

  // JSON ファイル読み込み
  const dataDir = path.join(__dirname, "data");
  const scenariosPath = path.join(dataDir, "scenarios.json");
  const faqsPath = path.join(dataDir, "faqs.json");

  if (!fs.existsSync(scenariosPath) || !fs.existsSync(faqsPath)) {
    console.error(
      "エラー: scripts/data/scenarios.json または faqs.json が見つかりません。\n" +
        "先に python scripts/convert-excel-to-json.py を実行してください。"
    );
    process.exit(1);
  }

  const scenarios: Document[] = JSON.parse(
    fs.readFileSync(scenariosPath, "utf-8")
  );
  const faqs: Document[] = JSON.parse(fs.readFileSync(faqsPath, "utf-8"));

  console.log("=".repeat(60));
  console.log("Cosmos DB テストデータ投入");
  console.log(`  モード: ${cleanMode ? "クリーン（全削除後投入）" : "upsert"}`);
  console.log(`  シナリオ: ${scenarios.length}件`);
  console.log(`  FAQ: ${faqs.length}件`);
  console.log("=".repeat(60));

  // Cosmos DB 接続
  const credential = new DefaultAzureCredential();
  const client = new CosmosClient({
    endpoint: ENDPOINT,
    aadCredentials: credential,
  });
  const db = client.database(DB_NAME);

  const scenarioContainer = db.container("scenarios");
  const faqContainer = db.container("faqs");

  // クリーンモード: 既存データ削除
  if (cleanMode) {
    console.log("\n--- 既存データ削除 ---");
    await deleteAllItems(scenarioContainer, "scenarios");
    await deleteAllItems(faqContainer, "faqs");
  }

  // データ投入
  console.log("\n--- データ投入 ---");
  const sCount = await upsertBatch(scenarioContainer, scenarios, "scenarios");
  const fCount = await upsertBatch(faqContainer, faqs, "faqs");

  console.log("\n" + "=".repeat(60));
  console.log(
    `完了: scenarios ${sCount}件, faqs ${fCount}件を投入しました`
  );
  console.log("=".repeat(60));
}

main().catch((err) => {
  console.error("エラー:", err.message);
  process.exit(1);
});
