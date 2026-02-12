import { CosmosClient, Container, Database } from "@azure/cosmos";
import { DefaultAzureCredential } from "@azure/identity";
import config from "./config";

// 遅延初期化（モジュール読み込み時に環境変数が空でもエラーにならない）
let _db: Database | null = null;
function getDb(): Database {
  if (!_db) {
    const credential = new DefaultAzureCredential();
    const client = new CosmosClient({
      endpoint: config.cosmosDbEndpoint,
      aadCredentials: credential,
    });
    _db = client.database(config.cosmosDbDatabase);
  }
  return _db;
}

/** FAQ論理削除 (isDeleted = true) */
export async function deleteFaqs(
  ids: string[]
): Promise<{ id: string; title: string }[]> {
  const container: Container = getDb().container("faqs");
  const results: { id: string; title: string }[] = [];

  for (const id of ids) {
    const { resources } = await container.items
      .query({
        query: "SELECT * FROM c WHERE c.id = @id",
        parameters: [{ name: "@id", value: id }],
      })
      .fetchAll();

    if (resources.length === 0) continue;

    const doc = resources[0];
    doc.isDeleted = true;
    await container.item(doc.id, doc.categoryId).replace(doc);
    results.push({ id: doc.id, title: doc.title });
  }
  return results;
}

/** シナリオ要修正フラグ記録 (impactAssessments へ新規作成) */
export async function saveNeedsUpdate(
  scenarioIds: string[],
  searchQuery: string,
  assessedBy: string
): Promise<{ id: string; title: string }[]> {
  const db = getDb();
  const scenariosContainer: Container = db.container("scenarios");
  const assessContainer: Container = db.container("impactAssessments");
  const now = new Date().toISOString();
  const searchId = `search-${now.replace(/[-:T.]/g, "").slice(0, 14)}`;
  const results: { id: string; title: string }[] = [];

  for (const scenarioId of scenarioIds) {
    const { resources } = await scenariosContainer.items
      .query({
        query: "SELECT c.id, c.title, c.categoryName FROM c WHERE c.id = @id",
        parameters: [{ name: "@id", value: scenarioId }],
      })
      .fetchAll();

    if (resources.length === 0) continue;
    const scenario = resources[0];

    const assessDoc = {
      id: `assess-${searchId}-${scenarioId}`,
      searchId,
      scenarioId,
      searchQuery,
      impactStatus: "needsUpdate",
      assessedBy,
      assessedAt: now,
    };
    await assessContainer.items.create(assessDoc);
    results.push({ id: scenario.id, title: scenario.title });
  }
  return results;
}
