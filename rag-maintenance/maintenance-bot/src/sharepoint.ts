import { Client } from "@microsoft/microsoft-graph-client";
import { TokenCredentialAuthenticationProvider } from "@microsoft/microsoft-graph-client/authProviders/azureTokenCredentials";
import { DefaultAzureCredential } from "@azure/identity";
import config, { SPO_SIMPLE_UPLOAD_LIMIT } from "./config";

export interface SpoUploadResult {
  webUrl: string;
  filename: string;
  folderUrl: string;
}

// --- Graph クライアント遅延初期化 ---
let _graphClient: Client | null = null;
function getGraphClient(): Client {
  if (!_graphClient) {
    const credential = new DefaultAzureCredential();
    const authProvider = new TokenCredentialAuthenticationProvider(credential, {
      scopes: ["https://graph.microsoft.com/.default"],
    });
    _graphClient = Client.initWithMiddleware({ authProvider });
  }
  return _graphClient;
}

const MAX_SIMPLE_UPLOAD_BYTES = SPO_SIMPLE_UPLOAD_LIMIT; // 4MB（config.ts で定義）

/** Excel バッファを SharePoint Online にアップロード（4MB 以下シンプルアップロード） */
export async function uploadExcelToSharePoint(
  buffer: Buffer,
  filename: string
): Promise<SpoUploadResult> {
  if (buffer.length > MAX_SIMPLE_UPLOAD_BYTES) {
    throw new Error(
      `ファイルサイズが4MBを超えています (${(buffer.length / 1024 / 1024).toFixed(1)}MB)。アップロードできません。`
    );
  }

  const client = getGraphClient();
  const folder = config.spoUploadFolder;
  const driveId = config.spoDriveId;

  const uploadPath = `/drives/${driveId}/root:/${encodeURIComponent(folder)}/${encodeURIComponent(filename)}:/content`;

  const response = await client
    .api(uploadPath)
    .header(
      "Content-Type",
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    .put(buffer);

  const fileWebUrl: string = response.webUrl ?? "";
  const lastSlash = fileWebUrl.lastIndexOf("/");
  return {
    webUrl: fileWebUrl,
    filename,
    folderUrl: lastSlash > 0 ? fileWebUrl.substring(0, lastSlash) : "",
  };
}

/** アップロード先フォルダの SharePoint WebURL を取得 */
export async function getFolderWebUrl(): Promise<string> {
  try {
    const client = getGraphClient();
    const driveId = config.spoDriveId;
    const folder = config.spoUploadFolder;
    const folderPath = `/drives/${driveId}/root:/${encodeURIComponent(folder)}`;
    const folderItem = await client.api(folderPath).select("webUrl").get();
    return folderItem.webUrl ?? "";
  } catch (e) {
    console.warn("[getFolderWebUrl] Failed:", e);
    return "";
  }
}
