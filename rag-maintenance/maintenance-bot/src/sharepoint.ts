import { Client } from "@microsoft/microsoft-graph-client";
import { TokenCredentialAuthenticationProvider } from "@microsoft/microsoft-graph-client/authProviders/azureTokenCredentials";
import { DefaultAzureCredential } from "@azure/identity";
import config from "./config";

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

const MAX_SIMPLE_UPLOAD_BYTES = 4 * 1024 * 1024; // 4MB

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

  // フォルダURLをファイルURLから構築（parentReference.webUrl のフォールバック）
  let folderUrl = response.parentReference?.webUrl ?? "";
  if (!folderUrl && response.webUrl) {
    const lastSlash = response.webUrl.lastIndexOf("/");
    if (lastSlash > 0) {
      folderUrl = response.webUrl.substring(0, lastSlash);
    }
  }

  return {
    webUrl: response.webUrl,
    filename,
    folderUrl,
  };
}
