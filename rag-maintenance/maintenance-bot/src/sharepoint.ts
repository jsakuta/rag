import { Client } from "@microsoft/microsoft-graph-client";
import { TokenCredentialAuthenticationProvider } from "@microsoft/microsoft-graph-client/authProviders/azureTokenCredentials";
import { DefaultAzureCredential } from "@azure/identity";
import config from "./config";

export interface SpoUploadResult {
  webUrl: string;
  filename: string;
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

/** Excel バッファを SharePoint Online にアップロード（4MB 以下シンプルアップロード） */
export async function uploadExcelToSharePoint(
  buffer: Buffer,
  filename: string
): Promise<SpoUploadResult> {
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

  return {
    webUrl: response.webUrl,
    filename,
  };
}
