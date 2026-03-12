import { Client } from "@microsoft/microsoft-graph-client";
import { TokenCredentialAuthenticationProvider } from "@microsoft/microsoft-graph-client/authProviders/azureTokenCredentials";
import { DefaultAzureCredential } from "@azure/identity";
import config, { SPO_SIMPLE_UPLOAD_LIMIT } from "./config";

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

const MAX_SIMPLE_UPLOAD_BYTES = SPO_SIMPLE_UPLOAD_LIMIT; // POC 実装上の安全側制限（config.ts で定義）

/** Excel バッファを SharePoint Online にアップロード（現行 POC では 4MB 以下に制限） */
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

  return {
    webUrl: response.webUrl ?? "",
    filename,
  };
}

/** アップロード先フォルダの SharePoint 閲覧URL を取得（AllItems.aspx 形式、結果キャッシュ） */
let _cachedFolderWebUrl: string | null = null;

export async function getFolderWebUrl(): Promise<string> {
  if (_cachedFolderWebUrl) return _cachedFolderWebUrl;

  try {
    const client = getGraphClient();
    const driveId = config.spoDriveId;
    const folder = config.spoUploadFolder;

    // ドライブ webUrl とフォルダ webUrl を並列取得
    const [driveInfo, folderItem] = await Promise.all([
      client.api(`/drives/${driveId}`).select("webUrl").get(),
      client.api(`/drives/${driveId}/root:/${encodeURIComponent(folder)}`).select("webUrl").get(),
    ]);

    const driveWebUrl: string = driveInfo.webUrl ?? "";
    const folderWebUrl: string = folderItem.webUrl ?? "";

    if (driveWebUrl && folderWebUrl) {
      const parsed = new URL(folderWebUrl);
      const serverRelativePath = decodeURIComponent(parsed.pathname);
      const result = `${driveWebUrl.replace(/\/+$/, "")}/Forms/AllItems.aspx?id=${encodeURIComponent(serverRelativePath)}`;
      _cachedFolderWebUrl = result;
      return result;
    }

    console.warn("[getFolderWebUrl] Partial data: driveWebUrl=%s, folderWebUrl=%s", driveWebUrl, folderWebUrl);
    return "";
  } catch (e) {
    console.warn("[getFolderWebUrl] Failed:", e);
    return "";
  }
}
