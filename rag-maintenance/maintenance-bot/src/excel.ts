import ExcelJS from "exceljs";
import { SearchResultItem } from "./cards";

/** ExcelJS で影響候補シナリオ一覧を生成（インメモリ） */
export async function generateImpactExcel(
  scenarios: SearchResultItem[],
  needsUpdateIds: Set<string>
): Promise<Buffer> {
  const workbook = new ExcelJS.Workbook();
  const sheet = workbook.addWorksheet("影響候補シナリオ");

  sheet.columns = [
    { header: "カテゴリ", key: "categoryName", width: 15 },
    { header: "タイトル", key: "title", width: 40 },
    { header: "本文抜粋", key: "content", width: 50 },
    { header: "スコア", key: "score", width: 12 },
    { header: "ステータス", key: "status", width: 12 },
  ];

  // ヘッダー行を太字
  const headerRow = sheet.getRow(1);
  headerRow.font = { bold: true };
  headerRow.commit();

  const yellowFill: ExcelJS.Fill = {
    type: "pattern",
    pattern: "solid",
    fgColor: { argb: "FFFFFF00" },
  };

  for (const scenario of scenarios) {
    const isNeedsUpdate = needsUpdateIds.has(scenario.id);
    const row = sheet.addRow({
      categoryName: scenario.categoryName,
      title: scenario.title,
      content: truncateContent(scenario.content, 200),
      score: Number(scenario.score.toFixed(4)),
      status: isNeedsUpdate ? "要修正" : "—",
    });

    if (isNeedsUpdate) {
      row.eachCell({ includeEmpty: true }, (cell) => {
        cell.fill = yellowFill;
      });
    }
    row.commit();
  }

  const arrayBuffer = await workbook.xlsx.writeBuffer();
  return Buffer.from(arrayBuffer);
}

/** JST タイムスタンプ付きファイル名を生成 */
export function generateExcelFilename(): string {
  const now = new Date();
  const jst = new Date(now.getTime() + 9 * 60 * 60 * 1000);
  const y = jst.getUTCFullYear();
  const m = String(jst.getUTCMonth() + 1).padStart(2, "0");
  const d = String(jst.getUTCDate()).padStart(2, "0");
  const h = String(jst.getUTCHours()).padStart(2, "0");
  const min = String(jst.getUTCMinutes()).padStart(2, "0");
  return `影響候補_${y}${m}${d}_${h}${min}.xlsx`;
}

function truncateContent(s: string, max: number): string {
  return s.length > max ? s.substring(0, max) + "..." : s;
}
