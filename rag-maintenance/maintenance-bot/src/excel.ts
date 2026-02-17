import ExcelJS from "exceljs";
import { SearchResultItem } from "./cards";

// --- 型定義 ---

export interface CategoryExcelResult {
  categoryId: string;
  categoryName: string;
  buffer: Buffer;
  filename: string;
  totalCount: number;
  needsUpdateCount: number;
}

// --- ユーティリティ ---

/**
 * タイトル文字列を "/" で分割し、階層配列として返す。
 * - null/undefined/空文字 → ["(空)"]
 * - 先頭・末尾のスラッシュは無視
 * - 連続スラッシュは空要素を生まない
 */
export function parseTitlePath(title: string | null | undefined): string[] {
  if (!title || title.trim() === "") return ["(空)"];
  const parts = title.split("/").map((p) => p.trim()).filter((p) => p !== "");
  return parts.length > 0 ? parts : ["(空)"];
}

/**
 * 1始まりの列番号を Excel 列文字に変換（1→A, 26→Z, 27→AA, ...）
 */
function columnLetter(colNum: number): string {
  let letter = "";
  let n = colNum;
  while (n > 0) {
    n--;
    letter = String.fromCharCode(65 + (n % 26)) + letter;
    n = Math.floor(n / 26);
  }
  return letter;
}

/**
 * JST タイムスタンプ文字列を生成（YYYYMMDD_HHMMSS）
 */
function jstTimestamp(): string {
  const now = new Date();
  const jst = new Date(now.getTime() + 9 * 60 * 60 * 1000);
  const y = jst.getUTCFullYear();
  const mo = String(jst.getUTCMonth() + 1).padStart(2, "0");
  const d = String(jst.getUTCDate()).padStart(2, "0");
  const h = String(jst.getUTCHours()).padStart(2, "0");
  const mi = String(jst.getUTCMinutes()).padStart(2, "0");
  const s = String(jst.getUTCSeconds()).padStart(2, "0");
  return `${y}${mo}${d}_${h}${mi}${s}`;
}

/**
 * ファイル名に使えない文字を除去
 */
function sanitizeFilename(s: string): string {
  return s.replace(/[\\/:*?"<>|]/g, "_");
}

// --- メイン関数 ---

/**
 * シナリオ一覧をカテゴリ別に分割し、各カテゴリの Excel ファイルを生成する。
 *
 * 列構成: Lv1 | 文字数 | Lv2 | 文字数 | ... | LvN | 文字数（回答もLvカラムに統合）
 * - タイトルを "/" で分割して階層化し、contentを末尾Lvに追加
 * - LEN 数式で文字数を自動算出
 * - 要修正行は黄色ハイライト（FFFFFF00）
 */
export async function generateCategoryExcels(
  scenarios: SearchResultItem[],
  needsUpdateIds: Set<string>
): Promise<CategoryExcelResult[]> {
  // カテゴリ別にグループ化
  const grouped = new Map<string, { categoryName: string; items: SearchResultItem[] }>();
  for (const s of scenarios) {
    const existing = grouped.get(s.categoryId);
    if (existing) {
      existing.items.push(s);
    } else {
      grouped.set(s.categoryId, { categoryName: s.categoryName, items: [s] });
    }
  }

  const results: CategoryExcelResult[] = [];
  const ts = jstTimestamp();

  for (const [categoryId, group] of grouped) {
    const { categoryName, items } = group;

    // orderでソート（元のExcel行順を復元）
    const sorted = [...items].sort((a, b) => (a.order ?? 0) - (b.order ?? 0));

    // タイトル解析 + contentをLvに統合（元のExcel構造を再現）
    const parsedRows = sorted.map((item) => {
      const pathLevels = parseTitlePath(item.title);
      const levels = [...pathLevels, item.content]; // 回答もLvカラムに統合
      return { item, levels };
    });
    const maxDepth = Math.max(1, ...parsedRows.map((r) => r.levels.length));

    // ワークブック生成
    const workbook = new ExcelJS.Workbook();
    const sheetName = categoryName.length > 31 ? categoryName.substring(0, 31) : categoryName;
    const sheet = workbook.addWorksheet(sheetName);

    // ヘッダー構築: Lv1 | 文字数 | Lv2 | 文字数 | ... | LvN | 文字数（回答列なし）
    const headerRow: string[] = [];
    for (let lv = 1; lv <= maxDepth; lv++) {
      headerRow.push(`Lv${lv}`, "文字数");
    }

    // 列幅設定
    const columns: Partial<ExcelJS.Column>[] = [];
    for (let lv = 1; lv <= maxDepth; lv++) {
      columns.push({ width: 25 }); // Lv列
      columns.push({ width: 8 });  // 文字数列
    }
    sheet.columns = columns;

    // ヘッダー行追加
    const hRow = sheet.addRow(headerRow);
    hRow.font = { bold: true };
    hRow.commit();

    // 黄色ハイライト用
    const yellowFill: ExcelJS.Fill = {
      type: "pattern",
      pattern: "solid",
      fgColor: { argb: "FFFFFF00" },
    };

    // データ行追加
    let needsUpdateCount = 0;
    for (const { item, levels } of parsedRows) {
      const isNeedsUpdate = needsUpdateIds.has(item.id);
      if (isNeedsUpdate) needsUpdateCount++;

      const rowIndex = sheet.rowCount + 1;
      const rowValues: (string | { formula: string })[] = [];

      // Lv1〜LvN の値 + LEN数式（contentもLvに含む）
      for (let lv = 0; lv < maxDepth; lv++) {
        const colNum = lv * 2 + 1; // 1始まり列番号（Lv列）
        const colLetter = columnLetter(colNum);
        rowValues.push(lv < levels.length ? levels[lv] : "");
        rowValues.push({ formula: `LEN(${colLetter}${rowIndex})` });
      }

      const row = sheet.addRow(rowValues);

      if (isNeedsUpdate) {
        row.eachCell({ includeEmpty: true }, (cell) => {
          cell.fill = yellowFill;
        });
      }
      row.commit();
    }

    const arrayBuffer = await workbook.xlsx.writeBuffer();
    const buffer = Buffer.from(arrayBuffer);

    const safeName = sanitizeFilename(categoryName);
    const filename = `scenario_${categoryId}_${safeName}_${ts}.xlsx`;

    results.push({
      categoryId,
      categoryName,
      buffer,
      filename,
      totalCount: items.length,
      needsUpdateCount,
    });
  }

  return results;
}
