/**
 * Word文書組み立てスクリプト (docx-js v8)
 *
 * アノテーション済みスクリーンショットとステップ説明を
 * ops_ui_guide.docx として組み立てる。
 */
const fs = require("fs");
const path = require("path");
const {
  Document,
  Paragraph,
  TextRun,
  ImageRun,
  Table,
  TableRow,
  TableCell,
  HeadingLevel,
  AlignmentType,
  BorderStyle,
  WidthType,
  PageBreak,
  Packer,
  LevelFormat,
  Header,
  ShadingType,
} = require("docx");

const GUIDE_DIR = process.argv[2] ? path.resolve(process.argv[2]) : __dirname;
const ANNOTATED_DIR = path.join(GUIDE_DIR, "annotated");
const SCREENSHOTS_DIR = path.join(GUIDE_DIR, "screenshots");
const STEP_FILE = path.join(GUIDE_DIR, "step_descriptions.json");

// ---------- helpers ----------

function loadImage(name) {
  // annotated版を優先、なければscreenshots版を使用
  const annotatedPath = path.join(ANNOTATED_DIR, `${name}.png`);
  const screenshotPath = path.join(SCREENSHOTS_DIR, `${name}.png`);
  if (fs.existsSync(annotatedPath)) return fs.readFileSync(annotatedPath);
  if (fs.existsSync(screenshotPath)) return fs.readFileSync(screenshotPath);
  console.warn(`Image not found: ${name}`);
  return null;
}

function imageRun(name) {
  const data = loadImage(name);
  if (!data) return new TextRun({ text: `[画像: ${name}]`, italics: true, color: "999999" });
  return new ImageRun({
    data,
    transformation: { width: 580, height: 326 },
    type: "png",
  });
}

function heading(text, level) {
  return new Paragraph({
    text,
    heading: level,
    spacing: { before: 240, after: 120 },
  });
}

function bodyText(text) {
  return new Paragraph({
    children: [new TextRun({ text, size: 21, font: "Meiryo UI" })],
    spacing: { after: 120 },
  });
}

function codeBlock(code) {
  return new Paragraph({
    children: [new TextRun({ text: code, font: "Consolas", size: 18, color: "333333" })],
    spacing: { before: 60, after: 120 },
    shading: { type: ShadingType.CLEAR, fill: "F5F5F5" },
    indent: { left: 360 },
  });
}

/** 番号・操作・説明 テーブル */
function stepsTable(steps) {
  const headerCellProps = {
    shading: { type: ShadingType.CLEAR, fill: "2B579A" },
    borders: allBorders(),
  };
  const headerRow = new TableRow({
    children: [
      new TableCell({
        ...headerCellProps,
        width: { size: 600, type: WidthType.DXA },
        children: [new Paragraph({ children: [new TextRun({ text: "#", bold: true, color: "FFFFFF", size: 20, font: "Meiryo UI" })], alignment: AlignmentType.CENTER })],
      }),
      new TableCell({
        ...headerCellProps,
        width: { size: 2400, type: WidthType.DXA },
        children: [new Paragraph({ children: [new TextRun({ text: "操作", bold: true, color: "FFFFFF", size: 20, font: "Meiryo UI" })] })],
      }),
      new TableCell({
        ...headerCellProps,
        width: { size: 6000, type: WidthType.DXA },
        children: [new Paragraph({ children: [new TextRun({ text: "説明", bold: true, color: "FFFFFF", size: 20, font: "Meiryo UI" })] })],
      }),
    ],
  });

  const dataRows = steps.map((s, idx) => {
    const cellProps = {
      borders: allBorders(),
      shading: idx % 2 === 0 ? { type: ShadingType.CLEAR, fill: "F8F9FA" } : undefined,
    };
    return new TableRow({
      children: [
        new TableCell({
          ...cellProps,
          width: { size: 600, type: WidthType.DXA },
          children: [new Paragraph({ children: [new TextRun({ text: circledNum(s.num), size: 20, font: "Meiryo UI", color: "DC3232" })], alignment: AlignmentType.CENTER })],
        }),
        new TableCell({
          ...cellProps,
          width: { size: 2400, type: WidthType.DXA },
          children: [new Paragraph({ children: [new TextRun({ text: s.action, bold: true, size: 20, font: "Meiryo UI" })] })],
        }),
        new TableCell({
          ...cellProps,
          width: { size: 6000, type: WidthType.DXA },
          children: [new Paragraph({ children: [new TextRun({ text: s.desc, size: 20, font: "Meiryo UI" })] })],
        }),
      ],
    });
  });

  return new Table({
    rows: [headerRow, ...dataRows],
    width: { size: 9000, type: WidthType.DXA },
  });
}

function allBorders() {
  const b = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
  return { top: b, bottom: b, left: b, right: b };
}

function circledNum(n) {
  const nums = ["", "\u2460", "\u2461", "\u2462", "\u2463", "\u2464", "\u2465", "\u2466", "\u2467", "\u2468"];
  return nums[n] || String(n);
}

// ---------- section builders ----------

function buildCover(cover) {
  return [
    new Paragraph({ spacing: { before: 3600 } }),
    new Paragraph({
      children: [new TextRun({ text: cover.title, size: 52, bold: true, font: "Meiryo UI", color: "2B579A" })],
      alignment: AlignmentType.CENTER,
      spacing: { after: 240 },
    }),
    new Paragraph({
      children: [new TextRun({ text: cover.subtitle, size: 36, font: "Meiryo UI", color: "555555" })],
      alignment: AlignmentType.CENTER,
      spacing: { after: 480 },
    }),
    new Paragraph({
      children: [new TextRun({ text: cover.date, size: 24, font: "Meiryo UI", color: "999999" })],
      alignment: AlignmentType.CENTER,
    }),
    new Paragraph({ children: [new PageBreak()] }),
  ];
}

function buildSectionWithImage(section) {
  const children = [];
  children.push(heading(section.heading, HeadingLevel.HEADING_2));

  if (section.text) children.push(bodyText(section.text));
  if (section.code) children.push(codeBlock(section.code));

  if (section.image) {
    children.push(new Paragraph({
      children: [imageRun(section.image)],
      alignment: AlignmentType.CENTER,
      spacing: { before: 120, after: 120 },
    }));
  }

  if (section.steps) {
    children.push(stepsTable(section.steps));
  }

  if (section.steps_group) {
    for (const group of section.steps_group) {
      if (group.image) {
        children.push(new Paragraph({
          children: [imageRun(group.image)],
          alignment: AlignmentType.CENTER,
          spacing: { before: 200, after: 120 },
        }));
      }
      if (group.steps) {
        children.push(stepsTable(group.steps));
        children.push(new Paragraph({ spacing: { after: 120 } }));
      }
    }
  }

  if (section.bullets) {
    for (const bullet of section.bullets) {
      children.push(new Paragraph({
        children: [new TextRun({ text: `\u2022 ${bullet}`, size: 21, font: "Meiryo UI" })],
        spacing: { after: 60 },
        indent: { left: 360 },
      }));
    }
  }

  return children;
}

function buildSubsections(subsections) {
  const children = [];
  for (const sub of subsections) {
    children.push(heading(sub.heading, HeadingLevel.HEADING_3));

    if (sub.text) children.push(bodyText(sub.text));
    if (sub.code) children.push(codeBlock(sub.code));

    if (sub.bullets) {
      for (const bullet of sub.bullets) {
        children.push(new Paragraph({
          children: [new TextRun({ text: `\u2022 ${bullet}`, size: 21, font: "Meiryo UI" })],
          spacing: { after: 60 },
          indent: { left: 360 },
        }));
      }
    }

    if (sub.image) {
      children.push(new Paragraph({
        children: [imageRun(sub.image)],
        alignment: AlignmentType.CENTER,
        spacing: { before: 120, after: 120 },
      }));
    }

    if (sub.steps) {
      children.push(stepsTable(sub.steps));
    }

    if (sub.steps_group) {
      for (const group of sub.steps_group) {
        if (group.image) {
          children.push(new Paragraph({
            children: [imageRun(group.image)],
            alignment: AlignmentType.CENTER,
            spacing: { before: 200, after: 120 },
          }));
        }
        if (group.steps) {
          children.push(stepsTable(group.steps));
          children.push(new Paragraph({ spacing: { after: 120 } }));
        }
      }
    }
  }
  return children;
}

// ---------- main ----------

async function main() {
  const data = JSON.parse(fs.readFileSync(STEP_FILE, "utf-8"));
  const OUTPUT_FILE = path.join(GUIDE_DIR, data.cover.output || "guide.docx");
  const headerText = data.cover.header || data.cover.title;
  const allChildren = [];

  // 表紙
  allChildren.push(...buildCover(data.cover));

  // 各セクション
  for (const section of data.sections) {
    allChildren.push(heading(section.heading, HeadingLevel.HEADING_1));

    if (section.subsections) {
      allChildren.push(...buildSubsections(section.subsections));
    }

    if (section.image) {
      allChildren.push(new Paragraph({
        children: [imageRun(section.image)],
        alignment: AlignmentType.CENTER,
        spacing: { before: 120, after: 120 },
      }));
    }

    if (section.steps) {
      allChildren.push(stepsTable(section.steps));
    }

    if (section.bullets) {
      for (const bullet of section.bullets) {
        allChildren.push(new Paragraph({
          children: [new TextRun({ text: `\u2022 ${bullet}`, size: 21, font: "Meiryo UI" })],
          spacing: { after: 60 },
          indent: { left: 360 },
        }));
      }
    }
  }

  const doc = new Document({
    styles: {
      default: {
        document: {
          run: { font: "Meiryo UI", size: 21 },
        },
        heading1: {
          run: { font: "Meiryo UI", size: 32, bold: true, color: "2B579A" },
          paragraph: { spacing: { before: 480, after: 200 } },
        },
        heading2: {
          run: { font: "Meiryo UI", size: 28, bold: true, color: "2B579A" },
          paragraph: { spacing: { before: 360, after: 160 } },
        },
        heading3: {
          run: { font: "Meiryo UI", size: 24, bold: true, color: "404040" },
          paragraph: { spacing: { before: 240, after: 120 } },
        },
      },
    },
    sections: [
      {
        headers: {
          default: new Header({
            children: [
              new Paragraph({
                children: [
                  new TextRun({ text: headerText, size: 16, color: "999999", font: "Meiryo UI" }),
                ],
                alignment: AlignmentType.RIGHT,
              }),
            ],
          }),
        },
        children: allChildren,
      },
    ],
  });

  const buffer = await Packer.toBuffer(doc);
  fs.writeFileSync(OUTPUT_FILE, buffer);
  console.log(`OK: ${OUTPUT_FILE}`);
  console.log(`Size: ${(buffer.length / 1024).toFixed(0)} KB`);
}

main().catch(console.error);
