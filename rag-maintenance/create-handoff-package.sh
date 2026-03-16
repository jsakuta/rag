#!/bin/bash
# =============================================================================
# 引き継ぎパッケージ作成スクリプト
#
# rag-maintenance/ 配下から引き継ぎに必要なファイルのみを
# handoff/ フォルダにコピーする。
#
# 使い方: bash create-handoff-package.sh
# 出力先: ./handoff/
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="$SCRIPT_DIR"
DEST="$SCRIPT_DIR/handoff"

# 既存の handoff/ があれば削除して再作成
if [ -d "$DEST" ]; then
    echo "[INFO] 既存の handoff/ を削除します..."
    rm -rf "$DEST"
fi

echo "[INFO] 引き継ぎパッケージを作成します: $DEST"
mkdir -p "$DEST"

# =========================================
# 1. トップレベルファイル
# =========================================
echo "[1/6] トップレベルファイルをコピー..."
cp "$SRC/README.md" "$DEST/"

# =========================================
# 2. docs/ （レビュー報告書、scripts/、plans/ を除外）
# =========================================
echo "[2/6] docs/ をコピー..."
mkdir -p "$DEST/docs"

# ドキュメント本体（DOCX版を引き継ぎ）
for f in "$SRC/docs/"*.docx; do
    [ -f "$f" ] || continue
    cp "$f" "$DEST/docs/"
    echo "  [COPY] docs/$(basename "$f")"
done
echo "  [SKIP] docs/*.md（DOCX版を使用）"

# screenshots/ と drawings/ は DOCX に埋め込み済みのため除外
echo "  [SKIP] docs/screenshots/（DOCX内に埋め込み済み）"
echo "  [SKIP] docs/drawings/（DOCX内に埋め込み済み）"

# docs/scripts/, docs/plans/ は除外
echo "  [SKIP] docs/scripts/（文書加工ツール）"
echo "  [SKIP] docs/plans/（設計ドラフト）"

# =========================================
# 3. scripts/ （seed-cosmos.js, data/ を除外）
# =========================================
echo "[3/6] scripts/ をコピー..."
mkdir -p "$DEST/scripts"

for f in "$SRC/scripts/"*; do
    basename="$(basename "$f")"
    # data/ ディレクトリを除外
    if [ -d "$f" ]; then
        echo "  [SKIP] scripts/$basename/（生成データ）"
        continue
    fi
    # seed-cosmos.js を除外（.ts版があるため）
    if [ "$basename" = "seed-cosmos.js" ]; then
        echo "  [SKIP] scripts/$basename（.ts版の重複）"
        continue
    fi
    # export-drawings.py を除外（drawings/ を引き継がないため不要）
    if [ "$basename" = "export-drawings.py" ]; then
        echo "  [SKIP] scripts/$basename（drawings除外のため不要）"
        continue
    fi
    cp "$f" "$DEST/scripts/"
done

# =========================================
# 4. maintenance-bot/ （不要物を除外）
# =========================================
echo "[4/6] maintenance-bot/ をコピー..."
BOT_SRC="$SRC/maintenance-bot"
BOT_DEST="$DEST/maintenance-bot"
mkdir -p "$BOT_DEST"

# 引き継ぎ対象のトップレベルファイル
for f in package.json package-lock.json tsconfig.json \
         m365agents.yml m365agents.local.yml \
         web.config create-deploy-zip.ps1 \
         .webappignore .gitignore; do
    if [ -f "$BOT_SRC/$f" ]; then
        cp "$BOT_SRC/$f" "$BOT_DEST/"
    fi
done

# src/
cp -r "$BOT_SRC/src" "$BOT_DEST/"

# .vscode/ （F5デバッグに必須。Toolkitはプロジェクト新規作成時のみ生成し、既存プロジェクトでは自動生成しない）
if [ -d "$BOT_SRC/.vscode" ]; then
    cp -r "$BOT_SRC/.vscode" "$BOT_DEST/"
    echo "  [COPY] maintenance-bot/.vscode/（F5デバッグ設定）"
fi

# infra/
if [ -d "$BOT_SRC/infra" ]; then
    cp -r "$BOT_SRC/infra" "$BOT_DEST/"
fi

# appPackage/ （build/ を除外、manifest.json のドメインをプレースホルダ化）
if [ -d "$BOT_SRC/appPackage" ]; then
    mkdir -p "$BOT_DEST/appPackage"
    for f in "$BOT_SRC/appPackage/"*; do
        basename="$(basename "$f")"
        if [ "$basename" = "build" ]; then
            echo "  [SKIP] maintenance-bot/appPackage/build/（ビルド生成物）"
            continue
        fi
        if [ -f "$f" ]; then
            cp "$f" "$BOT_DEST/appPackage/"
        fi
    done
    # manifest.json のドメインをプレースホルダに置換
    if [ -f "$BOT_DEST/appPackage/manifest.json" ]; then
        sed -i 's/bdxcorp\.sharepoint\.com/your-tenant.sharepoint.com/g' "$BOT_DEST/appPackage/manifest.json"
        echo "  [SANITIZED] manifest.json: ドメインをプレースホルダに置換"
    fi
fi

# env/ — 機密情報を含むファイルは除外、空ディレクトリのみ作成
mkdir -p "$BOT_DEST/env"
echo "  [MKDIR] maintenance-bot/env/（空ディレクトリ。.env.dev は手順書に従い手動作成）"

# 除外リスト表示
echo "  [SKIP] maintenance-bot/node_modules/（npm installで復元）"
echo "  [SKIP] maintenance-bot/lib/（tscで再生成）"
echo "  [SKIP] maintenance-bot/.plans/（実装計画書）"
echo "  [SKIP] maintenance-bot/.localConfigs*（ローカル一時設定）"
echo "  [SKIP] maintenance-bot/m365agents.playground.yml（Playground環境）"

# =========================================
# 5. 除外確認サマリ
# =========================================
echo ""
echo "============================================"
echo " 引き継ぎパッケージ作成完了"
echo "============================================"
echo ""
echo "出力先: $DEST"
echo ""

# ファイル数カウント
FILE_COUNT=$(find "$DEST" -type f | wc -l)
echo "ファイル数: $FILE_COUNT"
echo ""

echo "【除外されたもの（引き継ぎ不要）】"
echo "  - CLAUDE.md                  （Claude Code用コンテキスト）"
echo "  - .plans/                    （実装計画書アーカイブ）"
echo "  - .docreview/                （要件定義レビューログ）"
echo "  - .review/                   （コードレビュー差分）"
echo "  - .serena/                   （IDEメタデータ）"
echo "  - docs/レビュー報告書.md      （開発時品質レビュー）"
echo "  - docs/シナリオ情報設計.md    （引き継ぎ対象外）"
echo "  - docs/scripts/              （文書加工ツール）"
echo "  - docs/plans/                （設計ドラフト）"
echo "  - scripts/seed-cosmos.js     （.ts版の重複）"
echo "  - scripts/data/              （生成JSONデータ）"
echo "  - maintenance-bot/env/*.env* （機密情報。空ディレクトリのみ同梱）"
echo "  - maintenance-bot/node_modules/"
echo "  - maintenance-bot/lib/"
echo "  - maintenance-bot/appPackage/build/"
echo "  - maintenance-bot/.localConfigs*"
echo "  - maintenance-bot/.plans/"
echo "  - maintenance-bot/m365agents.playground.yml"
echo ""
echo "[INFO] 完了。handoff/ フォルダをそのまま引き継ぎに使用してください。"
