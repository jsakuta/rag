"""
スクリーンショットに番号付き注釈（赤丸＋白数字）を追加するスクリプト。
導入手順書 Word 変換用。
"""
from PIL import Image, ImageDraw, ImageFont
import os
import sys

BASE = os.path.join(os.path.dirname(__file__), "..", "screenshots")
OUT = os.path.join(BASE, "annotated")

# --- Annotation definitions ---
# Each entry: (x, y, number_str)
# x, y = center of the red circle (pixel coordinates)
ANNOTATIONS = {
    # Step 10.2/10.3: Toolkit sidebar
    # 352x1031 — 左端のため右上に配置
    "step08_1_toolkit_sidebar.png": [
        (48, 430, "1"),   # M365 Agents Toolkit icon (右上に配置、左端で見切れ防止)
        (135, 220, "2"),  # "新しいエージェント/アプリの作成" button (左上)
    ],
    # Step 10.6: Provision completed
    # 344x998 — 各セクションの左上に配置
    "step08_3_provision_completed.png": [
        (55, 35, "1"),    # ACCOUNTS section (左上)
        (55, 185, "2"),   # ENVIRONMENT section (左上)
        (55, 445, "3"),   # LIFECYCLE section (左上)
    ],
    # Step 16.4: Admin center app list
    # 1914x991
    "step14_2_admin_app_list.png": [
        (100, 730, "1"),  # App list row (左上)
        (1690, 730, "2"), # "maintenance-bot..." app name (左上)
    ],
    # Step 16.4: Admin center publish button
    # 1919x908 — button color RGB(91,95,199) at x:740-840, y:320-360
    "step14_3_admin_publish_button.png": [
        (730, 310, "1"),  # "公開" button (左上)
    ],
    # Step 16.4: Admin center publish confirm dialog
    # 1917x909 — dialog button at x:990-1085, y:536-567
    "step14_4_admin_publish_confirm.png": [
        (985, 530, "1"),  # "公開" button in dialog (左上)
    ],
    # Step 17.5: Teams app store
    # 1280x720 — 左端のため右上に配置
    "step15_4_apps_store.png": [
        (55, 535, "1"),   # "アプリ" icon (右上に配置、左端で見切れ防止)
        (110, 278, "2"),  # "組織向けに開発" category (左上)
    ],
    # Step 17.5: Org apps list
    # 1280x720
    "step15_5_org_apps.png": [
        (840, 375, "1"),  # "maintenance-botdev" app card (左上)
        (1190, 383, "2"), # "追加" button (左上)
    ],
    # Step 17.5: App detail → "追加"
    # 1280x720
    "step15_6_app_detail.png": [
        (385, 105, "1"),  # "追加" button (左上)
    ],
    # Step 17.5: Bot Adaptive Card response
    # 1280x720
    "step15_8_bot_response.png": [
        (230, 178, "1"),  # Card title "事務改定 影響候補検出" (左上)
        (290, 248, "2"),  # Tab area シナリオ (左上)
        (235, 298, "3"),  # Category checkboxes (左上)
        (235, 562, "4"),  # Search mode buttons (左上)
    ],
}


def get_circle_radius(img_width: int) -> int:
    """Image width に応じた注釈円の半径を返す"""
    if img_width < 500:
        return 16
    elif img_width < 1000:
        return 20
    else:
        return 24


def get_font(size: int) -> ImageFont.FreeTypeFont:
    """Bold font を取得"""
    font_paths = [
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            return ImageFont.truetype(fp, size)
    return ImageFont.load_default()


def draw_annotation(draw: ImageDraw.Draw, x: int, y: int, number: str,
                    radius: int, font: ImageFont.FreeTypeFont):
    """赤丸に白数字の注釈を描画"""
    # Red filled circle with white border
    bbox = (x - radius, y - radius, x + radius, y + radius)
    # Outer white border (for contrast on dark backgrounds)
    draw.ellipse(
        (bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2),
        fill="white",
    )
    # Red filled circle
    draw.ellipse(bbox, fill="#E03030")
    # White number text, centered
    text_bbox = draw.textbbox((0, 0), number, font=font)
    tw = text_bbox[2] - text_bbox[0]
    th = text_bbox[3] - text_bbox[1]
    tx = x - tw / 2
    ty = y - th / 2 - 1  # slight upward adjustment for visual centering
    draw.text((tx, ty), number, fill="white", font=font)


def annotate_image(filename: str, annotations: list):
    """画像にすべての注釈を追加して保存"""
    src = os.path.join(BASE, filename)
    if not os.path.exists(src):
        print(f"  SKIP (not found): {filename}")
        return

    img = Image.open(src).convert("RGBA")
    w, h = img.size
    radius = get_circle_radius(w)
    font_size = int(radius * 1.3)
    font = get_font(font_size)

    # Create overlay for annotations
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for (x, y, number) in annotations:
        # Clamp coordinates to image bounds
        x = max(radius + 3, min(w - radius - 3, x))
        y = max(radius + 3, min(h - radius - 3, y))
        draw_annotation(draw, x, y, number, radius, font)

    # Composite
    result = Image.alpha_composite(img, overlay).convert("RGB")
    dst = os.path.join(OUT, filename)
    result.save(dst, "PNG")
    print(f"  OK: {filename} ({len(annotations)} annotations)")


def main():
    os.makedirs(OUT, exist_ok=True)

    # Copy non-annotated images as-is
    all_pngs = [f for f in os.listdir(BASE)
                if f.endswith(".png") and os.path.isfile(os.path.join(BASE, f))]

    print(f"Processing {len(all_pngs)} screenshots...")
    for filename in sorted(all_pngs):
        if filename in ANNOTATIONS:
            annotate_image(filename, ANNOTATIONS[filename])
        else:
            # Copy as-is (no annotation needed)
            from shutil import copy2
            copy2(os.path.join(BASE, filename), os.path.join(OUT, filename))
            print(f"  COPY: {filename}")

    print(f"\nDone. Annotated images in: {OUT}")


if __name__ == "__main__":
    main()
