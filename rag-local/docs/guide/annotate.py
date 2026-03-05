"""Pillow アノテーションスクリプト

スクリーンショットに番号付き丸、ハイライト矩形、テキストラベルを追加する。
"""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

GUIDE_DIR = Path(__file__).parent
SCREENSHOTS_DIR = GUIDE_DIR / "screenshots"
ANNOTATED_DIR = GUIDE_DIR / "annotated"
CONFIG_FILE = GUIDE_DIR / "annotation_config.json"

# フォント設定
FONT_NUMBER = "C:/Windows/Fonts/arial.ttf"
FONT_LABEL = "C:/Windows/Fonts/meiryo.ttc"

# 色設定
CIRCLE_COLOR = (220, 50, 50)       # 赤
CIRCLE_TEXT_COLOR = (255, 255, 255) # 白
HIGHLIGHT_COLOR = (255, 255, 0, 50) # 半透明黄
LABEL_BG_COLOR = (50, 50, 50, 200)  # 暗色半透明
LABEL_TEXT_COLOR = (255, 255, 255)   # 白

CIRCLE_RADIUS = 14
FONT_SIZE_NUMBER = 16
FONT_SIZE_LABEL = 13


def load_fonts():
    try:
        font_num = ImageFont.truetype(FONT_NUMBER, FONT_SIZE_NUMBER)
    except OSError:
        font_num = ImageFont.load_default()
    try:
        font_label = ImageFont.truetype(FONT_LABEL, FONT_SIZE_LABEL)
    except OSError:
        font_label = ImageFont.load_default()
    return font_num, font_label


def draw_circle(draw, x, y, num, font_num):
    """番号付き赤丸を描画"""
    r = CIRCLE_RADIUS
    draw.ellipse([x - r, y - r, x + r, y + r], fill=CIRCLE_COLOR)
    text = str(num)
    bbox = font_num.getbbox(text)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text((x - tw / 2, y - th / 2 - 2), text, fill=CIRCLE_TEXT_COLOR, font=font_num)


def draw_highlight(overlay_draw, x1, y1, x2, y2):
    """半透明黄色ハイライト矩形を描画"""
    overlay_draw.rectangle([x1, y1, x2, y2], fill=HIGHLIGHT_COLOR)


def draw_label(draw, x, y, text, font_label, anchor="left"):
    """テキストラベルを描画（暗色背景 + 白文字）"""
    if not text:
        return
    bbox = font_label.getbbox(text)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    pad = 4

    if anchor == "left":
        lx = x + CIRCLE_RADIUS + 6
    else:
        lx = x - tw - CIRCLE_RADIUS - 6
    ly = y - th / 2 - pad / 2

    draw.rectangle([lx - pad, ly - pad, lx + tw + pad, ly + th + pad],
                   fill=LABEL_BG_COLOR)
    draw.text((lx, ly), text, fill=LABEL_TEXT_COLOR, font=font_label)


def annotate_image(img_name, config):
    """1枚のスクリーンショットにアノテーションを追加"""
    src_path = SCREENSHOTS_DIR / f"{img_name}.png"
    if not src_path.exists():
        print(f"SKIP: {src_path} not found")
        return

    img = Image.open(src_path).convert("RGBA")

    # ハイライトレイヤー
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    for hl in config.get("highlights", []):
        draw_highlight(overlay_draw, hl["x1"], hl["y1"], hl["x2"], hl["y2"])

    img = Image.alpha_composite(img, overlay)

    # テキスト・丸レイヤー
    draw = ImageDraw.Draw(img)
    font_num, font_label = load_fonts()

    for circle in config.get("circles", []):
        cx, cy = circle["x"], circle["y"]
        draw_circle(draw, cx, cy, circle["num"], font_num)
        label = circle.get("label", "")
        # ラベルが右に収まるかチェック
        if cx > img.width * 0.7:
            draw_label(draw, cx, cy, label, font_label, anchor="right")
        else:
            draw_label(draw, cx, cy, label, font_label, anchor="left")

    # RGBA→RGB変換して保存
    out = img.convert("RGB")
    out_path = ANNOTATED_DIR / f"{img_name}.png"
    out.save(out_path, "PNG")
    print(f"OK: {out_path}")


def main():
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

    with open(CONFIG_FILE, encoding="utf-8") as f:
        configs = json.load(f)

    for img_name, config in configs.items():
        annotate_image(img_name, config)

    print(f"\n完了: {len(configs)} 枚のアノテーション済み画像を生成しました")


if __name__ == "__main__":
    main()
