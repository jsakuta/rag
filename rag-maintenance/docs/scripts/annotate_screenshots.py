"""Generate annotated screenshots for the setup guide."""

from pathlib import Path
from shutil import copy2

from PIL import Image, ImageDraw, ImageFont


BASE = Path(__file__).resolve().parent.parent / "screenshots"
OUT = BASE / "annotated"

# Each entry is (x, y, number_str). x and y are the center of the circle.
ANNOTATIONS = {
    "step07_1_toolkit_sidebar.png": [
        (18, 350, "1"),
        (70, 210, "2"),
    ],
    "step07_3_provision_completed.png": [
        (55, 20, "1"),
        (55, 165, "2"),
        (55, 445, "3"),
    ],
    "step13_2_admin_app_list.png": [
        (1480, 672, "1"),
        (340, 750, "2"),
    ],
    "step13_3_admin_publish_button.png": [
        (730, 310, "1"),
    ],
    "step13_4_admin_publish_confirm.png": [
        (985, 530, "1"),
    ],
    "step14_4_apps_store.png": [
        (60, 520, "1"),
        (90, 243, "2"),
    ],
    "step14_5_org_apps.png": [
        (820, 355, "1"),
        (1170, 363, "2"),
    ],
    "step14_6_app_detail.png": [
        (385, 105, "1"),
    ],
    "step14_8_bot_response.png": [
        (373, 600, "1"),
        (373, 659, "2"),
        (373, 724, "3"),
    ],
}

RADIUS_OVERRIDE = {
    "step14_8_bot_response.png": 18,
}


def get_circle_radius(img_width: int, filename: str = "") -> int:
    if filename in RADIUS_OVERRIDE:
        return RADIUS_OVERRIDE[filename]
    if img_width < 500:
        return 16
    if img_width < 1000:
        return 20
    return 24


def get_font(size: int) -> ImageFont.FreeTypeFont:
    font_paths = [
        Path("C:/Windows/Fonts/arialbd.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    for path in font_paths:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def draw_annotation(
    draw: ImageDraw.Draw,
    x: int,
    y: int,
    number: str,
    radius: int,
    font: ImageFont.FreeTypeFont,
) -> None:
    bbox = (x - radius, y - radius, x + radius, y + radius)
    draw.ellipse((bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2), fill="white")
    draw.ellipse(bbox, fill="#E03030")

    text_bbox = draw.textbbox((0, 0), number, font=font)
    tw = text_bbox[2] - text_bbox[0]
    th = text_bbox[3] - text_bbox[1]
    tx = x - tw / 2
    ty = y - th / 2 - 2
    draw.text((tx, ty), number, fill="white", font=font)


def annotate_image(filename: str, annotations: list[tuple[int, int, str]]) -> None:
    src = BASE / filename
    if not src.exists():
        print(f"  SKIP (not found): {filename}")
        return

    img = Image.open(src).convert("RGBA")
    width, height = img.size
    radius = get_circle_radius(width, filename)
    font = get_font(int(radius * 1.3))

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for x, y, number in annotations:
        x = max(radius + 3, min(width - radius - 3, x))
        y = max(radius + 3, min(height - radius - 3, y))
        draw_annotation(draw, x, y, number, radius, font)

    result = Image.alpha_composite(img, overlay).convert("RGB")
    result.save(OUT / filename, "PNG")
    print(f"  OK: {filename} ({len(annotations)} annotations)")


def main() -> None:
    OUT.mkdir(exist_ok=True)

    all_pngs = sorted(
        path.name
        for path in BASE.iterdir()
        if path.suffix.lower() == ".png" and path.is_file()
    )

    print(f"Processing {len(all_pngs)} screenshots...")
    for filename in all_pngs:
        if filename in ANNOTATIONS:
            annotate_image(filename, ANNOTATIONS[filename])
        else:
            copy2(BASE / filename, OUT / filename)
            print(f"  COPY: {filename}")

    print(f"\nDone. Annotated images in: {OUT}")


if __name__ == "__main__":
    main()
