#!/usr/bin/env python
"""
draw.io ファイルを PNG にバッチエクスポートするスクリプト。

前提: draw.io デスクトップアプリがインストールされていること。
     Windows: C:\\Program Files\\draw.io\\draw.io.exe
     macOS:   /Applications/draw.io.app/Contents/MacOS/draw.io

使い方:
    python scripts/export-drawings.py
    python scripts/export-drawings.py --scale 2  # 2倍解像度
    python scripts/export-drawings.py --drawio-path "C:/path/to/draw.io.exe"
"""

import argparse
import subprocess
import sys
from pathlib import Path

# デフォルトのdraw.io実行パス（Windows/macOS対応）
DEFAULT_PATHS = [
    r"C:\Program Files\draw.io\draw.io.exe",
    r"C:\Program Files (x86)\draw.io\draw.io.exe",
    "/Applications/draw.io.app/Contents/MacOS/draw.io",
]

DRAWINGS_DIR = Path(__file__).parent.parent / "docs" / "drawings"


def find_drawio_exe(override: str | None = None) -> Path | None:
    if override:
        p = Path(override)
        return p if p.exists() else None
    for path in DEFAULT_PATHS:
        p = Path(path)
        if p.exists():
            return p
    return None


def export_drawio(drawio_exe: Path, src: Path, dst: Path, scale: float) -> bool:
    """1ファイルをPNGにエクスポートする。成功時True。"""
    cmd = [
        str(drawio_exe),
        "--export",
        "--format", "png",
        "--scale", str(scale),
        "--output", str(dst),
        str(src),
    ]
    print(f"Exporting: {src.name} -> {dst.name}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()}", file=sys.stderr)
        return False
    print(f"  OK: {dst}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Export draw.io files to PNG")
    parser.add_argument("--scale", type=float, default=2.0,
                        help="Export scale factor (default: 2.0 for Retina quality)")
    parser.add_argument("--drawio-path", default=None,
                        help="Path to draw.io executable (optional)")
    args = parser.parse_args()

    drawio_exe = find_drawio_exe(args.drawio_path)
    if drawio_exe is None:
        print("ERROR: draw.io executable not found.", file=sys.stderr)
        print("Install draw.io desktop from https://www.drawio.com/", file=sys.stderr)
        print("Or specify path with --drawio-path", file=sys.stderr)
        sys.exit(1)

    print(f"Using draw.io: {drawio_exe}")
    print(f"Scale: {args.scale}x")
    print()

    drawio_files = list(DRAWINGS_DIR.glob("*.drawio"))
    if not drawio_files:
        print(f"No .drawio files found in {DRAWINGS_DIR}")
        sys.exit(0)

    success, failed = 0, 0
    for src in sorted(drawio_files):
        dst = src.with_suffix(".png")
        if export_drawio(drawio_exe, src, dst, args.scale):
            success += 1
        else:
            failed += 1

    print()
    print(f"Done: {success} exported, {failed} failed")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
