"""Shrink every thumbnail in data/images/ to the current size limits.

Run once after upgrading to v0.10.3 to reclaim space from old full-resolution
thumbnails. Saves originals to data/images-backup-<timestamp>/ first.

    .testenv/bin/python tools/shrink_thumbnails.py            # in-place
    .testenv/bin/python tools/shrink_thumbnails.py --dry-run  # report only
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import __init__ as plib  # noqa: E402


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="report savings, don't write")
    args = ap.parse_args()

    images_dir = plib.IMAGES_DIR
    if not images_dir.exists():
        print(f"no images dir at {images_dir}")
        return 0

    files = sorted(p for p in images_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in plib._ALLOWED_IMAGE_EXT)
    if not files:
        print("no thumbnails to shrink")
        return 0

    before_total = sum(p.stat().st_size for p in files)
    print(f"found {len(files)} thumbnails, {human(before_total)} total")
    print(f"target: max edge {plib._THUMBNAIL_MAX_EDGE}px, JPEG q={plib._THUMBNAIL_JPEG_QUALITY}")

    if not args.dry_run:
        backup = images_dir.parent / f"images-backup-{int(time.time())}"
        shutil.copytree(images_dir, backup)
        print(f"backed up originals to {backup}")

    after_total = 0
    failed = 0
    for p in files:
        pid = p.stem
        before = p.stat().st_size
        try:
            data = p.read_bytes()
        except OSError as e:
            print(f"  ! {pid}: read failed: {e}")
            failed += 1
            after_total += before
            continue

        if args.dry_run:
            from io import BytesIO
            from PIL import Image
            try:
                img = Image.open(BytesIO(data))
                img.thumbnail((plib._THUMBNAIL_MAX_EDGE, plib._THUMBNAIL_MAX_EDGE))
                buf = BytesIO()
                if img.mode in ("RGBA", "LA"):
                    img.convert("RGBA").save(buf, format="PNG", optimize=True)
                else:
                    img.convert("RGB").save(buf, format="JPEG",
                                            quality=plib._THUMBNAIL_JPEG_QUALITY,
                                            optimize=True, progressive=True)
                est = len(buf.getvalue())
            except Exception as e:
                print(f"  ! {pid}: {e}")
                failed += 1
                est = before
            after_total += est
            continue

        ext = plib._save_thumbnail_bytes(pid, data)
        if ext is None:
            failed += 1
            after_total += before
            continue
        new_path = images_dir / f"{pid}{ext}"
        after_total += new_path.stat().st_size if new_path.exists() else before

    saved = before_total - after_total
    pct = (100 * saved / before_total) if before_total else 0
    print()
    print(f"before: {human(before_total)}")
    print(f"after:  {human(after_total)}")
    print(f"saved:  {human(saved)} ({pct:.0f}%)")
    if failed:
        print(f"{failed} thumbnails failed to process")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
