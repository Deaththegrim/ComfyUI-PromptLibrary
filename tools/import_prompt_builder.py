#!/usr/bin/env python3
"""Import a Prompt Builder zip into a running GrimmRibbity instance.

Usage:
    python tools/import_prompt_builder.py "/path/to/Prompts Builder - v1.1.zip"
    python tools/import_prompt_builder.py path/to/zip --host 127.0.0.1 --port 8188

The zip is uploaded to POST /prompt_library/import_tag_packs on the running
ComfyUI server. Imports are additive: re-running creates duplicates with
suffixed ids (legs_up, legs_up_2, ...). Bulk-delete the prompt-builder tag
first if you want a clean re-import.
"""
import argparse
import json
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def _multipart_body(filename: str, data: bytes) -> tuple[bytes, str]:
    boundary = f"----GrimmRibbity{uuid.uuid4().hex}"
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: application/zip\r\n\r\n"
    ).encode("utf-8")
    tail = f"\r\n--{boundary}--\r\n".encode("utf-8")
    return head + data + tail, boundary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("zip_path", type=Path, help="Path to the Prompt Builder zip")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8188)
    args = ap.parse_args()

    if not args.zip_path.is_file():
        print(f"error: not a file: {args.zip_path}", file=sys.stderr)
        return 2

    data = args.zip_path.read_bytes()
    body, boundary = _multipart_body(args.zip_path.name, data)
    url = f"http://{args.host}:{args.port}/prompt_library/import_tag_packs"
    req = urllib.request.Request(url, data=body, method="POST", headers={
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(body)),
    })
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        print(f"error: could not reach {url}: {e}", file=sys.stderr)
        return 1

    print(f"added:   {payload.get('added', 0)}")
    print(f"updated: {payload.get('updated', 0)}")
    errors = payload.get("errors") or []
    if errors:
        print(f"errors:  {len(errors)}")
        for e in errors[:20]:
            print(f"  - {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
