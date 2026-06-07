#!/usr/bin/env python3
"""Download QM9S dataset files."""

import argparse
import json
import os
import sys
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

FIGSHARE_ARTICLE_ID = "24235333"
API_URL = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE_ID}"


def fetch_json(url, retries=3, timeout=30):
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "MTO-downloader/0.2"})
            with urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except (HTTPError, URLError) as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def download_file(url, dest, retries=3, timeout=600):
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    for attempt in range(retries):
        try:
            print(f"  Downloading -> {dest} ...")
            req = Request(url, headers={"User-Agent": "MTO-downloader/0.2"})
            with urlopen(req, timeout=timeout) as resp:
                with open(dest, "wb") as f:
                    while True:
                        chunk = resp.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
            file_size = os.path.getsize(dest)
            print(f"  OK ({file_size} bytes)")
            return True
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            if attempt == retries - 1:
                raise
            time.sleep(5)
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/qm9s")
    parser.add_argument("--files", nargs="*", default=[],
                       help="Specific filenames to download (default: all)")
    args = parser.parse_args()

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    print(f"Fetching QM9S metadata from Figshare API...")
    try:
        article = fetch_json(API_URL)
        files = article.get("files", [])
        print(f"Found {len(files)} files")
    except Exception as e:
        print(f"ERROR: Cannot reach Figshare API: {e}")
        print("\nFallback: Please download QM9S manually from:")
        print(f"  https://figshare.com/articles/dataset/QM9S_dataset/{FIGSHARE_ARTICLE_ID}")
        print(f"\nPlace files in: {os.path.abspath(out_dir)}/")
        sys.exit(1)

    desired = set(args.files) if args.files else None

    manifest = {
        "article_id": FIGSHARE_ARTICLE_ID,
        "title": article.get("title", ""),
        "doi": article.get("doi", ""),
        "download_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": [],
    }

    for f in files:
        name = f["name"]
        if desired and name not in desired:
            continue

        info = {
            "name": name,
            "size": f["size"],
            "url": f.get("download_url", ""),
            "md5": f.get("computed_md5", ""),
            "status": "not_downloaded",
        }
        dest = os.path.join(out_dir, name)

        try:
            download_file(f["download_url"], dest)
            info["status"] = "downloaded"
            info["local_path"] = os.path.abspath(dest)
        except Exception as e:
            info["status"] = f"failed: {e}"
            print(f"  FAILED: {name} -> {e}")

        manifest["files"].append(info)

    manifest_path = os.path.join(out_dir, "MANIFEST.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\nManifest: {manifest_path}")

    # Count status
    done = sum(1 for fi in manifest["files"] if "downloaded" in str(fi.get("status", "")))
    total = len(manifest["files"])
    print(f"Downloaded: {done}/{total}")
    if done < total:
        print("Run again to retry failed files.")


if __name__ == "__main__":
    main()
