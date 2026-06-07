#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error

FIGSHARE_ARTICLE_ID = "24235333"
API_URL = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE_ID}"


def fetch_json(url, retries=3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "MTO-downloader/0.1"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def download_file(url, dest, retries=3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "MTO-downloader/0.1"}
            )
            with urllib.request.urlopen(req, timeout=300) as resp:
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with open(dest, "wb") as f:
                    while True:
                        chunk = resp.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                return True
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(5)
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/qm9s")
    args = parser.parse_args()

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    print(f"Fetching article metadata: {API_URL}")
    article = fetch_json(API_URL)
    files = article.get("files", [])

    manifest = {
        "article_id": FIGSHARE_ARTICLE_ID,
        "title": article.get("title", ""),
        "files": [],
        "download_timestamp": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
    }

    if not files:
        print("No files found in article metadata. Saving metadata only.")
    else:
        for f in files:
            info = {
                "name": f["name"],
                "size": f["size"],
                "url": f.get("download_url", ""),
                "md5": f.get("computed_md5", ""),
                "status": "unknown",
            }
            dest = os.path.join(out_dir, f["name"])
            print(f"Downloading {f[name]} ({f[size]} bytes) -> {dest}")
            try:
                download_file(f["download_url"], dest)
                info["status"] = "success"
                info["local_path"] = dest
                print(f"  OK")
            except Exception as e:
                info["status"] = f"failed: {e}"
                print(f"  FAILED: {e}")
            manifest["files"].append(info)

    manifest_path = os.path.join(out_dir, "MANIFEST.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved: {manifest_path}")

    readme_path = os.path.join(out_dir, "README.md")
    source_url = (
        "https://figshare.com/articles/dataset/"
        f"QM9S_dataset/{FIGSHARE_ARTICLE_ID}"
    )
    readme_lines = [
        f"# QM9S Dataset",
        "",
        f"Source: {source_url}",
        "",
    ]
    for fi in manifest["files"]:
        readme_lines.append(
            f"- {fi[name]}: {fi[size]} bytes [{fi[status]}]"
        )
    with open(readme_path, "w") as f:
        f.write("\n".join(readme_lines) + "\n")
    print(f"README saved: {readme_path}")
    return manifest


if __name__ == "__main__":
    main()
