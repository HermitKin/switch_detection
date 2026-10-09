"""Check README links, media integrity, and public sample provenance."""
import hashlib
import json
import re
from pathlib import Path

from PIL import Image


def main():
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    local_links = [url for url in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", readme)
                   if not url.startswith(("https://", "http://", "#"))]
    for url in local_links:
        if not (root / url.split("#")[0]).is_file():
            raise AssertionError(f"Broken README link: {url}")
    for path in (root / "assets/showcase").iterdir():
        if path.suffix.lower() in (".gif", ".png", ".jpg"):
            with Image.open(path) as image:
                if path.suffix == ".gif":
                    if image.n_frames < 3: raise AssertionError(f"GIF is not animated: {path}")
                    for frame in range(image.n_frames):
                        image.seek(frame)
                        image.load()
                    print(f"{path.name}: {image.n_frames} frames, {image.size}")
                else: image.verify()
    payload = json.loads((root / "samples/predictions.json").read_text(encoding="utf-8"))
    for record in payload["images"]:
        sample = root / "samples" / record["image"]
        if hashlib.sha256(sample.read_bytes()).hexdigest() != record["sha256"]:
            raise AssertionError(f"Sample hash differs: {sample.name}")
    for forbidden in ("ultralytics", "resume_work", "switch_detection_complete_handoff"):
        if (root / forbidden).exists(): raise AssertionError(f"Unexpected content in clean checkout: {forbidden}")
    print(f"PUBLICATION CHECK PASSED: {len(local_links)} links, {len(payload['images'])} samples")


if __name__ == "__main__": main()
