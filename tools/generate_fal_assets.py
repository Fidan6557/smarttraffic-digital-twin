import os
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import fal_client


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "assets" / "generated"


ASSETS = [
    {
        "name": "intersection_background",
        "prompt": (
            "Top-down orthographic realistic city ground texture for a traffic simulation, "
            "subtle grass and concrete sidewalks around the edges, no cars, no people, no text, "
            "no traffic lights, clean midday lighting, game-ready 2D asset, square composition."
        ),
        "postprocess": None,
    },
    {
        "name": "vehicle_topdown_source",
        "prompt": (
            "Single modern compact car viewed perfectly from above, centered, game sprite, "
            "crisp edges, no shadow, no text, no watermark, isolated on a perfectly flat pure "
            "#00ff00 chroma-key background."
        ),
        "postprocess": "green_to_alpha",
    },
]


def require_fal_key():
    if not os.getenv("FAL_KEY"):
        raise RuntimeError(
            "FAL_KEY is not set. Set it in PowerShell with: "
            '$env:FAL_KEY="your-fal-key"'
        )


def download(url, path):
    with urllib.request.urlopen(url, timeout=90) as response:
        path.write_bytes(response.read())


def green_to_alpha(source, target):
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read generated image: {source}")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (45, 80, 80), (85, 255, 255))
    alpha = cv2.bitwise_not(mask)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    b, g, r = cv2.split(image)
    cv2.imwrite(str(target), cv2.merge([b, g, r, alpha]))


def generate_asset(asset):
    print(f"Generating {asset['name']}...")
    result = fal_client.subscribe(
        "fal-ai/flux/schnell",
        arguments={
            "prompt": asset["prompt"],
            "image_size": "square_hd",
            "num_inference_steps": 4,
            "num_images": 1,
        },
        with_logs=True,
    )

    image_url = result["images"][0]["url"]
    source_path = OUT_DIR / f"{asset['name']}.png"
    download(image_url, source_path)

    if asset["postprocess"] == "green_to_alpha":
        final_path = OUT_DIR / "vehicle_topdown.png"
        green_to_alpha(source_path, final_path)
        print(f"Saved {final_path.relative_to(ROOT)}")
    else:
        print(f"Saved {source_path.relative_to(ROOT)}")


def main():
    require_fal_key()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for asset in ASSETS:
        generate_asset(asset)
        time.sleep(0.5)
    print("Done. Restart the simulation to use the generated assets.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Asset generation failed: {exc}", file=sys.stderr)
        raise
