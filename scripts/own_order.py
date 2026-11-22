"""Run the pipeline on your own order: make check-data DIR=... and make load DIR=...

Expected folder layout:

    my_order/
      order.json            the order, as you would POST it to /api/orders, plus "order_id"
      videos/               .mp4 clips of the order being made. The name says the stage:
                            salmon-prep-<n>, assembly, quality-check, packaging or delivery
      reference_images/     optional: <menu item>.jpg, for example salmon_poke_bowl.jpg.
                            Without it, ByteEats/data/reference_images/ is used.

--check-only reads every file and writes nothing. Without it, once every check
passes, it extracts the video features (step 4), runs the quality check (step 6),
builds the embeddings (step 7) and compares them with the other orders already
in video_embeddings/ (the sample order, after `make run`).
"""

import argparse
import contextlib
import importlib
import json
import logging
import os
import re
import sys
import tempfile
from datetime import datetime

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")  # unreadable videos are reported below
import cv2  # noqa: E402
from pydantic import ValidationError  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("api", "extractors", "analysis"):
    sys.path.insert(0, os.path.join(ROOT, "ByteEats", sub))
SAMPLE_REFERENCES = os.path.join(ROOT, "ByteEats", "data", "reference_images")
SAMPLE_ORDER = "ORD1001"
ORDER_ID = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
IMAGE_TYPES = (".jpg", ".jpeg", ".png")  # what the quality checker looks for
MIN_SIDE = 32
STAGES = "salmon-prep-<n>, assembly, quality-check, packaging or delivery"


def quiet_import(name):
    """Import a pipeline module without letting it create its output folders here."""
    if name in sys.modules:
        return sys.modules[name]
    here = os.getcwd()
    with tempfile.TemporaryDirectory() as scratch:
        os.chdir(scratch)
        try:
            with contextlib.redirect_stdout(open(os.devnull, "w")):
                return importlib.import_module(name)
        finally:
            os.chdir(here)


def is_lfs_pointer(path):
    with open(path, "rb") as f:
        return f.read(100).startswith(b"version https://git-lfs")


def check_order(folder, problems):
    path = os.path.join(folder, "order.json")
    if not os.path.isfile(path):
        problems.append("order.json: missing")
        return None
    try:
        with open(path, encoding="utf-8") as f:
            order = json.load(f)
    except UnicodeDecodeError:
        problems.append("order.json: not UTF-8 text")
        return None
    except json.JSONDecodeError as e:
        problems.append(f"order.json: not valid JSON (line {e.lineno}: {e.msg})")
        return None
    if not isinstance(order, dict):
        problems.append("order.json: must be a JSON object")
        return None

    order_api = quiet_import("order_api")
    try:
        order_api.OrderInput.model_validate(order)
    except ValidationError as e:
        for err in e.errors():
            where = ".".join(str(p) for p in err["loc"])
            problems.append(f"order.json: {where}: {err['msg']}")
    order_id = order.get("order_id")
    if not isinstance(order_id, str) or not ORDER_ID.match(order_id):
        problems.append("order.json: order_id: needs 1-40 letters, digits, - or _ (it becomes a file name)")
    elif order_id == SAMPLE_ORDER:
        problems.append(f"order.json: order_id: {SAMPLE_ORDER} is the sample order; pick another ID")
    for i, item in enumerate(order.get("items") or []):
        if isinstance(item, dict) and isinstance(item.get("quantity"), int) and item["quantity"] < 1:
            problems.append(f"order.json: items.{i}.quantity: must be 1 or more, got {item['quantity']}")
    if order.get("items") == []:
        problems.append("order.json: items: empty")
    if isinstance(order.get("menu_item"), str) and not order["menu_item"].strip():
        problems.append("order.json: menu_item: empty")
    if "order_placed_at" in order:
        try:
            datetime.fromisoformat(str(order["order_placed_at"]))
        except ValueError:
            problems.append(f"order.json: order_placed_at: not an ISO 8601 time: {order['order_placed_at']!r}")
    return order


def check_videos(folder, problems):
    video_dir = os.path.join(folder, "videos")
    if not os.path.isdir(video_dir):
        problems.append("videos/: missing")
        return []
    extractor = quiet_import("enhanced_extractor")
    stages = []
    for name in sorted(os.listdir(video_dir)):
        if name.startswith("."):
            continue
        rel, path = f"videos/{name}", os.path.join(video_dir, name)
        if not os.path.isfile(path):
            problems.append(f"{rel}: a folder; videos/ must hold files only")
            continue
        if not name.lower().endswith(".mp4"):
            problems.append(f"{rel}: not an .mp4 file (the extractor reads only .mp4)")
            continue
        if os.path.getsize(path) == 0:
            problems.append(f"{rel}: empty file")
            continue
        if is_lfs_pointer(path):
            problems.append(f"{rel}: a Git LFS pointer, not a video (run git lfs pull)")
            continue
        with open(path, "rb") as f:
            if f.read(12)[4:8] != b"ftyp":
                problems.append(f"{rel}: not an MP4 video (no MP4 header)")
                continue
        cap = cv2.VideoCapture(path)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
        fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0
        ok = cap.isOpened() and cap.read()[0]
        cap.release()
        if not ok or frames < 2:
            problems.append(f"{rel}: no readable video frames (needs at least 2)")
            continue
        if fps <= 0:
            problems.append(f"{rel}: no frame rate in the file")
            continue
        video_type, _ = extractor.identify_video_type(path)
        if video_type == "unknown":
            problems.append(f"{rel}: the name doesn't say the stage ({STAGES})")
            continue
        stages.append(video_type)
    if not stages and not any(p.startswith("videos/") for p in problems):
        problems.append("videos/: no .mp4 files")
    elif stages and not {"packaging", "assembly"} & set(stages):
        problems.append("videos/: needs a packaging or assembly clip; the quality check uses its last frame")
    return stages


def find_reference(folder, menu_item):
    """The quality checker's lookup: <menu item, lower case, spaces as _>.jpg/.jpeg/.png"""
    safe = menu_item.lower().replace(" ", "_")
    for ref_dir in (os.path.join(folder, "reference_images"), SAMPLE_REFERENCES):
        for ext in IMAGE_TYPES:
            if os.path.isfile(os.path.join(ref_dir, safe + ext)):
                return ref_dir, os.path.join(ref_dir, safe + ext)
    return None, None


def check_reference(folder, order, problems):
    ref_dir = os.path.join(folder, "reference_images")
    if os.path.isdir(ref_dir):
        for name in sorted(os.listdir(ref_dir)):
            path = os.path.join(ref_dir, name)
            if name.startswith(".") or not os.path.isfile(path):
                continue
            rel = f"reference_images/{name}"
            if not name.lower().endswith(IMAGE_TYPES):
                problems.append(f"{rel}: not .jpg, .jpeg or .png")
            elif is_lfs_pointer(path):
                problems.append(f"{rel}: a Git LFS pointer, not an image (run git lfs pull)")
            elif (img := cv2.imread(path)) is None:
                problems.append(f"{rel}: can't open as an image")
            elif min(img.shape[:2]) < MIN_SIDE:
                problems.append(f"{rel}: too small ({img.shape[1]}x{img.shape[0]}), "
                                f"needs at least {MIN_SIDE}x{MIN_SIDE}")
    menu_item = (order or {}).get("menu_item")
    if isinstance(menu_item, str) and menu_item.strip():
        found_dir, _ = find_reference(folder, menu_item)
        if found_dir is None:
            safe = menu_item.lower().replace(" ", "_")
            problems.append(f"reference_images/: no {safe}.jpg (or .jpeg, .png) for menu item {menu_item!r}")
        return found_dir
    return None


def scan(folder):
    """Check a folder. Returns (order, video stages, reference dir, problems). Writes nothing."""
    problems = []
    if not os.path.isdir(folder):
        return None, [], None, [f"{folder}: not a folder"]
    order = check_order(folder, problems)
    stages = check_videos(folder, problems)
    ref_dir = check_reference(folder, order, problems)
    return order, stages, ref_dir, problems


def load(folder, order, ref_dir):
    logging.disable(logging.INFO)  # the steps log every action; this prints a summary instead
    order_id, menu_item = order["order_id"], order["menu_item"]
    features_dir = os.path.join("video_features_enhanced", order_id)
    os.makedirs(features_dir, exist_ok=True)

    extractor = quiet_import("enhanced_extractor")
    extractor.VIDEO_DIR = os.path.join(folder, "videos")
    extractor.OUTPUT_DIR, extractor.ORDER_ID, extractor.MENU_ITEM = features_dir, order_id, menu_item
    log = os.path.join(features_dir, "extract.log")
    with open(log, "w") as f, contextlib.redirect_stdout(f):
        extractor.process_all_videos()
    with open(os.path.join(features_dir, f"{order_id}_combined_features.json")) as f:
        videos = json.load(f)["features"]
    print(f"Extracted features from {len(videos)} videos into {features_dir}/ (log: {log})")
    for v in videos:
        print(f"  {os.path.basename(v['video_path']):28} {v['video_type']:9} motion "
              f"{v['motion_analysis']['average_motion']:5.2f}%  "
              f"ingredients: {', '.join(v['detected_ingredients']['ingredients']) or '-'}")

    checker = quiet_import("quality_checker")
    report = checker.ImageQualityChecker(ref_dir, features_dir, "qa_results").check_quality(order_id, menu_item)
    if "error" in report:
        sys.exit(f"Quality check: {report['error']}")
    print(f"Quality check: {'PASSED' if report['quality_check_passed'] else 'FAILED'}, overall score "
          f"{report['similarity_metrics']['overall_score']:.2f}% (pass mark 70%), "
          f"report qa_results/{order_id}_quality_report.json")

    embeddings = quiet_import("embedding_generator")
    gen = embeddings.VideoEmbeddingGenerator(features_dir, "video_embeddings", "embedding_models")
    result = gen.process_video_features(order_id)
    if not result.get("success"):
        sys.exit(f"Embeddings: {result.get('error')}")
    print(f"Embeddings: {result['embeddings_count']} vectors in {result['embeddings_file']}")
    others = [f for f in os.listdir("video_embeddings")
              if f.endswith("_embeddings.pkl") and not f.startswith(order_id)]
    if not others:
        print("No other orders in video_embeddings/ to compare with. Run `make run` first.")
        return
    # A threshold above 1 returns every video with its score; the pipeline's default is 0.8
    scores = gen.detect_anomalies(order_id, threshold=1.01)["anomalies"]
    print(f"Compared with the same stage in {len(others)} other order(s) "
          f"(embedding_generator.py anomalies, threshold 0.8):")
    for a in scores:
        flag = "ANOMALY" if a["similarity_score"] < 0.8 else "ok"
        print(f"  {a['video_path']:28} {a['video_type']:9} similarity {a['similarity_score']:.3f}  {flag}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dir", required=True, help="folder with order.json and videos/")
    parser.add_argument("--check-only", action="store_true", help="check the files and write nothing")
    args = parser.parse_args()
    folder = os.path.abspath(args.dir)
    os.chdir(ROOT)  # outputs go where the pipeline writes them

    order, stages, ref_dir, problems = scan(folder)
    if problems:
        print(f"{len(problems)} problem(s) in {args.dir}; nothing was written:")
        for p in problems:
            print(f"  {p}")
        sys.exit(1)
    where = (os.path.join(args.dir, "reference_images") if ref_dir.startswith(folder)
             else os.path.relpath(ref_dir, ROOT))
    print(f"{args.dir}: order {order['order_id']} ({order['menu_item']}), {len(stages)} videos "
          f"({', '.join(sorted(set(stages)))}), reference image from {where}/. All checks passed.")
    if not args.check_only:
        load(folder, order, ref_dir)


if __name__ == "__main__":
    main()
