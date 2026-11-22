"""Try this 1: how many frames should ingestion sample per video? (make try-frame-sampling)

The extractor samples 12 evenly spaced frames from every 8-second clip (step 4).
This runs the same frame sampling, motion and ingredient code with 4 to 24
frames and reports, across the seven sample videos: the time spent, the average
motion, and how many videos get a different ingredient list or robot action
list than with 12 frames. Writes nothing.

Change FRAME_COUNTS and run it again.
"""

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ByteEats", "extractors"))
import enhanced_extractor as ex  # noqa: E402

FRAME_COUNTS = [4, 8, 12, 24]
BASELINE = 12


def analyse(path, n):
    frames, timestamps, _ = ex.extract_frames(path, num_frames=n)
    middle = len(frames) // 2
    colors = ex.analyze_video_color(frames[middle])
    edges = ex.detect_edges(frames[middle])["edge_density"]
    scores, regions = ex.detect_motion(frames, timestamps)
    avg_motion = sum(s["motion_score"] for s in scores) / len(scores)
    ingredients, _ = ex.detect_food_ingredients(colors, edges)
    video_type, stage = ex.identify_video_type(path)
    actions = ex.infer_robot_actions(video_type, stage, ingredients, scores, regions)
    return avg_motion, ingredients, actions


def main():
    videos = sorted(os.path.join(ex.VIDEO_DIR, f) for f in os.listdir(ex.VIDEO_DIR) if f.endswith(".mp4"))
    results = {}
    for n in FRAME_COUNTS:
        start = time.perf_counter()
        results[n] = {os.path.basename(v): analyse(v, n) for v in videos}
        results[n]["_time"] = time.perf_counter() - start

    base = results[BASELINE]
    print(f"{len(videos)} videos, 8 s each. Compared with {BASELINE} frames (what the extractor uses):\n")
    print(f"{'frames':>6}  {'time':>6}  {'average motion':>14}  {'ingredients differ':>18}  {'actions differ':>14}")
    for n in FRAME_COUNTS:
        r = results[n]
        names = [k for k in r if k != "_time"]
        motion = sum(r[k][0] for k in names) / len(names)
        ingr = sum(r[k][1] != base[k][1] for k in names)
        acts = sum(r[k][2] != base[k][2] for k in names)
        print(f"{n:>6}  {r['_time']:>5.1f}s  {motion:>13.2f}%  {ingr:>11} of {len(names)}  {acts:>7} of {len(names)}")
    print("\nMotion is the share of pixels that change between two consecutive sampled frames.")


if __name__ == "__main__":
    main()
