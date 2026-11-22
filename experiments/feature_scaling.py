"""Try this 2: does scaling the video features change which videos look alike? (make try-feature-scaling)

The embedding step (step 7) builds one vector per video from features on very
different scales: color and motion percentages (0-100), edge densities, and
0/1 flags for the video type and the ingredients. It compares the raw values
with cosine similarity. This compares the same vectors raw and after
standardizing each feature (mean 0, standard deviation 1 across the videos),
and reports the similarity range and each video's nearest neighbor.

Needs the extracted features: run `make run` first. Writes nothing.
Change scale() (for example to min-max scaling) and run it again.
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ByteEats", "analysis"))
# The order extract_features_vector() builds them in
NAMES = (["salmon red %", "salmon orange %", "seaweed green %", "sauce brown %", "rice white %",
          "container blue %", "yellow %", "edge mean %", "edge max %", "edge min %", "average motion %",
          "motion mean %", "motion max %", "motion std"]
         + ["type " + t for t in ("prep", "assembly", "quality", "packaging")] + ["stage / 10"]
         + ["has " + i for i in ("salmon", "rice", "seaweed", "sauce", "container")])
FEATURES = os.path.join(ROOT, "video_features_enhanced", "ORD1001_combined_features.json")


def scale(x):
    std = x.std(axis=0)
    std[std == 0] = 1
    return (x - x.mean(axis=0)) / std


def cosine(x):
    unit = x / np.linalg.norm(x, axis=1, keepdims=True)
    return unit @ unit.T


def main():
    if not os.path.exists(FEATURES):
        sys.exit(f"{FEATURES} not found. Run `make run` first.")
    from embedding_generator import VideoEmbeddingGenerator  # prints a TensorFlow note

    with open(FEATURES) as f:
        videos = json.load(f)["features"]
    gen = VideoEmbeddingGenerator.__new__(VideoEmbeddingGenerator)
    raw = np.stack([gen.extract_features_vector(v) for v in videos]).astype(float)
    names = [os.path.basename(v["video_path"]).rsplit(".", 1)[0] for v in videos]
    print(f"{len(names)} videos, {raw.shape[1]} features each.\n")

    sims = {"raw (as the pipeline does)": cosine(raw), "standardized": cosine(scale(raw))}
    off = ~np.eye(len(names), dtype=bool)
    print(f"{'features':28} {'lowest':>7} {'highest':>8} {'pairs above 0.8':>16}")
    for label, s in sims.items():
        v = s[off]
        print(f"{label:28} {v.min():>7.3f} {v.max():>8.3f} {int((v > 0.8).sum() / 2):>9} of {int(off.sum() / 2)}")

    print(f"\n{'video':24} {'nearest, raw':24} {'nearest, standardized':24}")
    for i, name in enumerate(names):
        near = [names[int(np.argmax(np.where(off[i], s[i], -2)))] for s in sims.values()]
        print(f"{name:24} {near[0]:24} {near[1]:24}")

    top = np.argsort(-raw.mean(axis=0))[:3]
    print(f"\nLargest raw features (mean over videos): "
          + ", ".join(f"{NAMES[j]} {raw[:, j].mean():.1f}" for j in top)
          + "; the type and ingredient flags are 0 or 1.")


if __name__ == "__main__":
    main()
