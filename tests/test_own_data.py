"""Checks for the "Use your own data" adapter (scripts/own_order.py). CPU only, no network."""

import json
import os
import shutil
import sys

import cv2
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from own_order import scan  # noqa: E402

VIDEOS = os.path.join(ROOT, "ByteEats", "data", "sample-videos")
ORDER = {"order_id": "ORD2001", "restaurant_id": "PH001", "restaurant_name": "Sample Poke Bar",
         "menu_item": "Salmon Poke Bowl", "items": [{"name": "Salmon", "quantity": 1}]}


def make_order(folder, order=ORDER, videos=("delivery-packaging-7.mp4",)):
    os.makedirs(os.path.join(folder, "videos"), exist_ok=True)
    with open(os.path.join(folder, "order.json"), "w", encoding="utf-8") as f:
        json.dump(order, f)
    for name in videos:
        shutil.copy(os.path.join(VIDEOS, name), os.path.join(folder, "videos", name))
    return str(folder)


def write(folder, rel, data):
    path = os.path.join(folder, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def test_good_folder_passes_and_writes_nothing(tmp_path):
    folder = make_order(tmp_path, videos=("salmon-prep-1.mp4", "delivery-packaging-7.mp4"))
    write(folder, "videos/.DS_Store", b"\0")
    before = set(os.listdir(ROOT))
    order, stages, ref_dir, problems = scan(folder)
    assert problems == []
    assert order["order_id"] == "ORD2001"
    assert sorted(stages) == ["packaging", "prep"]
    assert ref_dir.endswith(os.path.join("ByteEats", "data", "reference_images"))
    assert set(os.listdir(ROOT)) == before


def test_own_reference_image_is_used(tmp_path):
    folder = make_order(tmp_path, {**ORDER, "menu_item": "Miso Salmon Bowl"})
    _, _, _, problems = scan(folder)
    assert problems == ["reference_images/: no miso_salmon_bowl.jpg (or .jpeg, .png) for menu item 'Miso Salmon Bowl'"]
    os.makedirs(os.path.join(folder, "reference_images"))
    cv2.imwrite(os.path.join(folder, "reference_images", "miso_salmon_bowl.png"), np.zeros((64, 64, 3), np.uint8))
    _, _, ref_dir, problems = scan(folder)
    assert problems == [] and ref_dir == os.path.join(folder, "reference_images")


@pytest.mark.parametrize("rel,data,expected", [
    ("videos/salmon-prep-3.mp4", b"version https://git-lfs.github.com/spec/v1\n", "Git LFS pointer"),
    ("videos/quality-check.mp4", b"not a video", "not an MP4 video"),
    ("videos/assembly.mp4", b"", "empty file"),
    ("videos/assembly.mov", b"\0\0\0\x18ftypqt  ", "not an .mp4 file"),
    ("videos/assembly.mp4", b"\0\0\0\x18ftypisom" + b"\0" * 100, "no readable video frames"),
    ("reference_images/salmon_poke_bowl.jpg", b"not an image", "can't open as an image"),
    ("reference_images/notes.txt", b"hello", "not .jpg, .jpeg or .png"),
])
def test_bad_files_are_reported(tmp_path, rel, data, expected):
    folder = make_order(tmp_path)
    write(folder, rel, data)
    _, _, _, problems = scan(folder)
    assert len(problems) == 1 and problems[0].startswith(rel) and expected in problems[0], problems


def test_video_name_must_say_the_stage(tmp_path):
    folder = make_order(tmp_path)
    shutil.copy(os.path.join(VIDEOS, "salmon-prep-1.mp4"), os.path.join(folder, "videos", "kitchen-cam.mp4"))
    _, _, _, problems = scan(folder)
    assert len(problems) == 1 and "the name doesn't say the stage" in problems[0]


def test_needs_a_packaging_or_assembly_clip(tmp_path):
    folder = make_order(tmp_path, videos=("salmon-prep-1.mp4",))
    _, _, _, problems = scan(folder)
    assert problems == ["videos/: needs a packaging or assembly clip; the quality check uses its last frame"]


def test_tiny_reference_image_is_reported(tmp_path):
    folder = make_order(tmp_path)
    os.makedirs(os.path.join(folder, "reference_images"))
    cv2.imwrite(os.path.join(folder, "reference_images", "salmon_poke_bowl.jpg"), np.zeros((16, 16, 3), np.uint8))
    _, _, _, problems = scan(folder)
    assert problems == ["reference_images/salmon_poke_bowl.jpg: too small (16x16), needs at least 32x32"]


@pytest.mark.parametrize("change,expected", [
    ({"order_id": "ORD 1"}, "order_id: needs 1-40 letters"),
    ({"order_id": "ORD1001"}, "ORD1001 is the sample order"),
    ({"items": [{"name": "Salmon", "quantity": 0}]}, "items.0.quantity: must be 1 or more"),
    ({"items": []}, "items: empty"),
    ({"menu_item": " "}, "menu_item: empty"),
    ({"order_placed_at": "yesterday"}, "order_placed_at: not an ISO 8601 time"),
    ({"restaurant_id": None}, "restaurant_id: Input should be a valid string"),
])
def test_bad_orders_are_reported(tmp_path, change, expected):
    folder = make_order(tmp_path, {**ORDER, **change})
    _, _, _, problems = scan(folder)
    assert any(expected in p for p in problems), problems


def test_unreadable_order_files(tmp_path):
    folder = make_order(tmp_path)
    write(folder, "order.json", b"{not json")
    assert scan(folder)[3][0].startswith("order.json: not valid JSON")
    write(folder, "order.json", "{\"menu_item\": \"caf\xe9\"}".encode("latin-1"))
    assert scan(folder)[3][0] == "order.json: not UTF-8 text"
    os.remove(os.path.join(folder, "order.json"))
    assert scan(folder)[3][0] == "order.json: missing"
