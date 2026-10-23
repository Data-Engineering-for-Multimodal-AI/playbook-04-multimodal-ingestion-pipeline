"""Smoke tests: sample data, the API, every pipeline step, and run_pipeline.py.

Run with `make smoke` (or `uv run pytest -q`). CPU only, no network, about a minute.
"""

import json
import os
import py_compile
import shutil
import socket
import subprocess
import sys

import cv2
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "ByteEats", "data")
VIDEOS = sorted(os.listdir(os.path.join(DATA, "sample-videos")))

WRAPPERS = {
    "byteats_order_api.py": "ByteEats/api/order_api.py",
    "robot_trigger_service.py": "ByteEats/services/robot_trigger.py",
    "pickup_notification_service.py": "ByteEats/services/notification.py",
    "video_feature_extractor.py": "ByteEats/extractors/video_extractor.py",
    "enhanced_video_extractor.py": "ByteEats/extractors/enhanced_extractor.py",
    "image_quality_checker.py": "ByteEats/analysis/quality_checker.py",
    "video_embedding_generator.py": "ByteEats/analysis/embedding_generator.py",
    "image_similarity_demo.py": "ByteEats/analysis/similarity_demo.py",
}


def run(script, *args, cwd):
    env = dict(os.environ, MPLBACKEND="Agg")
    return subprocess.run([sys.executable, os.path.join(ROOT, script), *args], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=300)


# ------------------------------------------------------------------ data and code

def test_sample_data_complete_and_documented():
    with open(os.path.join(DATA, "README.md")) as fh:
        readme = fh.read()
    assert len(VIDEOS) == 7
    for name in VIDEOS:
        cap = cv2.VideoCapture(os.path.join(DATA, "sample-videos", name))
        assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0, name
        cap.release()
        assert name in readme
    for folder in ("reference_images", "sample_bowls"):
        for name in os.listdir(os.path.join(DATA, folder)):
            assert cv2.imread(os.path.join(DATA, folder, name)) is not None, name
            assert name in readme


def test_every_script_compiles():
    # video_extractor.py and advanced_video_extractor.py were a SyntaxError in the first edition
    scripts = [os.path.join(dp, f) for dp, _, fs in os.walk(os.path.join(ROOT, "ByteEats")) for f in fs
               if f.endswith(".py")]
    scripts += [os.path.join(ROOT, f) for f in os.listdir(ROOT) if f.endswith(".py")]
    for path in scripts:
        py_compile.compile(path, doraise=True)


def test_root_wrappers_point_at_their_module():
    for wrapper, target in WRAPPERS.items():
        with open(os.path.join(ROOT, wrapper)) as fh:
            source = fh.read()
        assert ", ".join(repr(p) for p in target.split("/")) in source, wrapper
        assert os.path.exists(os.path.join(ROOT, target))


# ------------------------------------------------------------------ order API

def test_order_api(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.chdir(tmp_path)
    sys.path.insert(0, os.path.join(ROOT, "ByteEats", "api"))
    import order_api
    client = TestClient(order_api.app)

    assert client.post("/api/process-sample-order").json()["order_id"] == "ORD1001"
    assert client.get("/api/orders/ORD1001").json()["menu_item"] == "Salmon Poke Bowl"

    created = client.post("/api/orders", json={
        "restaurant_id": "PH001", "restaurant_name": "Poke House", "menu_item": "Miso Salmon Bowl",
        "items": [{"name": "Salmon", "quantity": 1}]}).json()
    saved = client.get(f"/api/orders/{created['order_id']}").json()
    assert saved["order_status"] == "received" and saved["items"][0]["name"] == "Salmon"


# ------------------------------------------------------------------ pipeline steps

@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """The README's steps 2-7, run one by one in an empty folder, after the sample order."""
    work = tmp_path_factory.mktemp("steps")
    from fastapi.testclient import TestClient
    sys.path.insert(0, os.path.join(ROOT, "ByteEats", "api"))
    import order_api
    cwd = os.getcwd()
    os.chdir(work)
    try:
        TestClient(order_api.app).post("/api/process-sample-order")
    finally:
        os.chdir(cwd)

    results = {
        "robot": run("ByteEats/services/robot_trigger.py", "--once", cwd=work),
        "extract": run("ByteEats/extractors/enhanced_extractor.py", cwd=work),
        "notify": run("ByteEats/services/notification.py", "--once", cwd=work),
        "qa": run("ByteEats/analysis/quality_checker.py", "ORD1001", "Salmon Poke Bowl", cwd=work),
        "embed": run("ByteEats/analysis/embedding_generator.py", "generate", "ORD1001", cwd=work),
        "similar": run("ByteEats/analysis/similarity_demo.py", cwd=work),
    }
    for name, result in results.items():
        assert result.returncode == 0, f"{name}:\n{result.stdout}\n{result.stderr}"
    return work, results


def load(path):
    with open(path) as fh:
        return json.load(fh)


def test_robot_and_notification(pipeline):
    work, _ = pipeline
    assert not os.listdir(work / "orders")                     # --once processed the order
    assert load(work / "robot_logs" / "ORD1001_robot_log.json")["status"] == "completed"
    assert load(work / "processed_orders" / "ORD1001.json")["order_status"] == "ready_for_pickup"
    assert load(work / "notification_logs" / "ORD1001_notification.json")["customer_id"] == "CUST7890"


def test_video_features(pipeline):
    work, _ = pipeline
    combined = load(work / "video_features_enhanced" / "ORD1001_combined_features.json")
    assert combined["videos_processed"] == 7
    types = sorted(f["video_type"] for f in combined["features"])
    assert types == ["assembly", "packaging", "prep", "prep", "prep", "prep", "quality"]


def test_quality_check(pipeline):
    work, results = pipeline
    report = load(work / "qa_results" / "ORD1001_quality_report.json")
    # The packaging frame is a side view; the reference is a top-down photo, so the check fails
    assert report["quality_check_passed"] is False
    assert round(report["similarity_metrics"]["overall_score"], 2) == 16.13
    assert "Overall score: 16.13%" in results["qa"].stderr
    assert (work / "qa_results" / "ORD1001_comparison.jpg").exists()


def test_quality_scores_identical_images_100():
    # SSIM was computed on 8-bit pixels in the first edition and overflowed past 1.0
    sys.path.insert(0, os.path.join(ROOT, "ByteEats", "analysis"))
    import quality_checker
    checker = quality_checker.ImageQualityChecker("unused", "unused", "unused")
    ref = cv2.imread(os.path.join(DATA, "reference_images", "salmon_poke_bowl.jpg"))
    same = checker.calculate_image_similarity(ref, ref.copy())
    assert round(same["ssim"], 6) == 100 and round(same["overall_score"], 6) == 100
    other = checker.calculate_image_similarity(ref, cv2.imread(os.path.join(DATA, "sample_bowls", "miso_salmon_bowl.jpg")))
    assert -100 <= other["ssim"] <= 100


def test_embeddings(pipeline):
    work, results = pipeline
    assert json.loads(results["embed"].stdout[results["embed"].stdout.index("{"):])["embeddings_count"] == 7
    assert (work / "video_embeddings" / "ORD1001_embeddings_viz.png").exists()   # t-SNE plot
    anomalies = run("ByteEats/analysis/embedding_generator.py", "anomalies", "ORD1001", cwd=work)
    assert anomalies.returncode == 0 and '"is_anomalous": false' in anomalies.stdout
    missing = run("ByteEats/analysis/embedding_generator.py", "generate", "NOSUCHORDER", cwd=work)
    assert missing.returncode == 1


def test_similarity(pipeline):
    work, results = pipeline
    out = results["similar"].stdout
    assert "Processing 7 images..." in out
    assert "2. miso salmon bowl (similarity: 0.7184)" in out
    for name in ("similarity_matrix_opencv.png", "embeddings_visualization_opencv.png",
                 "similar_to_reference_opencv.png", "bowl_embeddings_opencv.pkl"):
        assert (work / "similarity_results" / name).exists()


def test_quality_check_fails_without_features(tmp_path):
    result = run("ByteEats/analysis/quality_checker.py", "ORD1001", "Salmon Poke Bowl", cwd=tmp_path)
    assert result.returncode == 1


def test_root_wrapper_runs_the_module(tmp_path):
    result = run("image_quality_checker.py", cwd=tmp_path)
    assert result.returncode == 1 and "Usage: python image_quality_checker.py" in result.stdout


# ------------------------------------------------------------------ run_pipeline.py

def port_in_use(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


@pytest.mark.skipif(port_in_use(8000), reason="something is already listening on port 8000")
def test_run_pipeline_end_to_end(tmp_path):
    # run_pipeline.py writes to the repository root, so run it on a copy
    copy = tmp_path / "repo"
    shutil.copytree(ROOT, copy, ignore=shutil.ignore_patterns(".git", ".venv", "tests", "__pycache__"))
    result = subprocess.run([sys.executable, str(copy / "ByteEats" / "run_pipeline.py")],
                            capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Successfully completed 7/7 steps." in result.stdout
    assert not port_in_use(8000)          # the API it started was stopped
