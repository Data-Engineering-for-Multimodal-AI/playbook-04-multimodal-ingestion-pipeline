# ByteEats Core Ops Pipeline

**Chapter 4 playbook** from *Data Engineering for Multimodal AI* (O'Reilly, 2026). Page numbers to be confirmed.
Book: https://oreil.ly/data-engineering-multimodal

> Want the code exactly as printed? Use the `first-edition` tag. This branch is the 2026 maintenance update; see [Differences from the printed book](#differences-from-the-printed-book).

## What you'll build

ByteEats is an end-to-end ELT (Extract, Load, Transform) pipeline for automated food preparation. The system processes videos of food preparation, extracts meaningful features, generates vector embeddings for similarity search and anomaly detection, and provides quality assurance through image comparison.

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Order     │    │   Robot     │    │   Video     │    │   Quality   │
│  Ingestion  │─→  │ Preparation │─→  │  Feature    │─→  │  Assurance  │
│             │    │             │    │ Extraction  │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                               │
                                                               ▼
                                      ┌─────────────┐    ┌─────────────┐
                                      │ Similarity  │    │   Vector    │
                                      │   Search    │←─  │ Embeddings  │
                                      │             │    │             │
                                      └─────────────┘    └─────────────┘
```

The pipeline has six stages, plus a pickup notification once the order is packaged:

1. **Order Ingestion** (`ByteEats/api/order_api.py`): a FastAPI service that receives orders and saves them as JSON.
2. **Robot Preparation** (`ByteEats/services/robot_trigger.py`): picks up new orders and simulates the robot's preparation sequence, with a timestamped log.
3. **Video Feature Extraction** (`ByteEats/extractors/enhanced_extractor.py`): samples frames from the seven preparation videos with OpenCV and records colors, edges, motion, likely ingredients and robot actions.
4. **Quality Assurance** (`ByteEats/analysis/quality_checker.py`): compares the last packaging frame with a reference image (MSE, SSIM, color histograms, ORB keypoints).
5. **Vector Embeddings** (`ByteEats/analysis/embedding_generator.py`): turns each video's features into a vector, then finds similar videos and anomalies.
6. **Similarity Search** (`ByteEats/analysis/similarity_demo.py`): ranks bowl images by visual similarity to the reference.
7. **Pickup Notification** (`ByteEats/services/notification.py`): marks the order ready for pickup once the packaging video shows it's done.

Everything runs on a laptop CPU with the bundled sample data: no cloud account, no GPU, no network after installation.

## Quick start

You need Python 3.12 or newer and about 450 MB of disk space for the environment.

### 1. Clone the repository

```bash
git clone https://github.com/Data-Engineering-for-Multimodal-AI/playbook-04-multimodal-ingestion-pipeline.git
cd playbook-04-multimodal-ingestion-pipeline
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The versions are pinned. Using [uv](https://docs.astral.sh/uv/) instead: `uv sync`, then prefix the Python commands below with `uv run` (or run `make pipeline`).

### 4. Create necessary directories

```bash
mkdir -p orders processed_orders robot_logs video_features_enhanced notification_logs qa_results video_embeddings
```

This step is optional now: each script creates the folders it writes to.

### Run the pipeline step by step

Run these from the repository root, in this order. The outputs go to the folders above.

**1. Start the Order API** (in its own terminal; it runs until you press Ctrl-C):

```bash
python ByteEats/api/order_api.py
```

The API will start on http://localhost:8000.

**2. Process a sample order** (in a second terminal):

```bash
curl -X POST "http://localhost:8000/api/process-sample-order"
```

**3. Run the Robot Trigger Service:**

```bash
python ByteEats/services/robot_trigger.py --once
```

`--once` processes the waiting order and exits. The printed command, without `--once`, still works: it keeps watching `orders/` for new orders until you press Ctrl-C.

**4. Extract video features:**

```bash
python ByteEats/extractors/enhanced_extractor.py
```

It reads the seven videos in `ByteEats/data/sample-videos/`. Use `--video-dir`, `--output-dir` and `--order-id` to point it elsewhere.

**5. Send the pickup notification:**

```bash
python ByteEats/services/notification.py --once
```

Like the robot service, it polls until Ctrl-C without `--once`.

**6. Run quality checks:**

```bash
python ByteEats/analysis/quality_checker.py ORD1001 "Salmon Poke Bowl"
```

**7. Generate vector embeddings:**

```bash
python ByteEats/analysis/embedding_generator.py generate ORD1001
```

The same script finds the videos most similar to one video, and flags videos that differ from the rest of the order:

```bash
python ByteEats/analysis/embedding_generator.py similar ByteEats/data/sample-videos/salmon-prep-1.mp4 --top 3
python ByteEats/analysis/embedding_generator.py anomalies ORD1001 --threshold 0.8
```

**8. Run the similarity search demo:**

```bash
python ByteEats/analysis/similarity_demo.py
```

Stop the Order API with Ctrl-C when you're done.

### Run the complete pipeline

```bash
python ByteEats/run_pipeline.py
```

This runs every step in sequence and prints a summary. It starts the Order API itself (and stops it at the end) unless one is already running on port 8000, so you don't need step 1 first. It takes about 20 seconds and exits with code 1 if any step fails.

### Other scripts

- **Older script names.** The scripts at the repository root (`byteats_order_api.py`, `robot_trigger_service.py`, `pickup_notification_service.py`, `video_feature_extractor.py`, `enhanced_video_extractor.py`, `image_quality_checker.py`, `video_embedding_generator.py`, `image_similarity_demo.py`) run the matching `ByteEats/` module with the same arguments, so `python image_quality_checker.py ORD1001 "Salmon Poke Bowl"` still works.
- **Basic extractor.** `python ByteEats/extractors/video_extractor.py` is the simpler extractor; it writes to `video_features/`.
- **TensorFlow (optional, untested).** `advanced_video_extractor.py` detects objects with an SSD MobileNet v2 model from TensorFlow Hub, and `embedding_generator.py` uses the Universal Sentence Encoder if TensorFlow is installed. Neither is part of the tested pipeline, and `requirements.txt` doesn't install TensorFlow. To try them: `pip install tensorflow tensorflow-hub` (TensorFlow 2.21 supports Python 3.12 and 3.13). The `tfhub.dev` model links now resolve to Kaggle Models; Google states `hub.load()` still accepts them. Without TensorFlow, the embedding generator prints "TensorFlow not available. Using fallback embedding method." and uses the video features directly.

### Checks

```bash
make smoke    # or: uv run pytest -q
```

`make smoke` (about a minute on a laptop CPU) checks the sample data, the Order API, each step's output and exit code, the root wrappers, and `run_pipeline.py` end to end. GitHub Actions runs it on Python 3.12 and 3.13, plus the printed `pip install -r requirements.txt` path with `run_pipeline.py`.

## Tested on

| | |
|---|---|
| Date | 2026-10-23 |
| OS | macOS 26.5.1 (arm64); GitHub Actions `ubuntu-latest` |
| Python | 3.12.15 and 3.13.16 |
| Key packages | opencv-python-headless 5.0.0.93, numpy 2.5.4, scikit-learn 1.9.1, matplotlib 3.11.2, fastapi 0.161.1, pydantic 2.14.1, uvicorn 0.54.0 |
| Install time | `pip install -r requirements.txt`: 16 s (environment 415 MB) |
| Pipeline | `python ByteEats/run_pipeline.py`: about 21 s from a fresh clone |

Every command above was run from a fresh clone of this branch on 3.12. The extracted video features are identical to a run of the first-edition code with its original pinned versions (OpenCV 4.8, NumPy 1.26).

## Differences from the printed book

| What | Printed book | Now | Why |
|---|---|---|---|
| Clone URL and folder | `github.com/yourusername/ByteEats` | `playbook-04-multimodal-ingestion-pipeline` | The repository's actual location |
| Python | 3.9+ | 3.12+ | Current dependencies. The 2023 pins didn't install on Python 3.13 |
| Video extraction (step 4) | Crashed: it read `sample-data/sample-prep-cycle-end-to-end/sample-prep-cycle`, which isn't in the repository | Reads `ByteEats/data/sample-videos` | Bug. Its `--video-dir`, `--output-dir` and `--order-id` options were also ignored; they work now |
| `video_extractor.py`, `advanced_video_extractor.py` | `SyntaxError` (`global` after use); never ran | Fixed | Bug |
| Quality check result | `PASSED`, overall score 72.30% | `FAILED`, overall score 16.13% | The SSIM term was computed on 8-bit pixels and overflowed to 1.92 (the valid range is -1 to 1), which supplied about 67 of the 72.30 points. Computed correctly, SSIM is 0.31: the last packaging frame is a side view and the reference is a top-down photo. Identical images still score 100% |
| Robot trigger and notification services | Polled forever | `--once` processes what's waiting and exits; without it they poll as before | So the steps, the complete pipeline and the tests can finish |
| `run_pipeline.py` | Hung at step 2; needed the API already running and `curl` | Finishes; starts and stops the API itself; exits 1 on failure | The services never exited, and it ran them in `ByteEats/` while the API wrote to `./orders` |
| Pickup notification | Not in the step list | Step 5 | `run_pipeline.py` already ran it |
| Embedding t-SNE plot | Never written (a warning in the log) | `video_embeddings/ORD1001_embeddings_viz.png` | t-SNE's default perplexity (30) must be below the number of videos (7) |
| Similarity figure | Showed the first three images in the folder | Shows the three most similar | Bug |
| Exit codes | 0 even when the input was missing | 1 when the quality check or embedding step can't run | So failures are visible |
| Duplicate files | Eight scripts and two image folders existed both at the root and in `ByteEats/` | One copy in `ByteEats/`; the root scripts are wrappers with the old names | The copies were identical. The code read the root images; it now reads `ByteEats/data`, which the README documented |
| `setup.py`, `setup_github.sh`, empty files `#` and `On` | Present | Removed | `pyproject.toml` replaces `setup.py`; the GitHub helper set up a new repository under a placeholder account |
| OpenCV | `opencv-python` 4.8 | `opencv-python-headless` 5.0 | OpenCV 5.0 was released in June 2026; the code uses no window functions, and the results are identical |
| Pydantic | `order.dict()` | `order.model_dump()` | `.dict()` is deprecated in Pydantic 2 |
| Sample order's restaurant | "Poke House" | "Sample Poke Bar" | The sample named a real restaurant chain; the playbook uses fictional businesses |
| `docs/images/`, `tests/` in the project tree | Listed, but missing | `tests/` added; `docs/images/` removed from the tree | Accuracy |

Details are in [CHANGELOG.md](CHANGELOG.md).

## Data and licenses

| Files | Source | License |
|---|---|---|
| `ByteEats/data/sample-videos/*.mp4` (7 videos) | Created by the author with a Google generative video model | MIT |
| `ByteEats/data/reference_images/`, `ByteEats/data/sample_bowls/` (6 images) | Created by the author with the same model; two are frames from the videos | MIT |

None of it shows a real restaurant, kitchen, robot or person. [ByteEats/data/README.md](ByteEats/data/README.md) lists every file and what uses it.

## Expected output

From `python ByteEats/run_pipeline.py` and the step-by-step commands in a fresh clone, 2026-10-23, macOS 26.5.1 (arm64), Python 3.12.15. Log timestamps are trimmed.

**Step 3, robot trigger:**

```
robot-trigger - INFO - Found 1 orders to process
robot-trigger - INFO - Processing order ORD1001
robot-trigger - INFO - Triggering robot prep for order ORD1001
robot-trigger - INFO -   → Robot action: initialize_robot (0.5s)
robot-trigger - INFO -   → Robot action: grab_knife (1.2s)
robot-trigger - INFO -   → Robot action: cut_salmon (2.5s)
robot-trigger - INFO -   → Robot action: portion_salmon (1.0s)
robot-trigger - INFO -   → Robot action: grab_scoop (0.8s)
robot-trigger - INFO -   → Robot action: place_rice (1.5s)
robot-trigger - INFO -   → Robot action: add_salmon (1.0s)
robot-trigger - INFO -   → Robot action: add_seaweed (0.7s)
robot-trigger - INFO -   → Robot action: add_sauce (0.5s)
robot-trigger - INFO -   → Robot action: finalize_bowl (1.3s)
robot-trigger - INFO - Robot preparation complete. Log saved to robot_logs/ORD1001_robot_log.json
robot-trigger - INFO - Order ORD1001 processed and moved to processed_orders/ORD1001.json
```

**Step 4, video features** (one block per video; the first is shown, then the per-video results):

```
Found 7 videos to process
Processing video: delivery-packaging-7.mp4
  Extracted 12 frames over 8.00s
  Primary frame analysis: salmon_red:8.71%, salmon_orange:45.26%, sauce_brown:47.31%, rice_white:9.17%
  Edge density: 3.74%
  Average motion: 11.14%
  Video type: packaging, Stage: 0
  Detected ingredients: salmon, sauce
  Inferred robot actions: grab_container, complete_packaging
Features saved to video_features_enhanced/delivery-packaging-7_features.json
```

| Video | Average motion | Detected ingredients | Inferred robot actions |
|---|---|---|---|
| delivery-packaging-7 | 11.14% | salmon, sauce | grab_container, complete_packaging |
| quality-check-visual-6 | 22.53% | salmon, rice, seaweed, sauce | scan_bowl, measure_weight, verify_appearance, verify_ingredients |
| salmon-bowl-assembly-5 | 29.66% | salmon, rice, seaweed, sauce, container | initialize_assembly, place_rice, add_salmon, add_seaweed, add_sauce |
| salmon-prep-1 | 20.64% | container | initialize_robot, grab_knife, position_knife |
| salmon-prep-2 | 25.16% | container | position_salmon, begin_cutting |
| salmon-prep-3 | 21.93% | container | cut_salmon, portion_salmon |
| salmon-prep-4 | 16.62% | container | complete_salmon_prep, transfer_to_assembly |

The ingredient detection is color thresholding, so it is approximate: on the prep videos, the edge density is high enough to report a container but no ingredient color covers enough of the frame.

**Step 5, pickup notification:**

```
pickup-notification - INFO - Checking 1 orders for pickup readiness
pickup-notification - INFO - Checking pickup status for order ORD1001
pickup-notification - INFO - Order ORD1001 is ready for pickup!
pickup-notification - INFO - Sending pickup notification for order ORD1001 to customer CUST7890
pickup-notification - INFO - Order ORD1001 marked as ready for pickup
```

**Step 6, quality check:**

```
image-quality-checker - INFO - Checking quality for order ORD1001 (Salmon Poke Bowl)
image-quality-checker - INFO - Loading reference image: <repository>/ByteEats/data/reference_images/salmon_poke_bowl.jpg
image-quality-checker - INFO - Using packaging final frame: video_features_enhanced/delivery-packaging-7/frame_11.jpg
image-quality-checker - INFO - Quality check for order ORD1001: FAILED
image-quality-checker - INFO - Overall score: 16.13%
image-quality-checker - INFO - Quality report saved to qa_results/ORD1001_quality_report.json
```

The report's metrics (each scaled to 100): SSIM 31.17, ORB feature score 25.40, histogram correlation 0.70, MSE score 0.01. The pass threshold is 70%. The comparison image is `qa_results/ORD1001_comparison.jpg`.

On Linux x86-64 (GitHub Actions, both Python versions) the score is 16.41%: the video frame decodes slightly differently there (MSE 15,105 against 15,125), which moves the ORB feature score to 26.20 and the histogram correlation to 1.30. The check fails either way.

**Step 7, embeddings:**

```
video-embeddings - INFO - Embeddings visualization saved to video_embeddings/ORD1001_embeddings_viz.png
video-embeddings - INFO - Generated 7 embeddings for order ORD1001
TensorFlow not available. Using fallback embedding method.
{
  "success": true,
  "order_id": "ORD1001",
  "embeddings_count": 7,
  "embeddings_file": "video_embeddings/ORD1001_embeddings.pkl"
}
```

**Step 8, similarity search:**

```
Processing 7 images...
Similarity matrix saved to similarity_results/similarity_matrix_opencv.png
Embeddings visualization saved to similarity_results/embeddings_visualization_opencv.png

Most similar to Reference Bowl:
1. Reference Bowl (similarity: 1.0000)
2. miso salmon bowl (similarity: 0.7184)
3. salmon edamame bowl (similarity: 0.4624)
Query results visualization saved to similarity_results/similar_to_reference_opencv.png
Embeddings saved to similarity_results/bowl_embeddings_opencv.pkl
```

The first match scores 1.0000 because `sample_bowls/salmon_poke_bowl.jpg` is the same image as the reference. Matplotlib may also print a "Tight layout not applied" warning; the figures are still saved.

**Complete pipeline** (`python ByteEats/run_pipeline.py`) ends with:

```
Successfully completed 7/7 steps.

The entire ByteEats pipeline has completed successfully!
Check the output directories for generated files and visualizations.
```

"Completed" means every step ran; the quality check inside it still reports `FAILED` for this order.

## Project structure

```
├── ByteEats/
│   ├── api/order_api.py              # 1. Order ingestion API (FastAPI)
│   ├── services/
│   │   ├── robot_trigger.py          # 2. Robot preparation service
│   │   └── notification.py           # 7. Pickup notification service
│   ├── extractors/
│   │   ├── video_extractor.py        # Basic video feature extractor
│   │   └── enhanced_extractor.py     # 3. Enhanced feature extraction
│   ├── analysis/
│   │   ├── quality_checker.py        # 4. Image quality comparison
│   │   ├── embedding_generator.py    # 5. Vector embeddings, similar videos, anomalies
│   │   └── similarity_demo.py        # 6. Image similarity search
│   ├── data/                         # Sample videos and images (see data/README.md)
│   ├── docs/pipeline.md              # Architecture notes
│   ├── models/, utils/               # Empty packages
│   └── run_pipeline.py               # Runs every step
├── advanced_video_extractor.py       # Optional TensorFlow object detection (untested)
├── *.py                              # Older script names; each runs its ByteEats/ module
├── tests/test_smoke.py               # make smoke
├── Makefile
├── pyproject.toml, uv.lock           # Dependencies (uv)
├── requirements.txt                  # Pinned, exported from uv.lock
├── CHANGELOG.md
└── LICENSE
```

## Data engineering highlights

The ByteEats pipeline showcases several advanced data engineering concepts:

- **Extract-Load-Transform (ELT)**: Converting raw video data into structured features and vector embeddings
- **Computer Vision Integration**: Using OpenCV for extracting meaningful features from video frames
- **Vector Embeddings**: Transforming complex visual data into high-dimensional vectors
- **Similarity Search**: Finding similar items based on vector similarity
- **Anomaly Detection**: Identifying unusual patterns in preparation sequences
- **Service-Oriented Architecture**: Building a pipeline of decoupled services

## Production considerations

For a production deployment, consider:

1. **Scalability**:
   - Distributed video processing with Apache Spark
   - Vector databases like FAISS or Pinecone for efficient similarity search

2. **Reliability**:
   - Message queues (Kafka, RabbitMQ) for reliable service communication
   - Service orchestration with Kubernetes

3. **Monitoring**:
   - Metrics collection for pipeline performance
   - Anomaly detection to identify quality issues

4. **Storage Optimization**:
   - Video compression and selective storage
   - Time-to-live policies for raw data

## Troubleshooting

- **`Address already in use` / the API doesn't start**: something else is on port 8000. Stop it, or stop your own earlier `order_api.py`.
- **`run_pipeline.py` used an API that was already running**: that API writes orders to the folder it was started from. Start it from the repository root, or stop it and let `run_pipeline.py` start its own.
- **The robot or notification service never returns**: add `--once`, or press Ctrl-C.
- **`No features file found for order ORD1001`**: run the extractor (step 4) first, from the same folder.

## License

This project is licensed under the MIT License - see the LICENSE file for details. The sample videos and images in `ByteEats/data/` are MIT-licensed too.

## Acknowledgments

- OpenCV for computer vision capabilities
- scikit-learn for machine learning utilities
- FastAPI for the API framework
