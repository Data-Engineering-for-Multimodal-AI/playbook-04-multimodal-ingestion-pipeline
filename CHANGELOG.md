# Changelog

## 2026.10.1 (2026-10-27)

### Changed
- README, after checking against the book (Early Release, pages 126–133): page numbers; the root
  script names are the ones the book prints; the differences table's left column is the first
  edition; a link to the bonus repository.

## 2026.10 (2026-10-23): 2026 maintenance update

The printed version is tagged `first-edition`.

### Fixed
- **Video extraction (README step 4) crashed**: the extractors read `sample-data/sample-prep-cycle-end-to-end/sample-prep-cycle`, which isn't in the repository. They read `ByteEats/data/sample-videos` now. `enhanced_extractor.py` ignored its `--video-dir`, `--output-dir` and `--order-id` options; they work now.
- `video_extractor.py` and `advanced_video_extractor.py` were a `SyntaxError` (`global` declared after the names were used), so they never ran.
- **Quality check score**: SSIM was computed on 8-bit pixels and overflowed (1.92 for the sample order; the valid range is -1 to 1), which supplied most of the overall score. It is computed in floating point now. The sample order's score goes from 72.30% (PASSED) to 16.13% (FAILED): the packaging frame is a side view and the reference is a top-down photo. Identical images still score 100%.
- **`run_pipeline.py` hung** at the robot step: the service polls forever, and reading its output blocked before the timeout applied. It also ran the steps in `ByteEats/` while the API wrote to `./orders`.
- The embedding step never wrote its t-SNE plot (perplexity 30 must be below the number of videos, 7).
- The similarity demo's figure showed the first three images in the folder instead of the three most similar.
- The quality checker and the embedding generator exited 0 when their input was missing; they exit 1 now.
- README: the real clone URL and folder, Python 3.12+, and a project tree that matches the repository.

### Added
- `--once` on the robot trigger and pickup notification services: handle what's waiting, then exit. Without it they poll as before.
- `run_pipeline.py` starts the Order API itself unless one is running on port 8000, stops it at the end, posts the sample order without `curl`, and exits 1 if a step fails.
- `ByteEats/data/README.md`: the source (created by the author with a Google generative video model) and license (MIT) of every video and image, and what uses each one.
- `pyproject.toml`, `uv.lock`, and `requirements.txt` exported from the lock and pinned.
- `tests/test_smoke.py`, `make smoke`, and GitHub Actions on Python 3.12 and 3.13, including the printed `pip install -r requirements.txt` path.

### Changed
- One copy of the code: the eight root scripts were identical to modules in `ByteEats/`. They are now wrappers that run the `ByteEats/` module, so the old names still work.
- The quality checker and similarity demo read images from `ByteEats/data` (found relative to the script) instead of root copies of the same files.
- `opencv-python` 4.8 → `opencv-python-headless` 5.0 (the code uses no window functions; the extracted features are identical). The other dependencies move to current versions; Python 3.12+.
- `order.dict()` → `order.model_dump()` (Pydantic 2).
- The sample order's restaurant is the fictional "Sample Poke Bar" (it named a real restaurant chain).
- The README lists the pickup notification as its own step, documents the `similar` and `anomalies` commands, and marks the TensorFlow paths as optional and untested.

### Removed
- The root copies of `reference_images/` and `sample_bowl_images/` (identical to `ByteEats/data/reference_images` and `ByteEats/data/sample_bowls`).
- `setup.py` (replaced by `pyproject.toml`), `setup_github.sh` (set up a new repository under a placeholder account), and two empty files named `#` and `On`.
- python-multipart (unused) from the dependencies. Pillow, python-dateutil and typing-extensions are no longer listed directly; they still install as dependencies of other packages.
