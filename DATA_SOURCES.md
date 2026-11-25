# Data licenses

This file lists the datasets and models this playbook uses, where they come from, and their licenses. The code in this repository is MIT-licensed. That license does not cover third-party data or models; each one keeps its own license, listed below.

## Datasets

| Dataset | Source | License | In this repo |
|---|---|---|---|
| Food-preparation videos (7 clips of one Salmon Poke Bowl order) | Created by the author with a Google generative video model | MIT (this repository) | `ByteEats/data/sample-videos/` (19 MB) |
| Bowl images (6; two are frames from the videos) | Created by the author with the same model | MIT (this repository) | `ByteEats/data/reference_images/` and `ByteEats/data/sample_bowls/` (3.4 MB) |

## Models

| Model | Source | License | How it's used |
|---|---|---|---|
| SSD MobileNet v2 (optional) | TensorFlow Hub, `https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2` (now served from Kaggle Models) | Not stated in this repo | Object detection in `advanced_video_extractor.py`. Optional and untested; downloaded by `hub.load()` only if you install TensorFlow yourself |
| Universal Sentence Encoder (optional) | TensorFlow Hub, `https://tfhub.dev/google/universal-sentence-encoder/4` (now served from Kaggle Models) | Not stated in this repo | Text embeddings in `ByteEats/analysis/embedding_generator.py`. Optional; without TensorFlow the generator uses the video features instead |

## Notes

- The repository does not name the generative video model used to create the sample videos and images. `ByteEats/data/README.md` says they show no real restaurant, kitchen, robot or person.
- The two TensorFlow Hub models are not installed by `requirements.txt` and are not part of the tested pipeline. The repo does not state their licenses.

## Citations

- No citation given by the sources for these items.
