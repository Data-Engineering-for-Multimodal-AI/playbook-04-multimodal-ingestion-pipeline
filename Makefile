.PHONY: setup run serve order similar check-data load try-frame-sampling try-feature-scaling try-order-validation smoke pipeline clean

# A bare `make` runs the smoke test, as it did before the uniform targets were added.
.DEFAULT_GOAL := smoke

# Uniform targets for The Playbooks. The book's commands still work as printed: the root scripts
# (python byteats_order_api.py, python robot_trigger_service.py,
# python image_quality_checker.py ORD1001 "Salmon Poke Bowl" and the others), and
# python ByteEats/run_pipeline.py for the whole pipeline.

# Install the locked dependencies into .venv.
setup:
	uv sync --frozen

# Run the whole pipeline on the bundled ByteEats sample data and print a summary of each step.
# Same as `make pipeline`. It starts and stops the Order API itself.
run: setup
	@$(MAKE) --no-print-directory pipeline

# Start the Order API on http://localhost:8000 (README step 1). Runs until Ctrl-C.
serve: setup
	uv run --frozen python ByteEats/api/order_api.py

# With the API running (make serve): send one order to POST /api/orders and read it back.
#   make order ITEM="Miso Salmon Bowl"
order:
	uv run --frozen python scripts/send_order.py --item "$(or $(ITEM),Salmon Poke Bowl)"

# After make run: the sample videos most like one video (step 7, embedding_generator.py similar).
#   make similar VIDEO=salmon-prep-1
similar:
	uv run --frozen python ByteEats/analysis/embedding_generator.py similar "ByteEats/data/sample-videos/$(or $(VIDEO),salmon-prep-1).mp4" --top 3

# Your own order: order.json, videos/ and optionally reference_images/ in DIR.
# check-data checks every file and writes nothing. See README, "Use your own data".
#   make check-data DIR=my_order
#   make load DIR=my_order
check-data:
	@test -n "$(DIR)" || { echo "Usage: make check-data DIR=path/to/folder"; exit 2; }
	uv run --frozen python scripts/own_order.py --dir "$(DIR)" --check-only

load:
	@test -n "$(DIR)" || { echo "Usage: make load DIR=path/to/folder"; exit 2; }
	MPLBACKEND=Agg uv run --frozen python scripts/own_order.py --dir "$(DIR)"

# Try this: three measured experiments (see README, "Try this"). No Docker, no network.
try-frame-sampling: setup
	uv run --frozen python experiments/frame_sampling.py

# Needs the features from make run.
try-feature-scaling: setup
	uv run --frozen python experiments/feature_scaling.py

try-order-validation: setup
	uv run --frozen python experiments/order_validation.py

# Sample data, the API, every step and run_pipeline.py. CPU only, no network.
smoke:
	uv run --frozen pytest -q tests/test_smoke.py

# The whole pipeline in one command (README "Run the complete pipeline")
pipeline:
	uv run --frozen python ByteEats/run_pipeline.py

# Remove the folders the pipeline writes
clean:
	rm -rf orders processed_orders robot_logs video_features video_features_enhanced video_features_advanced notification_logs qa_results video_embeddings embedding_models similarity_results
