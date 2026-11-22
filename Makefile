.PHONY: setup run serve smoke pipeline clean

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

# Sample data, the API, every step and run_pipeline.py. CPU only, no network.
smoke:
	uv run --frozen pytest -q tests/test_smoke.py

# The whole pipeline in one command (README "Run the complete pipeline")
pipeline:
	uv run --frozen python ByteEats/run_pipeline.py

# Remove the folders the pipeline writes
clean:
	rm -rf orders processed_orders robot_logs video_features video_features_enhanced video_features_advanced notification_logs qa_results video_embeddings embedding_models similarity_results
