.PHONY: smoke pipeline clean

# Sample data, the API, every step and run_pipeline.py. CPU only, no network.
smoke:
	uv run --frozen pytest -q tests/test_smoke.py

# The whole pipeline in one command (README "Run the complete pipeline")
pipeline:
	uv run --frozen python ByteEats/run_pipeline.py

# Remove the folders the pipeline writes
clean:
	rm -rf orders processed_orders robot_logs video_features video_features_enhanced video_features_advanced notification_logs qa_results video_embeddings embedding_models similarity_results
