#!/usr/bin/env python3
"""
ByteEats Complete Pipeline Runner
--------------------------------
This script demonstrates the entire ByteEats pipeline from order to similarity search.

It runs every step from the repository root, so outputs land in the same
folders as the step-by-step commands in the README. It starts the Order API
itself unless one is already answering on port 8000, runs the two polling
services with --once, and exits 1 if any step fails.
"""

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

# Run from the repository root (the folder above ByteEats/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

API_URL = "http://localhost:8000"
PYTHON = sys.executable

# Create necessary directories
directories = [
    "orders",
    "processed_orders",
    "robot_logs",
    "video_features_enhanced",
    "notification_logs",
    "qa_results",
    "video_embeddings",
    "similarity_results"
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)

def print_header(message):
    """Print a formatted header message"""
    print("\n" + "=" * 80)
    print(f" {message} ".center(80, "="))
    print("=" * 80 + "\n", flush=True)

def api_is_up():
    try:
        with urllib.request.urlopen(f"{API_URL}/", timeout=2) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError):
        return False

def start_api():
    """Start the Order API unless one is already running; return the process we started"""
    if api_is_up():
        print(f"Using the Order API already running at {API_URL}")
        return None
    print(f"Starting the Order API at {API_URL}")
    process = subprocess.Popen([PYTHON, "ByteEats/api/order_api.py"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        if api_is_up():
            return process
        if process.poll() is not None:
            break
        time.sleep(0.5)
    process.terminate()
    raise SystemExit(f"The Order API did not start (is port 8000 in use?). "
                     f"Start it yourself with: {PYTHON} ByteEats/api/order_api.py")

def create_sample_order():
    """Step 1: the same call as `curl -X POST http://localhost:8000/api/process-sample-order`"""
    print(f"Running: POST {API_URL}/api/process-sample-order\n")
    request = urllib.request.Request(f"{API_URL}/api/process-sample-order", method="POST")
    with urllib.request.urlopen(request, timeout=10) as response:
        print(json.dumps(json.load(response)))
    # The API saves the order in a background task; wait for the file
    for _ in range(20):
        if os.path.exists("orders/ORD1001.json"):
            return True
        time.sleep(0.25)
    print("orders/ORD1001.json was not written")
    return False

def run_process(command, description, timeout=300):
    """Run a command as a subprocess"""
    print_header(description)
    print(f"Running: {' '.join(command)}\n", flush=True)

    try:
        process = subprocess.run(command, timeout=timeout)
        print(f"\nCommand completed with exit code: {process.returncode}")
        return process.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"\nCommand timed out after {timeout} seconds")
        return False
    except Exception as e:
        print(f"\nError running command: {e}")
        return False

def main():
    """Run the complete ByteEats pipeline"""
    print_header("ByteEats Food Preparation Pipeline Demo")
    os.environ.setdefault("MPLBACKEND", "Agg")  # the scripts save figures; no window needed

    steps = [
        {
            "name": "Trigger Robot Preparation",
            "command": [PYTHON, "ByteEats/services/robot_trigger.py", "--once"],
            "description": "Step 2: Triggering robot preparation"
        },
        {
            "name": "Extract Video Features",
            "command": [PYTHON, "ByteEats/extractors/enhanced_extractor.py"],
            "description": "Step 3: Extracting features from preparation videos"
        },
        {
            "name": "Check Order Readiness",
            "command": [PYTHON, "ByteEats/services/notification.py", "--once"],
            "description": "Step 4: Checking if order is ready for pickup"
        },
        {
            "name": "Perform Quality Assurance",
            "command": [PYTHON, "ByteEats/analysis/quality_checker.py", "ORD1001", "Salmon Poke Bowl"],
            "description": "Step 5: Performing quality assurance checks"
        },
        {
            "name": "Generate Vector Embeddings",
            "command": [PYTHON, "ByteEats/analysis/embedding_generator.py", "generate", "ORD1001"],
            "description": "Step 6: Generating vector embeddings"
        },
        {
            "name": "Run Similarity Search",
            "command": [PYTHON, "ByteEats/analysis/similarity_demo.py"],
            "description": "Step 7: Demonstrating similarity search"
        }
    ]

    api_process = start_api()
    try:
        print_header("Step 1: Creating a sample order")
        successful_steps = 1 if create_sample_order() else 0
        if not successful_steps:
            print("\nWarning: Step 'Process Sample Order' did not complete successfully.")

        # Run each step
        for step in steps:
            if run_process(step["command"], step["description"]):
                successful_steps += 1
            else:
                print(f"\nWarning: Step '{step['name']}' did not complete successfully.")
    finally:
        if api_process is not None:
            api_process.terminate()
            api_process.wait(timeout=10)

    total_steps = len(steps) + 1

    # Final summary
    print_header("Pipeline Execution Summary")
    print(f"Successfully completed {successful_steps}/{total_steps} steps.")

    if successful_steps == total_steps:
        print("\nThe entire ByteEats pipeline has completed successfully!")
        print("Check the output directories for generated files and visualizations.")
        return 0
    print("\nSome steps in the pipeline did not complete successfully.")
    print("Please check the output above for more details.")
    return 1

if __name__ == "__main__":
    sys.exit(main())
