#!/usr/bin/env python3
"""
ByteEats Complete Pipeline Runner
--------------------------------
This script demonstrates the entire ByteEats pipeline from order to similarity search.
"""

import os
import subprocess
import time
import sys
import json
from pathlib import Path

# Ensure we're in the right directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Create necessary directories
directories = [
    "../orders",
    "../processed_orders",
    "../robot_logs",
    "../video_features_enhanced",
    "../notification_logs",
    "../qa_results",
    "../video_embeddings",
    "../similarity_results"
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)

def print_header(message):
    """Print a formatted header message"""
    print("\n" + "=" * 80)
    print(f" {message} ".center(80, "="))
    print("=" * 80 + "\n")

def run_process(command, description, timeout=10):
    """Run a command as a subprocess"""
    print_header(description)
    print(f"Running: {' '.join(command)}\n")
    
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Display output in real-time
        for line in process.stdout:
            print(line, end="")
        
        process.wait(timeout=timeout)
        print(f"\nCommand completed with exit code: {process.returncode}")
        return process.returncode == 0
    except subprocess.TimeoutExpired:
        process.kill()
        print(f"\nCommand timed out after {timeout} seconds")
        return False
    except Exception as e:
        print(f"\nError running command: {e}")
        return False

def main():
    """Run the complete ByteEats pipeline"""
    print_header("ByteEats Food Preparation Pipeline Demo")
    
    steps = [
        {
            "name": "Process Sample Order",
            "command": ["curl", "-X", "POST", "http://localhost:8000/api/process-sample-order"],
            "description": "Step 1: Creating a sample order",
            "timeout": 5
        },
        {
            "name": "Trigger Robot Preparation",
            "command": ["python", "services/robot_trigger.py"],
            "description": "Step 2: Triggering robot preparation",
            "timeout": 10
        },
        {
            "name": "Extract Video Features",
            "command": ["python", "extractors/enhanced_extractor.py"],
            "description": "Step 3: Extracting features from preparation videos",
            "timeout": 30
        },
        {
            "name": "Check Order Readiness",
            "command": ["python", "services/notification.py"],
            "description": "Step 4: Checking if order is ready for pickup",
            "timeout": 10
        },
        {
            "name": "Perform Quality Assurance",
            "command": ["python", "analysis/quality_checker.py", "ORD1001", "Salmon Poke Bowl"],
            "description": "Step 5: Performing quality assurance checks",
            "timeout": 10
        },
        {
            "name": "Generate Vector Embeddings",
            "command": ["python", "analysis/embedding_generator.py", "generate", "ORD1001"],
            "description": "Step 6: Generating vector embeddings",
            "timeout": 10
        },
        {
            "name": "Run Similarity Search",
            "command": ["python", "analysis/similarity_demo.py"],
            "description": "Step 7: Demonstrating similarity search",
            "timeout": 10
        }
    ]
    
    # Track successful steps
    successful_steps = 0
    
    # Run each step
    for step in steps:
        success = run_process(
            step["command"],
            step["description"],
            step.get("timeout", 10)
        )
        
        if success:
            successful_steps += 1
        else:
            print(f"\nWarning: Step '{step['name']}' did not complete successfully.")
        
        # Brief pause between steps
        time.sleep(1)
    
    # Final summary
    print_header("Pipeline Execution Summary")
    print(f"Successfully completed {successful_steps}/{len(steps)} steps.")
    
    if successful_steps == len(steps):
        print("\nThe entire ByteEats pipeline has completed successfully!")
        print("Check the output directories for generated files and visualizations.")
    else:
        print("\nSome steps in the pipeline did not complete successfully.")
        print("Please check the output above for more details.")

if __name__ == "__main__":
    main() 