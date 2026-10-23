#!/usr/bin/env python3
"""
ByteEats Video Feature Extractor
--------------------------------
Extracts features from food preparation videos including:
- Order ID
- Menu item
- Robot action sequence
- Sensor logs
"""

import os
import cv2
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import uuid
import argparse

# Configuration
# Sample data ships in ByteEats/data, found relative to this file so the
# script works from any folder. Outputs go to the current folder.
DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
VIDEO_DIR = os.path.join(DATA_DIR, "sample-videos")
OUTPUT_DIR = "video_features"
ORDER_ID = "ORD1001"  # Using the existing order ID
MENU_ITEM = "Salmon Poke Bowl"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_frame(video_path, frame_number):
    """Extract a specific frame from a video"""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Handle case where requested frame is beyond video length
    if frame_number >= frame_count:
        frame_number = frame_count - 1
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None
    
    return frame

def analyze_video_color(frame):
    """Analyze color distribution in a frame"""
    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define color ranges
    color_ranges = {
        "red": ([0, 50, 50], [10, 255, 255]),
        "orange": ([11, 50, 50], [25, 255, 255]),
        "yellow": ([26, 50, 50], [35, 255, 255]),
        "green": ([36, 50, 50], [70, 255, 255]),
        "blue": ([100, 50, 50], [130, 255, 255]),
        "brown": ([10, 100, 20], [20, 255, 200]),
        "white": ([0, 0, 200], [180, 30, 255])
    }
    
    results = {}
    
    for color_name, (lower, upper) in color_ranges.items():
        lower = np.array(lower)
        upper = np.array(upper)
        mask = cv2.inRange(hsv, lower, upper)
        ratio = cv2.countNonZero(mask) / (frame.shape[0] * frame.shape[1])
        results[color_name] = round(ratio * 100, 2)
    
    return results

def detect_motion(video_path, sample_rate=10):
    """Detect motion in video by comparing frames"""
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    # Sample frames at regular intervals
    frames = []
    motion_scores = []
    
    for frame_idx in range(0, frame_count, sample_rate):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            break
            
        # Convert to grayscale for motion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
        
        # Calculate motion between consecutive frames
        if len(frames) > 1:
            diff = cv2.absdiff(frames[-2], frames[-1])
            _, threshold = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            motion_score = cv2.countNonZero(threshold) / (gray.shape[0] * gray.shape[1])
            timestamp = frame_idx / fps
            motion_scores.append({"t": round(timestamp, 2), "motion_score": round(motion_score * 100, 2)})
    
    cap.release()
    return motion_scores, duration

def identify_video_type(video_path):
    """Identify video type based on filename"""
    filename = os.path.basename(video_path)
    
    if "salmon-prep" in filename:
        return "prep", int(filename.split('-')[-1].split('.')[0]) if filename.split('-')[-1].split('.')[0].isdigit() else 0
    elif "assembly" in filename:
        return "assembly", 0
    elif "quality-check" in filename:
        return "quality", 0
    elif "packaging" in filename or "delivery" in filename:
        return "packaging", 0
    else:
        return "unknown", 0

def extract_video_features(video_path):
    """Extract features from a video file"""
    print(f"Processing video: {os.path.basename(video_path)}")
    
    # Basic video properties
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    cap.release()
    
    # Extract middle frame for analysis
    middle_frame_idx = frame_count // 2
    middle_frame = extract_frame(video_path, middle_frame_idx)
    
    # Save frame as image
    video_name = os.path.basename(video_path).split('.')[0]
    frame_image_path = f"{OUTPUT_DIR}/{video_name}_frame.jpg"
    if middle_frame is not None:
        cv2.imwrite(frame_image_path, middle_frame)
        print(f"  Frame saved to: {frame_image_path}")
    
    # Analyze colors in the frame
    color_analysis = {}
    if middle_frame is not None:
        color_analysis = analyze_video_color(middle_frame)
        print(f"  Color analysis: {', '.join([f'{k}:{v}%' for k,v in color_analysis.items() if v > 5])}")
    
    # Detect motion
    motion_scores, _ = detect_motion(video_path)
    avg_motion = sum(score['motion_score'] for score in motion_scores) / len(motion_scores) if motion_scores else 0
    print(f"  Average motion: {avg_motion:.2f}%")
    
    # Identify video type and stage
    video_type, stage = identify_video_type(video_path)
    print(f"  Video type: {video_type}, Stage: {stage}")
    
    # Generate robot actions based on video type
    robot_actions = infer_robot_actions(video_type, stage, color_analysis, motion_scores)
    
    # Generate simulated sensor logs
    sensor_logs = generate_sensor_logs(video_type, stage, motion_scores, duration)
    
    # Combine all features
    features = {
        "order_id": ORDER_ID,
        "menu_item": MENU_ITEM,
        "video_path": video_path,
        "video_type": video_type,
        "stage": stage,
        "duration": round(duration, 2),
        "frame_count": frame_count,
        "fps": round(fps, 2),
        "frame_image_path": frame_image_path if middle_frame is not None else None,
        "color_analysis": {k: v for k, v in color_analysis.items() if v > 1},
        "robot_actions": robot_actions,
        "sensor_logs": sensor_logs,
        "motion_analysis": {
            "average_motion": round(avg_motion, 2),
            "samples": motion_scores[:10]  # Limit to first 10 samples for brevity
        },
        "extraction_timestamp": datetime.now().isoformat()
    }
    
    return features

def infer_robot_actions(video_type, stage, color_analysis, motion_scores):
    """Infer robot actions based on video type and analysis"""
    actions = []
    
    if video_type == "prep":
        # Actions for prep videos
        if stage == 1:
            actions = ["initialize", "grab_knife"]
        elif stage == 2:
            actions = ["position_salmon", "begin_cutting"]
        elif stage == 3:
            actions = ["cut_salmon", "portion_salmon"]
        elif stage == 4:
            actions = ["complete_salmon_prep", "transfer_to_station"]
        
    elif video_type == "assembly":
        # Actions for assembly videos
        has_rice = color_analysis.get("white", 0) > 15
        has_salmon = color_analysis.get("red", 0) > 5 or color_analysis.get("orange", 0) > 5
        
        actions = ["initialize_assembly"]
        if has_rice:
            actions.append("place_rice")
        if has_salmon:
            actions.append("add_salmon")
        
        actions.extend(["add_toppings", "finish_assembly"])
        
    elif video_type == "quality":
        # Actions for quality check videos
        actions = ["scan_bowl", "measure_weight", "verify_appearance"]
        
        # Add conditional checks based on color analysis
        if sum(color_analysis.values()) > 100:  # If many colors detected
            actions.append("verify_ingredients")
        
    elif video_type == "packaging":
        # Actions for packaging videos
        high_motion = any(score["motion_score"] > 20 for score in motion_scores)
        
        actions = ["grab_container"]
        if high_motion:
            actions.extend(["seal_bowl", "package_order"])
        else:
            actions.append("prepare_packaging")
    
    return actions

def generate_sensor_logs(video_type, stage, motion_scores, duration):
    """Generate simulated sensor logs based on video type and analysis"""
    logs = []
    
    # Generate timestamps over the duration of the video
    num_points = min(10, int(duration * 2))  # 2 points per second, max 10 points
    if num_points <= 0:
        num_points = 1
    
    timestamps = [duration * i / num_points for i in range(num_points)]
    
    for t in timestamps:
        # Base log entry
        log_entry = {"t": round(t, 2)}
        
        # Add type-specific sensor data
        if video_type == "prep":
            if t < duration / 3:
                log_entry["temperature"] = f"{2 + t:.1f}°C"
            else:
                log_entry["temperature"] = f"{3 + t/2:.1f}°C"
                
            if stage >= 3:
                log_entry["cut_precision"] = f"{85 + int(t) % 10}%"
                
        elif video_type == "assembly":
            base_weight = 200
            if t < duration / 3:
                log_entry["weight"] = f"{base_weight + int(t * 10)}g"
            elif t < 2 * duration / 3:
                log_entry["weight"] = f"{base_weight + 100 + int(t * 5)}g"
            else:
                log_entry["weight"] = f"{base_weight + 180 + int(t * 2)}g"
                
        elif video_type == "quality":
            if t < duration / 2:
                log_entry["qa_progress"] = f"{int(t / duration * 100)}%"
            else:
                log_entry["qa_score"] = f"{80 + int(t) % 15}%"
                
        elif video_type == "packaging":
            if t < duration / 3:
                log_entry["packaging_stage"] = "sealing"
            elif t < 2 * duration / 3:
                log_entry["packaging_stage"] = "labeling"
            else:
                log_entry["packaging_stage"] = "complete"
                log_entry["final_weight"] = f"{450 + int(t) % 30}g"
        
        # Add motion data from actual analysis if available
        matching_motion = next((m for m in motion_scores if abs(m["t"] - t) < 0.2), None)
        if matching_motion:
            log_entry["motion"] = matching_motion["motion_score"]
            
        logs.append(log_entry)
    
    return logs

def process_all_videos():
    """Process all videos in the directory"""
    all_videos = sorted([os.path.join(VIDEO_DIR, f) for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')])
    
    if not all_videos:
        print(f"No videos found in {VIDEO_DIR}")
        return
    
    print(f"Found {len(all_videos)} videos to process")
    
    all_features = []
    
    for video_path in all_videos:
        features = extract_video_features(video_path)
        all_features.append(features)
        
        # Save individual feature file
        video_name = os.path.basename(video_path).split('.')[0]
        feature_file = f"{OUTPUT_DIR}/{video_name}_features.json"
        with open(feature_file, 'w') as f:
            json.dump(features, f, indent=2)
        print(f"Features saved to {feature_file}")
        print("-" * 50)
    
    # Save combined features
    combined_file = f"{OUTPUT_DIR}/{ORDER_ID}_combined_features.json"
    combined_data = {
        "order_id": ORDER_ID,
        "menu_item": MENU_ITEM,
        "videos_processed": len(all_features),
        "features": all_features,
        "processing_timestamp": datetime.now().isoformat()
    }
    
    with open(combined_file, 'w') as f:
        json.dump(combined_data, f, indent=2)
    
    print(f"Combined features for all videos saved to {combined_file}")
    
    # Generate summary report
    generate_summary_report(all_features, combined_file)

def generate_summary_report(all_features, combined_file):
    """Generate a human-readable summary report"""
    report_file = f"{OUTPUT_DIR}/{ORDER_ID}_summary_report.txt"
    
    with open(report_file, 'w') as f:
        f.write(f"ByteEats Video Feature Extraction Report\n")
        f.write(f"=======================================\n\n")
        f.write(f"Order ID: {ORDER_ID}\n")
        f.write(f"Menu Item: {MENU_ITEM}\n")
        f.write(f"Videos Processed: {len(all_features)}\n")
        f.write(f"Processing Time: {datetime.now().isoformat()}\n\n")
        
        f.write(f"Video Summaries:\n")
        f.write(f"--------------\n\n")
        
        for idx, features in enumerate(all_features, 1):
            video_name = os.path.basename(features["video_path"])
            f.write(f"{idx}. {video_name} ({features['video_type']}, Stage {features['stage']})\n")
            f.write(f"   Duration: {features['duration']}s\n")
            f.write(f"   Robot Actions: {', '.join(features['robot_actions'])}\n")
            f.write(f"   Sensor Highlights: ")
            
            if features['sensor_logs']:
                logs = features['sensor_logs']
                if 'weight' in logs[-1]:
                    f.write(f"Final Weight: {logs[-1]['weight']}, ")
                if any('temperature' in log for log in logs):
                    temp_logs = [log for log in logs if 'temperature' in log]
                    f.write(f"Temperature Range: {temp_logs[0]['temperature']} to {temp_logs[-1]['temperature']}, ")
                if any('qa_score' in log for log in logs):
                    qa_logs = [log for log in logs if 'qa_score' in log]
                    f.write(f"QA Score: {qa_logs[-1]['qa_score']}, ")
            
            f.write("\n\n")
        
        f.write(f"\nComplete data available in: {os.path.basename(combined_file)}\n")
    
    print(f"Summary report generated at {report_file}")

def main():
    """Main entry point"""
    global VIDEO_DIR, OUTPUT_DIR, ORDER_ID
    parser = argparse.ArgumentParser(description='Extract features from food preparation videos')
    parser.add_argument('--video-dir', default=VIDEO_DIR, help='Directory containing video files')
    parser.add_argument('--output-dir', default=OUTPUT_DIR, help='Directory to save output files')
    parser.add_argument('--order-id', default=ORDER_ID, help='Order ID to associate with videos')
    
    args = parser.parse_args()
    
    VIDEO_DIR = args.video_dir
    OUTPUT_DIR = args.output_dir
    ORDER_ID = args.order_id
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process all videos
    process_all_videos()

if __name__ == "__main__":
    main() 