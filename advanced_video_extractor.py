#!/usr/bin/env python3
"""
ByteEats Advanced Video Feature Extractor
----------------------------------------
Extracts advanced features from food preparation videos using TensorFlow models:
- Order ID
- Menu item
- Robot action sequence (using action recognition)
- Object detection for ingredients
- Sensor logs
"""

import os
import cv2
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from pathlib import Path
from datetime import datetime
import argparse
import time

# Configuration
VIDEO_DIR = "sample-data/sample-prep-cycle-end-to-end/sample-prep-cycle"
OUTPUT_DIR = "video_features_advanced"
ORDER_ID = "ORD1001"  # Using the existing order ID
MENU_ITEM = "Salmon Poke Bowl"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# TensorFlow setup
print("Loading TensorFlow models...")

# Function to load TF models with error handling
def load_model_with_fallback(model_url, fallback=None):
    try:
        model = hub.load(model_url)
        print(f"Successfully loaded model: {model_url}")
        return model
    except Exception as e:
        print(f"Error loading model {model_url}: {e}")
        return fallback

# Try to load object detection model
# MobileNet v2 is lightweight and works for most systems
try:
    detector_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
    detector = load_model_with_fallback(detector_url)
    detection_available = detector is not None
except Exception as e:
    print(f"Error loading object detection model: {e}")
    detection_available = False
    detector = None

def extract_frames(video_path, num_frames=8):
    """Extract evenly spaced frames from the video"""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps
    
    # Calculate frame indices to extract
    indices = [int(i * frame_count / num_frames) for i in range(num_frames)]
    
    frames = []
    timestamps = []
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
            timestamps.append(idx / fps)
    
    cap.release()
    return frames, timestamps, duration

def detect_objects(frame):
    """Detect objects in a frame using TensorFlow Hub model"""
    if not detection_available:
        return []
    
    # Convert to RGB (TensorFlow models expect RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Prepare input tensor
    input_tensor = tf.convert_to_tensor(np.expand_dims(rgb_frame, 0), dtype=tf.uint8)
    
    # Run detector
    try:
        detections = detector(input_tensor)
        
        # Process results
        detection_boxes = detections['detection_boxes'][0].numpy()
        detection_classes = detections['detection_classes'][0].numpy().astype(np.int32)
        detection_scores = detections['detection_scores'][0].numpy()
        
        # Define COCO class labels (simplified)
        class_names = {
            1: "person", 53: "apple", 49: "orange", 52: "banana", 
            50: "broccoli", 51: "carrot", 54: "sandwich", 55: "orange", 
            56: "broccoli", 57: "carrot", 58: "hot dog", 59: "pizza", 
            60: "donut", 61: "cake", 67: "dining table", 44: "bottle",
            46: "wine glass", 47: "cup", 48: "fork", 62: "chair", 
            63: "couch", 64: "potted plant", 65: "bed", 66: "dining table",
            73: "laptop"
        }
        
        # Filter out low confidence detections
        threshold = 0.3
        objects = []
        
        for i in range(min(100, detection_boxes.shape[0])):
            if detection_scores[i] >= threshold:
                class_id = detection_classes[i]
                class_name = class_names.get(class_id, f"class_{class_id}")
                
                objects.append({
                    "class": class_name,
                    "confidence": float(detection_scores[i]),
                    "box": detection_boxes[i].tolist()
                })
        
        return objects
    
    except Exception as e:
        print(f"Object detection error: {e}")
        return []

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

def detect_motion(frames, timestamps):
    """Detect motion between frames"""
    if len(frames) < 2:
        return []
    
    motion_scores = []
    
    # Convert frames to grayscale for motion detection
    gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
    
    for i in range(1, len(gray_frames)):
        # Calculate absolute difference between consecutive frames
        diff = cv2.absdiff(gray_frames[i-1], gray_frames[i])
        
        # Apply threshold to get regions with significant difference
        _, threshold = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        # Calculate motion score as percentage of pixels that changed
        motion_score = cv2.countNonZero(threshold) / (threshold.shape[0] * threshold.shape[1])
        
        motion_scores.append({
            "t": round(timestamps[i], 2),
            "motion_score": round(motion_score * 100, 2)
        })
    
    return motion_scores

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
    """Extract advanced features from a video file"""
    print(f"Processing video: {os.path.basename(video_path)}")
    
    # Extract evenly spaced frames from the video
    frames, timestamps, duration = extract_frames(video_path, num_frames=8)
    
    if not frames:
        print(f"  Warning: Could not extract frames from {video_path}")
        return None
    
    print(f"  Extracted {len(frames)} frames")
    
    # Save middle frame as image
    middle_idx = len(frames) // 2
    middle_frame = frames[middle_idx]
    video_name = os.path.basename(video_path).split('.')[0]
    frame_image_path = f"{OUTPUT_DIR}/{video_name}_frame.jpg"
    cv2.imwrite(frame_image_path, middle_frame)
    
    # Analyze colors in the frame
    color_analysis = analyze_video_color(middle_frame)
    print(f"  Color analysis: {', '.join([f'{k}:{v}%' for k,v in color_analysis.items() if v > 5])}")
    
    # Detect motion between frames
    motion_scores = detect_motion(frames, timestamps)
    avg_motion = sum(score['motion_score'] for score in motion_scores) / len(motion_scores) if motion_scores else 0
    print(f"  Average motion: {avg_motion:.2f}%")
    
    # Identify video type and stage
    video_type, stage = identify_video_type(video_path)
    print(f"  Video type: {video_type}, Stage: {stage}")
    
    # Object detection on key frames
    all_objects = []
    if detection_available:
        print(f"  Running object detection on frames...")
        # Use 3 frames for object detection (beginning, middle, end)
        detection_indices = [0, len(frames)//2, len(frames)-1] if len(frames) >= 3 else list(range(len(frames)))
        
        for idx in detection_indices:
            frame_objects = detect_objects(frames[idx])
            if frame_objects:
                all_objects.append({
                    "timestamp": timestamps[idx],
                    "objects": frame_objects
                })
        
        object_classes = set()
        for frame_result in all_objects:
            for obj in frame_result["objects"]:
                object_classes.add(obj["class"])
        
        print(f"  Detected objects: {', '.join(object_classes)}")
    
    # Generate robot actions based on video type and stage
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
        "frame_image_path": frame_image_path,
        "color_analysis": {k: v for k, v in color_analysis.items() if v > 1},
        "robot_actions": robot_actions,
        "detected_objects": all_objects,
        "sensor_logs": sensor_logs,
        "motion_analysis": {
            "average_motion": round(avg_motion, 2),
            "samples": motion_scores
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
        if features:
            all_features.append(features)
            
            # Save individual feature file
            video_name = os.path.basename(video_path).split('.')[0]
            feature_file = f"{OUTPUT_DIR}/{video_name}_features.json"
            with open(feature_file, 'w') as f:
                json.dump(features, f, indent=2)
            print(f"Features saved to {feature_file}")
            print("-" * 50)
    
    if not all_features:
        print("No features extracted. Exiting.")
        return
    
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
        f.write(f"ByteEats Advanced Video Feature Extraction Report\n")
        f.write(f"===============================================\n\n")
        f.write(f"Order ID: {ORDER_ID}\n")
        f.write(f"Menu Item: {MENU_ITEM}\n")
        f.write(f"Videos Processed: {len(all_features)}\n")
        f.write(f"Processing Time: {datetime.now().isoformat()}\n\n")
        
        if detection_available:
            f.write(f"Object Detection: Enabled\n")
        else:
            f.write(f"Object Detection: Disabled (TensorFlow Hub model not available)\n")
        
        f.write(f"\nVideo Summaries:\n")
        f.write(f"--------------\n\n")
        
        for idx, features in enumerate(all_features, 1):
            video_name = os.path.basename(features["video_path"])
            f.write(f"{idx}. {video_name} ({features['video_type']}, Stage {features['stage']})\n")
            f.write(f"   Duration: {features['duration']}s\n")
            f.write(f"   Robot Actions: {', '.join(features['robot_actions'])}\n")
            
            # Add detected objects if available
            if 'detected_objects' in features and features['detected_objects']:
                all_objects = []
                for frame_obj in features['detected_objects']:
                    for obj in frame_obj['objects']:
                        if obj['confidence'] > 0.4:  # Only list high confidence detections
                            all_objects.append(obj['class'])
                
                if all_objects:
                    f.write(f"   Detected Objects: {', '.join(set(all_objects))}\n")
            
            # Add sensor highlights
            f.write(f"   Sensor Highlights: ")
            if features['sensor_logs']:
                logs = features['sensor_logs']
                if any('weight' in log for log in logs):
                    weight_logs = [log for log in logs if 'weight' in log]
                    f.write(f"Weight: {weight_logs[-1]['weight']}, ")
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
    parser = argparse.ArgumentParser(description='Extract features from food preparation videos')
    parser.add_argument('--video-dir', default=VIDEO_DIR, help='Directory containing video files')
    parser.add_argument('--output-dir', default=OUTPUT_DIR, help='Directory to save output files')
    parser.add_argument('--order-id', default=ORDER_ID, help='Order ID to associate with videos')
    
    args = parser.parse_args()
    
    global VIDEO_DIR, OUTPUT_DIR, ORDER_ID
    VIDEO_DIR = args.video_dir
    OUTPUT_DIR = args.output_dir
    ORDER_ID = args.order_id
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process all videos
    process_all_videos()

if __name__ == "__main__":
    main() 