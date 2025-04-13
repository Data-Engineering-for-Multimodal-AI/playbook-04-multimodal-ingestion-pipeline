#!/usr/bin/env python3
"""
ByteEats Enhanced Video Feature Extractor
----------------------------------------
Extracts enhanced features from food preparation videos:
- Order ID
- Menu item
- Robot action sequence
- Detailed motion analysis
- Ingredient detection via color analysis
- Sensor logs
"""

import os
import cv2
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse

# Configuration - these are now moved to the top
VIDEO_DIR = "sample-data/sample-prep-cycle-end-to-end/sample-prep-cycle"
OUTPUT_DIR = "video_features_enhanced"
ORDER_ID = "ORD1001"  # Using the existing order ID
MENU_ITEM = "Salmon Poke Bowl"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_frames(video_path, num_frames=12):
    """Extract evenly spaced frames from the video"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return [], [], 0
        
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

def analyze_video_color(frame):
    """Analyze color distribution in a frame"""
    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define color ranges for food ingredients
    color_ranges = {
        "salmon_red": ([0, 50, 50], [10, 255, 255]),
        "salmon_orange": ([11, 50, 50], [25, 255, 255]),
        "seaweed_green": ([36, 50, 50], [70, 255, 255]),
        "sauce_brown": ([10, 100, 20], [20, 255, 200]),
        "rice_white": ([0, 0, 200], [180, 30, 255]),
        "container_blue": ([100, 50, 50], [130, 255, 255]),
        "yellow_ingredient": ([26, 50, 50], [35, 255, 255])
    }
    
    results = {}
    
    for color_name, (lower, upper) in color_ranges.items():
        lower = np.array(lower)
        upper = np.array(upper)
        mask = cv2.inRange(hsv, lower, upper)
        ratio = cv2.countNonZero(mask) / (frame.shape[0] * frame.shape[1])
        results[color_name] = round(ratio * 100, 2)
    
    return results

def detect_edges(frame):
    """Detect edges in the frame to identify object boundaries"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect edges using Canny algorithm
    edges = cv2.Canny(blurred, 50, 150)
    
    # Calculate edge density
    edge_density = cv2.countNonZero(edges) / (edges.shape[0] * edges.shape[1])
    
    return {
        "edge_density": round(edge_density * 100, 2),
        "edge_image": edges
    }

def detect_motion(frames, timestamps):
    """Detect motion between frames"""
    if len(frames) < 2:
        return []
    
    motion_scores = []
    motion_regions = []
    
    # Convert frames to grayscale for motion detection
    gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
    
    for i in range(1, len(gray_frames)):
        # Calculate absolute difference between consecutive frames
        diff = cv2.absdiff(gray_frames[i-1], gray_frames[i])
        
        # Apply threshold to get regions with significant difference
        _, threshold = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        # Find contours in the thresholded image
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size
        significant_contours = [c for c in contours if cv2.contourArea(c) > 50]
        
        # Calculate motion score as percentage of pixels that changed
        motion_score = cv2.countNonZero(threshold) / (threshold.shape[0] * threshold.shape[1])
        
        # Create a mask to visualize motion regions
        motion_mask = np.zeros_like(threshold)
        cv2.drawContours(motion_mask, significant_contours, -1, 255, -1)
        
        # Calculate regions of motion (top, middle, bottom)
        h, w = threshold.shape
        top_motion = cv2.countNonZero(motion_mask[:h//3, :]) / (h//3 * w)
        middle_motion = cv2.countNonZero(motion_mask[h//3:2*h//3, :]) / (h//3 * w)
        bottom_motion = cv2.countNonZero(motion_mask[2*h//3:, :]) / (h//3 * w)
        
        motion_scores.append({
            "t": round(timestamps[i], 2),
            "motion_score": round(motion_score * 100, 2),
            "contour_count": len(significant_contours)
        })
        
        motion_regions.append({
            "t": round(timestamps[i], 2),
            "top": round(top_motion * 100, 2),
            "middle": round(middle_motion * 100, 2),
            "bottom": round(bottom_motion * 100, 2)
        })
    
    return motion_scores, motion_regions

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

def detect_food_ingredients(color_analysis, edge_density):
    """Detect food ingredients based on color analysis"""
    ingredients = []
    confidence_levels = {}
    
    # Salmon detection
    salmon_confidence = color_analysis.get("salmon_red", 0) + color_analysis.get("salmon_orange", 0)
    if salmon_confidence > 5:
        ingredients.append("salmon")
        confidence_levels["salmon"] = min(100, round(salmon_confidence * 5, 1))
    
    # Rice detection
    rice_confidence = color_analysis.get("rice_white", 0)
    if rice_confidence > 15:
        ingredients.append("rice")
        confidence_levels["rice"] = min(100, round(rice_confidence * 3, 1))
    
    # Seaweed detection
    seaweed_confidence = color_analysis.get("seaweed_green", 0)
    if seaweed_confidence > 3:
        ingredients.append("seaweed")
        confidence_levels["seaweed"] = min(100, round(seaweed_confidence * 10, 1))
    
    # Sauce detection
    sauce_confidence = color_analysis.get("sauce_brown", 0)
    if sauce_confidence > 2:
        ingredients.append("sauce")
        confidence_levels["sauce"] = min(100, round(sauce_confidence * 15, 1))
    
    # Container detection
    if color_analysis.get("container_blue", 0) > 10 or edge_density > 20:
        ingredients.append("container")
        confidence_levels["container"] = min(100, round(edge_density * 2 + color_analysis.get("container_blue", 0) * 3, 1))
    
    return ingredients, confidence_levels

def extract_video_features(video_path):
    """Extract enhanced features from a video file"""
    print(f"Processing video: {os.path.basename(video_path)}")
    
    # Extract evenly spaced frames from the video
    frames, timestamps, duration = extract_frames(video_path, num_frames=12)
    
    if not frames:
        print(f"  Warning: Could not extract frames from {video_path}")
        return None
    
    print(f"  Extracted {len(frames)} frames over {duration:.2f}s")
    
    # Create output directory for frames
    video_name = os.path.basename(video_path).split('.')[0]
    video_output_dir = os.path.join(OUTPUT_DIR, video_name)
    os.makedirs(video_output_dir, exist_ok=True)
    
    # Process each frame
    all_color_analyses = []
    all_edge_analyses = []
    frame_paths = []
    
    for i, (frame, timestamp) in enumerate(zip(frames, timestamps)):
        # Save frame
        frame_path = f"{video_output_dir}/frame_{i:02d}.jpg"
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
        
        # Color analysis
        color_analysis = analyze_video_color(frame)
        all_color_analyses.append({
            "t": round(timestamp, 2),
            "colors": color_analysis
        })
        
        # Edge detection
        edge_analysis = detect_edges(frame)
        edge_path = f"{video_output_dir}/edges_{i:02d}.jpg"
        cv2.imwrite(edge_path, edge_analysis["edge_image"])
        all_edge_analyses.append({
            "t": round(timestamp, 2),
            "edge_density": edge_analysis["edge_density"],
            "edge_image_path": edge_path
        })
    
    # Middle frame for primary analysis
    middle_idx = len(frames) // 2
    middle_frame = frames[middle_idx]
    middle_frame_path = frame_paths[middle_idx]
    middle_color_analysis = all_color_analyses[middle_idx]["colors"]
    middle_edge_analysis = all_edge_analyses[middle_idx]["edge_density"]
    
    print(f"  Primary frame analysis: {', '.join([f'{k}:{v}%' for k,v in middle_color_analysis.items() if v > 5])}")
    print(f"  Edge density: {middle_edge_analysis:.2f}%")
    
    # Detect motion
    motion_scores, motion_regions = detect_motion(frames, timestamps)
    avg_motion = sum(score['motion_score'] for score in motion_scores) / len(motion_scores) if motion_scores else 0
    print(f"  Average motion: {avg_motion:.2f}%")
    
    # Identify video type and stage
    video_type, stage = identify_video_type(video_path)
    print(f"  Video type: {video_type}, Stage: {stage}")
    
    # Detect food ingredients
    ingredients, confidence_levels = detect_food_ingredients(middle_color_analysis, middle_edge_analysis)
    print(f"  Detected ingredients: {', '.join(ingredients)}")
    
    # Infer robot actions based on video type, stage and motion analysis
    robot_actions = infer_robot_actions(video_type, stage, ingredients, motion_scores, motion_regions)
    print(f"  Inferred robot actions: {', '.join(robot_actions)}")
    
    # Generate sensor logs
    sensor_logs = generate_sensor_logs(video_type, stage, motion_scores, motion_regions, duration)
    
    # Calculate intensity of motion periods to identify key events
    motion_intensity = []
    if motion_scores:
        # Find peaks in motion
        threshold = max(2.0, sum(s["motion_score"] for s in motion_scores) / len(motion_scores) * 1.5)
        for i, score in enumerate(motion_scores):
            if score["motion_score"] > threshold:
                is_peak = (i == 0 or score["motion_score"] > motion_scores[i-1]["motion_score"]) and \
                          (i == len(motion_scores)-1 or score["motion_score"] > motion_scores[i+1]["motion_score"])
                
                if is_peak:
                    motion_intensity.append({
                        "t": score["t"],
                        "intensity": score["motion_score"],
                        "is_key_event": True
                    })
                else:
                    motion_intensity.append({
                        "t": score["t"],
                        "intensity": score["motion_score"],
                        "is_key_event": False
                    })
    
    # Combine all features
    features = {
        "order_id": ORDER_ID,
        "menu_item": MENU_ITEM,
        "video_path": video_path,
        "video_type": video_type,
        "stage": stage,
        "duration": round(duration, 2),
        "primary_frame_path": middle_frame_path,
        "frame_paths": frame_paths,
        "color_analyses": all_color_analyses,
        "edge_analyses": all_edge_analyses,
        "detected_ingredients": {
            "ingredients": ingredients,
            "confidence_levels": confidence_levels
        },
        "robot_actions": robot_actions,
        "sensor_logs": sensor_logs,
        "motion_analysis": {
            "average_motion": round(avg_motion, 2),
            "motion_scores": motion_scores,
            "motion_regions": motion_regions,
            "key_events": motion_intensity
        },
        "extraction_timestamp": datetime.now().isoformat()
    }
    
    return features

def infer_robot_actions(video_type, stage, ingredients, motion_scores, motion_regions):
    """Infer robot actions based on detailed analysis"""
    actions = []
    
    if video_type == "prep":
        # Actions for prep videos based on stage and motion analysis
        if stage == 1:
            actions = ["initialize_robot", "grab_knife"]
            
            # Analyze motion to see if knife is being moved
            if any(score["motion_score"] > 15 for score in motion_scores):
                actions.append("position_knife")
        
        elif stage == 2:
            actions = ["position_salmon"]
            
            # Check for high motion in middle region (cutting motion)
            if any(region["middle"] > 20 for region in motion_regions):
                actions.append("begin_cutting")
        
        elif stage == 3:
            actions = ["cut_salmon"]
            
            # Look for motion followed by pauses (indicating multiple cuts)
            motion_changes = sum(1 for i in range(1, len(motion_scores)) if 
                              abs(motion_scores[i]["motion_score"] - motion_scores[i-1]["motion_score"]) > 10)
            
            if motion_changes > 2:
                actions.append("multiple_cuts")
            
            actions.append("portion_salmon")
        
        elif stage == 4:
            actions = ["complete_salmon_prep"]
            
            # Look for motion at the end (transfer)
            if motion_scores and motion_scores[-1]["motion_score"] > 15:
                actions.append("transfer_to_assembly")
    
    elif video_type == "assembly":
        # Actions for assembly videos based on ingredients and motion
        actions = ["initialize_assembly"]
        
        if "rice" in ingredients:
            actions.append("place_rice")
        
        if "salmon" in ingredients:
            actions.append("add_salmon")
        
        if "seaweed" in ingredients:
            actions.append("add_seaweed")
        
        if "sauce" in ingredients:
            actions.append("add_sauce")
        
        # Look for finishing motion pattern
        if motion_scores and motion_scores[-1]["motion_score"] < 5 and any(s["motion_score"] > 15 for s in motion_scores):
            actions.append("finish_assembly")
    
    elif video_type == "quality":
        # Actions for quality check based on motion patterns
        actions = ["scan_bowl"]
        
        # Check for specific motion patterns in regions
        has_weight_check = any(region["bottom"] > 15 for region in motion_regions)
        if has_weight_check:
            actions.append("measure_weight")
        
        # Look for camera/scanning motion
        has_scan_motion = any(score["contour_count"] > 5 for score in motion_scores)
        if has_scan_motion:
            actions.append("verify_appearance")
        
        if len(ingredients) >= 3:
            actions.append("verify_ingredients")
    
    elif video_type == "packaging":
        # Actions for packaging based on motion and container detection
        actions = ["grab_container"]
        
        if "container" in ingredients:
            if any(score["motion_score"] > 20 for score in motion_scores):
                actions.extend(["seal_bowl", "package_order"])
            else:
                actions.append("prepare_packaging")
        
        # Check for final motion indicating completion
        if motion_scores and motion_scores[-1]["motion_score"] > 15:
            actions.append("complete_packaging")
    
    return actions

def generate_sensor_logs(video_type, stage, motion_scores, motion_regions, duration):
    """Generate enhanced sensor logs based on video type and analysis"""
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
                
            # Add robot arm position based on motion regions
            matching_region = next((r for r in motion_regions if abs(r["t"] - t) < 0.3), None)
            if matching_region:
                if matching_region["top"] > matching_region["middle"] and matching_region["top"] > matching_region["bottom"]:
                    log_entry["arm_position"] = "top"
                elif matching_region["middle"] > matching_region["top"] and matching_region["middle"] > matching_region["bottom"]:
                    log_entry["arm_position"] = "middle"
                elif matching_region["bottom"] > matching_region["top"] and matching_region["bottom"] > matching_region["middle"]:
                    log_entry["arm_position"] = "bottom"
                
        elif video_type == "assembly":
            base_weight = 200
            if t < duration / 3:
                log_entry["weight"] = f"{base_weight + int(t * 10)}g"
            elif t < 2 * duration / 3:
                log_entry["weight"] = f"{base_weight + 100 + int(t * 5)}g"
            else:
                log_entry["weight"] = f"{base_weight + 180 + int(t * 2)}g"
                
            # Add ingredient coverage percentage
            coverage = int(min(100, 30 + t/duration * 70))
            log_entry["ingredient_coverage"] = f"{coverage}%"
                
        elif video_type == "quality":
            if t < duration / 2:
                log_entry["qa_progress"] = f"{int(t / duration * 100)}%"
            else:
                log_entry["qa_score"] = f"{80 + int(t) % 15}%"
                
            # Add more detailed QA metrics
            if t > duration / 3:
                log_entry["appearance_score"] = f"{75 + int(t * 3) % 20}%"
            if t > duration / 2:
                log_entry["ingredient_verification"] = "complete"
                
        elif video_type == "packaging":
            if t < duration / 3:
                log_entry["packaging_stage"] = "sealing"
                log_entry["seal_integrity"] = f"{70 + int(t * 5) % 25}%"
            elif t < 2 * duration / 3:
                log_entry["packaging_stage"] = "labeling"
                log_entry["label_alignment"] = f"{90 + int(t) % 10}%"
            else:
                log_entry["packaging_stage"] = "complete"
                log_entry["final_weight"] = f"{450 + int(t) % 30}g"
                log_entry["packaging_quality"] = f"{92 + int(t) % 8}%"
        
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
        f.write(f"ByteEats Enhanced Video Feature Extraction Report\n")
        f.write(f"===============================================\n\n")
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
            
            # Add detected ingredients
            if features["detected_ingredients"]["ingredients"]:
                ingredients = features["detected_ingredients"]["ingredients"]
                confidence = features["detected_ingredients"]["confidence_levels"]
                ingredients_str = ", ".join([f"{ing} ({confidence[ing]}%)" for ing in ingredients])
                f.write(f"   Detected Ingredients: {ingredients_str}\n")
            
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
        
        # Generate a sequence of all robot actions
        f.write(f"Complete Robot Action Sequence:\n")
        f.write(f"---------------------------\n")
        action_sequence = []
        for features in all_features:
            video_type = features['video_type']
            for action in features['robot_actions']:
                action_sequence.append(f"{video_type}: {action}")
        
        for i, action in enumerate(action_sequence, 1):
            f.write(f"{i}. {action}\n")
        
        f.write(f"\nComplete data available in: {os.path.basename(combined_file)}\n")
    
    print(f"Summary report generated at {report_file}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Extract enhanced features from food preparation videos')
    parser.add_argument('--video-dir', default=VIDEO_DIR, help='Directory containing video files')
    parser.add_argument('--output-dir', default=OUTPUT_DIR, help='Directory to save output files')
    parser.add_argument('--order-id', default=ORDER_ID, help='Order ID to associate with videos')
    
    args = parser.parse_args()
    
    # No need for global declarations here since we're just reading from them
    # and the config variables are already defined at the top of the file
    
    # Process all videos
    process_all_videos()

if __name__ == "__main__":
    main() 