#!/usr/bin/env python3
"""
ByteEats Image Quality Checker
-----------------------------
Compares prepared food images against reference images to ensure quality standards.
Uses OpenCV to calculate similarity scores and detect anomalies.
"""

import os
import cv2
import numpy as np
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("image-quality-checker")

# Configuration
REFERENCE_IMAGES_DIR = "reference_images"
VIDEO_FEATURES_DIR = "video_features_enhanced"
QA_RESULTS_DIR = "qa_results"

# Ensure directories exist
os.makedirs(REFERENCE_IMAGES_DIR, exist_ok=True)
os.makedirs(QA_RESULTS_DIR, exist_ok=True)

class ImageQualityChecker:
    """Compares prepared food images against reference images for QA"""
    
    def __init__(self, reference_dir: str, features_dir: str, results_dir: str):
        self.reference_dir = reference_dir
        self.features_dir = features_dir
        self.results_dir = results_dir
        
    def load_reference_image(self, menu_item: str) -> Optional[np.ndarray]:
        """Load reference image for a specific menu item"""
        # Normalize menu item name for filename
        safe_name = menu_item.lower().replace(' ', '_')
        possible_extensions = ['.jpg', '.jpeg', '.png']
        
        for ext in possible_extensions:
            ref_path = os.path.join(self.reference_dir, f"{safe_name}{ext}")
            if os.path.exists(ref_path):
                logger.info(f"Loading reference image: {ref_path}")
                return cv2.imread(ref_path)
        
        logger.warning(f"No reference image found for {menu_item}")
        return None
    
    def load_final_prepared_image(self, order_id: str) -> Optional[Tuple[np.ndarray, str]]:
        """Load the final frame from packaging or assembly videos"""
        # Find combined features file
        combined_file = os.path.join(self.features_dir, f"{order_id}_combined_features.json")
        
        if not os.path.exists(combined_file):
            logger.warning(f"No features file found for order {order_id}")
            return None
        
        try:
            # Load features
            with open(combined_file, 'r') as f:
                features_data = json.load(f)
            
            # Get all features
            all_features = features_data.get("features", [])
            
            # First try to find packaging video
            packaging_videos = [f for f in all_features if f.get("video_type") == "packaging"]
            
            if packaging_videos:
                # Use last frame from the packaging video
                frame_paths = packaging_videos[-1].get("frame_paths", [])
                if frame_paths:
                    final_frame_path = frame_paths[-1]
                    logger.info(f"Using packaging final frame: {final_frame_path}")
                    return cv2.imread(final_frame_path), "packaging"
            
            # If no packaging video, try assembly video
            assembly_videos = [f for f in all_features if f.get("video_type") == "assembly"]
            
            if assembly_videos:
                # Use last frame from the assembly video
                frame_paths = assembly_videos[-1].get("frame_paths", [])
                if frame_paths:
                    final_frame_path = frame_paths[-1]
                    logger.info(f"Using assembly final frame: {final_frame_path}")
                    return cv2.imread(final_frame_path), "assembly"
            
            logger.warning(f"No suitable final frames found for order {order_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error loading final frame for {order_id}: {str(e)}")
            return None
    
    def calculate_image_similarity(self, img1: np.ndarray, img2: np.ndarray) -> Dict[str, float]:
        """Calculate similarity metrics between two images"""
        # Resize images to same dimensions if needed
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # Convert images to same colorspace if needed
        if len(img1.shape) != len(img2.shape):
            if len(img1.shape) == 3:
                img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
            else:
                img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        
        # 1. Calculate MSE (Mean Squared Error) - lower is better
        mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
        
        # 2. Calculate SSIM (Structural Similarity Index) - higher is better
        # Convert to grayscale for SSIM
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
        
        # Constants for SSIM calculation
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        # Calculate mean, variance, and covariance
        mu1 = cv2.GaussianBlur(gray1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(gray2, (11, 11), 1.5)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = cv2.GaussianBlur(gray1 ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(gray2 ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(gray1 * gray2, (11, 11), 1.5) - mu1_mu2
        
        # Calculate SSIM
        ssim_num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        ssim_denom = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim = np.mean(ssim_num / ssim_denom)
        
        # 3. Histogram comparison
        hist1 = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        
        # Normalize histograms
        cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
        
        # Calculate histogram correlation
        hist_correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        # 4. Feature matching using ORB
        # Convert to grayscale for feature detection
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)
        
        # Calculate feature matching score
        feature_score = 0
        if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            feature_score = len(matches) / max(len(kp1), len(kp2)) if max(len(kp1), len(kp2)) > 0 else 0
        
        # Combine metrics into overall similarity score (higher is better)
        # Convert MSE to similarity score (inverse relationship)
        mse_score = 1 / (1 + mse) if mse > 0 else 1
        
        # Calculate weighted overall score
        overall_score = (0.25 * mse_score + 0.35 * ssim + 0.2 * hist_correlation + 0.2 * feature_score) * 100
        
        return {
            "mse": float(mse),
            "mse_score": float(mse_score * 100),
            "ssim": float(ssim * 100),
            "histogram_correlation": float(hist_correlation * 100),
            "feature_score": float(feature_score * 100),
            "overall_score": float(overall_score)
        }
    
    def generate_visual_comparison(self, reference_img: np.ndarray, prepared_img: np.ndarray, 
                                  order_id: str) -> str:
        """Generate visual comparison between reference and prepared images"""
        # Resize images to same dimensions
        if reference_img.shape != prepared_img.shape:
            prepared_img = cv2.resize(prepared_img, (reference_img.shape[1], reference_img.shape[0]))
        
        # Create a side-by-side comparison
        comparison = np.hstack((reference_img, prepared_img))
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, "Reference", (10, 30), font, 1, (0, 255, 0), 2)
        cv2.putText(comparison, "Prepared", (reference_img.shape[1] + 10, 30), font, 1, (0, 255, 0), 2)
        
        # Save comparison image
        comparison_path = os.path.join(self.results_dir, f"{order_id}_comparison.jpg")
        cv2.imwrite(comparison_path, comparison)
        
        return comparison_path
    
    def check_quality(self, order_id: str, menu_item: str) -> Dict[str, Any]:
        """Check quality of prepared food against reference image"""
        logger.info(f"Checking quality for order {order_id} ({menu_item})")
        
        # Load reference image
        reference_img = self.load_reference_image(menu_item)
        
        if reference_img is None:
            return {
                "order_id": order_id,
                "menu_item": menu_item,
                "quality_check_passed": None,
                "error": "No reference image available"
            }
        
        # Load prepared food image
        prepared_result = self.load_final_prepared_image(order_id)
        
        if prepared_result is None:
            return {
                "order_id": order_id,
                "menu_item": menu_item,
                "quality_check_passed": None,
                "error": "No prepared food image available"
            }
        
        prepared_img, frame_source = prepared_result
        
        # Calculate similarity
        similarity = self.calculate_image_similarity(reference_img, prepared_img)
        
        # Generate comparison visualization
        comparison_path = self.generate_visual_comparison(reference_img, prepared_img, order_id)
        
        # Determine if quality check passed (threshold: 70%)
        quality_passed = similarity["overall_score"] >= 70
        
        # Prepare quality report
        quality_report = {
            "order_id": order_id,
            "menu_item": menu_item,
            "timestamp": datetime.now().isoformat(),
            "quality_check_passed": quality_passed,
            "frame_source": frame_source,
            "similarity_metrics": similarity,
            "comparison_image_path": comparison_path
        }
        
        # Save quality report
        report_path = os.path.join(self.results_dir, f"{order_id}_quality_report.json")
        with open(report_path, 'w') as f:
            json.dump(quality_report, f, indent=2)
        
        logger.info(f"Quality check for order {order_id}: {'PASSED' if quality_passed else 'FAILED'}")
        logger.info(f"Overall score: {similarity['overall_score']:.2f}%")
        logger.info(f"Quality report saved to {report_path}")
        
        return quality_report

def check_order_quality(order_id: str, menu_item: str) -> Dict[str, Any]:
    """Check quality for a specific order"""
    checker = ImageQualityChecker(REFERENCE_IMAGES_DIR, VIDEO_FEATURES_DIR, QA_RESULTS_DIR)
    return checker.check_quality(order_id, menu_item)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python image_quality_checker.py <order_id> <menu_item>")
        sys.exit(1)
    
    order_id = sys.argv[1]
    menu_item = sys.argv[2]
    
    check_order_quality(order_id, menu_item) 