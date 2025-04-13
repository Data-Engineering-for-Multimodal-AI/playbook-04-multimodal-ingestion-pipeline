#!/usr/bin/env python3
"""
ByteEats Pickup Notification Service
------------------------------------
Analyzes video features to determine when an order is ready for pickup
and sends notifications to customers.
Part of the ByteEats ELT pipeline (step 3).
"""

import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("pickup-notification")

# Configuration
VIDEO_FEATURES_DIR = "video_features_enhanced"
PROCESSED_ORDERS_DIR = "processed_orders"
NOTIFICATION_LOGS_DIR = "notification_logs"
NOTIFICATION_API_URL = "http://localhost:8001/api/notifications"  # Simulated notification API

# Ensure directories exist
os.makedirs(NOTIFICATION_LOGS_DIR, exist_ok=True)

class PickupNotificationService:
    """Service to detect when orders are ready for pickup based on video features"""
    
    def __init__(self, video_features_dir: str, processed_orders_dir: str):
        self.video_features_dir = video_features_dir
        self.processed_orders_dir = processed_orders_dir
    
    def analyze_video_features(self, order_id: str) -> Dict[str, Any]:
        """Analyze video features to determine if order is ready for pickup"""
        try:
            # Look for combined features file
            combined_file = f"{self.video_features_dir}/{order_id}_combined_features.json"
            
            if not os.path.exists(combined_file):
                logger.warning(f"No video features found for order {order_id}")
                return {"ready": False, "reason": "no_video_features_found"}
            
            # Load combined features
            with open(combined_file, 'r') as f:
                combined_data = json.load(f)
            
            # Get all features
            features = combined_data.get("features", [])
            
            if not features:
                return {"ready": False, "reason": "empty_features"}
            
            # Check if packaging video exists (usually the last video in sequence)
            packaging_videos = [f for f in features if f.get("video_type") == "packaging"]
            
            if not packaging_videos:
                return {"ready": False, "reason": "no_packaging_video"}
            
            # Get the last packaging video
            packaging_video = packaging_videos[-1]
            
            # Check if key robot actions were performed
            robot_actions = packaging_video.get("robot_actions", [])
            required_actions = ["complete_packaging", "finalize_bowl"]
            
            if not any(action in robot_actions for action in required_actions):
                return {"ready": False, "reason": "packaging_incomplete"}
            
            # Check motion analysis to ensure video shows completion
            motion_analysis = packaging_video.get("motion_analysis", {})
            avg_motion = motion_analysis.get("average_motion", 0)
            
            # If there's too much motion at the end, the packaging may not be complete
            if avg_motion > 30:
                return {"ready": False, "reason": "excessive_motion"}
            
            # Order is ready for pickup
            return {
                "ready": True,
                "video_type": packaging_video.get("video_type"),
                "completion_time": packaging_video.get("extraction_timestamp"),
                "packaging_details": {
                    "actions": robot_actions,
                    "avg_motion": avg_motion
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing video features for {order_id}: {str(e)}")
            return {"ready": False, "reason": f"error: {str(e)}"}
    
    def get_order_details(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get order details from processed orders directory"""
        try:
            order_file = f"{self.processed_orders_dir}/{order_id}.json"
            
            if not os.path.exists(order_file):
                logger.warning(f"Order file not found: {order_file}")
                return None
            
            with open(order_file, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"Error reading order {order_id}: {str(e)}")
            return None
    
    def send_pickup_notification(self, order_data: Dict[str, Any], ready_status: Dict[str, Any]) -> bool:
        """Send pickup notification to customer"""
        try:
            order_id = order_data.get("order_id")
            customer_id = order_data.get("customer_id")
            
            if not customer_id:
                logger.warning(f"No customer ID for order {order_id}")
                return False
            
            # Prepare notification data
            notification_data = {
                "order_id": order_id,
                "customer_id": customer_id,
                "notification_type": "pickup_ready",
                "message": f"Your order {order_id} is ready for pickup!",
                "timestamp": datetime.now().isoformat(),
                "details": {
                    "restaurant_name": order_data.get("restaurant_name"),
                    "menu_item": order_data.get("menu_item"),
                    "completion_time": ready_status.get("completion_time")
                }
            }
            
            # In a real system, this would call an actual notification API
            # Here we simulate the notification
            logger.info(f"Sending pickup notification for order {order_id} to customer {customer_id}")
            
            # Log the notification
            notification_log = f"{NOTIFICATION_LOGS_DIR}/{order_id}_notification.json"
            with open(notification_log, 'w') as f:
                json.dump(notification_data, f, indent=2)
            
            # Update order status in the processed orders file
            order_data["order_status"] = "ready_for_pickup"
            order_data["pickup_ready_at"] = datetime.now().isoformat()
            
            with open(f"{self.processed_orders_dir}/{order_id}.json", 'w') as f:
                json.dump(order_data, f, indent=2)
            
            logger.info(f"Order {order_id} marked as ready for pickup")
            return True
            
        except Exception as e:
            logger.error(f"Error sending notification for {order_data.get('order_id')}: {str(e)}")
            return False
    
    def process_order(self, order_id: str) -> bool:
        """Process a single order for pickup notification"""
        logger.info(f"Checking pickup status for order {order_id}")
        
        # Get order details
        order_data = self.get_order_details(order_id)
        
        if not order_data:
            logger.warning(f"Cannot find order details for {order_id}")
            return False
        
        # Check if order is already marked as ready for pickup
        if order_data.get("order_status") == "ready_for_pickup":
            logger.info(f"Order {order_id} already marked as ready for pickup")
            return True
        
        # Check if robot preparation has completed
        if order_data.get("order_status") != "preparation_completed":
            logger.info(f"Order {order_id} preparation not completed yet")
            return False
        
        # Analyze video features
        ready_status = self.analyze_video_features(order_id)
        
        if ready_status.get("ready"):
            logger.info(f"Order {order_id} is ready for pickup!")
            # Send notification
            return self.send_pickup_notification(order_data, ready_status)
        else:
            logger.info(f"Order {order_id} is not ready for pickup yet: {ready_status.get('reason')}")
            return False

def poll_for_completed_orders():
    """Poll for orders that have completed preparation"""
    logger.info("Starting ByteEats Pickup Notification Service")
    
    notification_service = PickupNotificationService(VIDEO_FEATURES_DIR, PROCESSED_ORDERS_DIR)
    
    while True:
        try:
            # Find all processed order files
            order_files = list(Path(PROCESSED_ORDERS_DIR).glob("*.json"))
            order_ids = [os.path.basename(f).split('.')[0] for f in order_files if not os.path.basename(f).endswith("_original.json")]
            
            if order_ids:
                logger.info(f"Checking {len(order_ids)} orders for pickup readiness")
                
                for order_id in order_ids:
                    notification_service.process_order(order_id)
            
            # Sleep before checking again
            time.sleep(5)
            
        except KeyboardInterrupt:
            logger.info("Service shutting down")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            time.sleep(5)  # Longer sleep on error

if __name__ == "__main__":
    poll_for_completed_orders() 