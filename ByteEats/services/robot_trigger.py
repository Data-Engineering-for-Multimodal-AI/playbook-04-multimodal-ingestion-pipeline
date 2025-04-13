#!/usr/bin/env python3
"""
ByteEats Robot Trigger Service
------------------------------
Listens for new orders and triggers the robot preparation sequence.
Part of the ByteEats ELT pipeline (step 2).
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
logger = logging.getLogger("robot-trigger")

# Configuration
ORDERS_DIR = "orders"
ROBOT_API_URL = "http://localhost:5000/api/robot"  # Simulated robot API
PROCESSED_DIR = "processed_orders"
ROBOT_LOGS_DIR = "robot_logs"

# Ensure directories exist
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(ROBOT_LOGS_DIR, exist_ok=True)

class RobotController:
    """Handles communication with the robot system"""
    
    def __init__(self, api_url: str):
        self.api_url = api_url
        self.robot_id = "PREP-BOT-001"
    
    def trigger_prep_sequence(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger the robot to prepare an order"""
        # In a real system, this would call the actual robot API
        # Here we simulate the robot response
        
        logger.info(f"Triggering robot prep for order {order_data['order_id']}")
        
        # Extract items to prepare
        items = order_data.get("items", [])
        item_names = [item.get("name") for item in items]
        
        # Determine prep sequence based on menu item
        menu_item = order_data.get("menu_item", "").lower()
        if "salmon" in menu_item and "poke" in menu_item:
            prep_sequence = self._salmon_poke_prep_sequence()
        elif "bowl" in menu_item:
            prep_sequence = self._generic_bowl_prep_sequence()
        else:
            prep_sequence = self._default_prep_sequence()
        
        # Generate robot actions log
        robot_actions = []
        timestamped_states = []
        
        start_time = datetime.now()
        current_time = 0.0
        
        for step in prep_sequence:
            # Simulate action duration
            duration = step.get("duration", 1.0)
            current_time += duration
            
            # Record action
            robot_actions.append(step["action"])
            
            # Record state
            state = {
                "t": current_time,
                "action": step["action"]
            }
            
            # Add measurements based on action type
            if "rice" in step["action"]:
                state["weight"] = f"{200 + int(current_time * 10)}g"
            elif "salmon" in step["action"]:
                state["weight"] = f"{300 + int(current_time * 5)}g"
                state["temp"] = f"{3 + current_time:.1f}°C"
            elif "sauce" in step["action"] or "mayo" in step["action"]:
                state["amount"] = f"{20 + int(current_time)}ml"
            
            timestamped_states.append(state)
            
            # Simulate robot operation
            logger.info(f"  → Robot action: {step['action']} ({duration:.1f}s)")
            time.sleep(0.1)  # Just a small delay for simulation
        
        # Total preparation time
        total_time = sum(step.get("duration", 1.0) for step in prep_sequence)
        
        # Create robot log
        robot_log = {
            "order_id": order_data["order_id"],
            "robot_id": self.robot_id,
            "prep_started_at": start_time.isoformat(),
            "total_prep_time_seconds": total_time,
            "robot_actions_log": robot_actions,
            "timestamped_state": timestamped_states,
            "status": "completed"
        }
        
        # Save robot log to file
        log_filename = f"{ROBOT_LOGS_DIR}/{order_data['order_id']}_robot_log.json"
        with open(log_filename, 'w') as f:
            json.dump(robot_log, f, indent=2)
        
        logger.info(f"Robot preparation complete. Log saved to {log_filename}")
        return robot_log
    
    def _salmon_poke_prep_sequence(self) -> List[Dict[str, Any]]:
        """Define preparation sequence for Salmon Poke Bowl"""
        return [
            {"action": "initialize_robot", "duration": 0.5},
            {"action": "grab_knife", "duration": 1.2},
            {"action": "cut_salmon", "duration": 2.5},
            {"action": "portion_salmon", "duration": 1.0},
            {"action": "grab_scoop", "duration": 0.8},
            {"action": "place_rice", "duration": 1.5},
            {"action": "add_salmon", "duration": 1.0},
            {"action": "add_seaweed", "duration": 0.7},
            {"action": "add_sauce", "duration": 0.5},
            {"action": "finalize_bowl", "duration": 1.3}
        ]
    
    def _generic_bowl_prep_sequence(self) -> List[Dict[str, Any]]:
        """Define preparation sequence for a generic bowl"""
        return [
            {"action": "initialize_robot", "duration": 0.5},
            {"action": "grab_scoop", "duration": 1.0},
            {"action": "place_base", "duration": 1.5},
            {"action": "add_protein", "duration": 2.0},
            {"action": "add_toppings", "duration": 1.5},
            {"action": "add_sauce", "duration": 0.5},
            {"action": "finalize_bowl", "duration": 1.0}
        ]
    
    def _default_prep_sequence(self) -> List[Dict[str, Any]]:
        """Define default preparation sequence"""
        return [
            {"action": "initialize_robot", "duration": 0.5},
            {"action": "prepare_ingredients", "duration": 3.0},
            {"action": "assemble_order", "duration": 2.5},
            {"action": "finalize_order", "duration": 1.0}
        ]

def process_order(order_file: str) -> bool:
    """Process a single order file and trigger robot preparation"""
    try:
        # Read order data
        with open(order_file, 'r') as f:
            order_data = json.load(f)
        
        order_id = order_data.get("order_id")
        logger.info(f"Processing order {order_id}")
        
        # Initialize robot controller
        robot = RobotController(ROBOT_API_URL)
        
        # Trigger robot preparation
        robot_log = robot.trigger_prep_sequence(order_data)
        
        # Update order status
        order_data["order_status"] = "preparation_completed"
        order_data["prep_completed_at"] = datetime.now().isoformat()
        order_data["robot_log_id"] = f"{order_id}_robot_log.json"
        
        # Save updated order to processed directory
        processed_file = os.path.join(PROCESSED_DIR, os.path.basename(order_file))
        with open(processed_file, 'w') as f:
            json.dump(order_data, f, indent=2)
        
        logger.info(f"Order {order_id} processed and moved to {processed_file}")
        
        # Move original order file to processed directory
        os.rename(order_file, os.path.join(PROCESSED_DIR, f"{order_id}_original.json"))
        
        return True
    
    except Exception as e:
        logger.error(f"Error processing order {order_file}: {str(e)}")
        return False

def poll_for_orders():
    """Poll the orders directory for new orders"""
    logger.info("Starting ByteEats Robot Trigger Service")
    logger.info(f"Watching directory: {ORDERS_DIR}")
    
    while True:
        try:
            # Find all JSON files in the orders directory
            order_files = list(Path(ORDERS_DIR).glob("*.json"))
            
            if order_files:
                logger.info(f"Found {len(order_files)} new orders to process")
                
                for order_file in order_files:
                    process_order(str(order_file))
            
            # Sleep before checking again
            time.sleep(2)
            
        except KeyboardInterrupt:
            logger.info("Service shutting down")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            time.sleep(5)  # Longer sleep on error

if __name__ == "__main__":
    poll_for_orders() 