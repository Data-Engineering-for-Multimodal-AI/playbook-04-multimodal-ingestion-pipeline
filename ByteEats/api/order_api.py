# ByteEats_order_api.py
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import json
import uuid
import os

app = FastAPI(title="ByteEats Order API")

# Simplified model for order items
class OrderItem(BaseModel):
    name: str
    quantity: int

# Simplified model for order input
class OrderInput(BaseModel):
    restaurant_id: str
    restaurant_name: str
    menu_item: str
    items: List[OrderItem]
    instructions: Optional[str] = None
    customer_id: Optional[str] = None

# Process order in background
def process_order_in_background(order_data):
    """Process order by saving to local file"""
    print(f"Processing ByteEats order: {order_data['order_id']}")
    
    # Create orders directory if it doesn't exist
    os.makedirs("orders", exist_ok=True)
    
    # Save order to file
    output_file = f"orders/{order_data['order_id']}.json"
    with open(output_file, 'w') as f:
        json.dump(order_data, f, indent=2)
    
    print(f"Order saved to: {output_file}")

@app.post("/api/orders")
async def create_order(order: OrderInput, background_tasks: BackgroundTasks):
    """Create a new order"""
    # Generate order ID and timestamp
    order_id = f"ORD{uuid.uuid4().hex[:4].upper()}"
    timestamp = datetime.now().isoformat()
    
    # Create order data
    order_data = order.model_dump()
    order_data.update({
        "order_id": order_id,
        "order_placed_at": timestamp,
        "order_status": "received"
    })
    
    # Process in background
    background_tasks.add_task(process_order_in_background, order_data)
    
    # Return response
    return {
        "order_id": order_id,
        "status": "received",
        "message": "Order received successfully"
    }

@app.post("/api/process-sample-order")
async def process_sample_order(background_tasks: BackgroundTasks):
    """Process the sample salmon poke bowl order"""
    # Create sample order
    sample_order = {
        "order_id": "ORD1001",
        "restaurant_id": "PH001",
        "restaurant_name": "Sample Poke Bar",
        "menu_item": "Salmon Poke Bowl",
        "order_placed_at": datetime.now().isoformat(),
        "order_status": "received",
        "delivery_type": "doorstep",
        "instructions": "Extra spicy mayo, no onions",
        "items": [
            {"name": "Salmon", "quantity": 1},
            {"name": "Sushi Rice", "quantity": 1},
            {"name": "Seaweed Salad", "quantity": 1},
            {"name": "Spicy Mayo", "quantity": 1}
        ],
        "customer_id": "CUST7890"
    }
    
    # Process in background
    background_tasks.add_task(process_order_in_background, sample_order)
    
    # Return response
    return {
        "order_id": sample_order["order_id"],
        "status": "received",
        "message": "Sample order processed successfully"
    }

@app.get("/api/orders/{order_id}")
async def get_order(order_id: str):
    """Get order details"""
    file_path = f"orders/{order_id}.json"
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            order_data = json.load(f)
        return order_data
    return {"error": f"Order {order_id} not found"}

@app.get("/")
def read_root():
    return {"message": "ByteEats Order API", "version": "1.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)