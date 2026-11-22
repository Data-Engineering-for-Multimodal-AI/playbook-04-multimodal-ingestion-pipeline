"""Send one order to the running Order API and read it back (make order ITEM="...").

The same request as:
    curl -X POST http://localhost:8000/api/orders -H "Content-Type: application/json" -d '{...}'
followed by GET /api/orders/<order_id>. Start the API first with `make serve`.
"""

import argparse
import json
import sys
import time

import requests

API = "http://localhost:8000"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--item", default="Salmon Poke Bowl", help="menu item")
    parser.add_argument("--instructions", default="No onions")
    args = parser.parse_args()
    order = {"restaurant_id": "PH001", "restaurant_name": "Sample Poke Bar", "menu_item": args.item,
             "items": [{"name": "Salmon", "quantity": 1}, {"name": "Sushi Rice", "quantity": 1}],
             "instructions": args.instructions, "customer_id": "CUST0001"}
    try:
        response = requests.post(f"{API}/api/orders", json=order, timeout=10)
    except requests.ConnectionError:
        sys.exit(f"No Order API at {API}. Start it in another terminal with: make serve")
    print(f"POST /api/orders -> HTTP {response.status_code} {response.text}")
    response.raise_for_status()
    order_id = response.json()["order_id"]
    for _ in range(20):  # the API saves the order in a background task
        saved = requests.get(f"{API}/api/orders/{order_id}", timeout=10).json()
        if "error" not in saved:
            break
        time.sleep(0.25)
    print(f"GET /api/orders/{order_id} -> {json.dumps(saved)}")
    print(f"\nSaved as orders/{order_id}.json in the folder the API was started from. "
          f"Next: python ByteEats/services/robot_trigger.py --once")


if __name__ == "__main__":
    main()
