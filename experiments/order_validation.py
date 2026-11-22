"""Try this 3: which bad orders get through the Order API? (make try-order-validation)

Sends one good order and seven bad ones to POST /api/orders (step 1, through
FastAPI's test client, so no server is needed), then hands each saved order
to the robot trigger (step 2). Then it checks the same requests against a
stricter order schema defined below. Everything runs in a temporary folder;
the repository's orders/ and robot_logs/ are not touched.

Change StrictOrderInput (or add cases to CASES) and run it again.
"""

import contextlib
import io
import logging
import os
import shutil
import sys
import tempfile
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, ValidationError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ByteEats", "api"))
sys.path.insert(0, os.path.join(ROOT, "ByteEats", "services"))

GOOD = {"restaurant_id": "PH001", "restaurant_name": "Sample Poke Bar", "menu_item": "Salmon Poke Bowl",
        "items": [{"name": "Salmon", "quantity": 1}, {"name": "Sushi Rice", "quantity": 1}],
        "customer_id": "CUST7890"}

CASES = [
    ("good order", GOOD),
    ("quantity 0", {**GOOD, "items": [{"name": "Salmon", "quantity": 0}]}),
    ("quantity -2", {**GOOD, "items": [{"name": "Salmon", "quantity": -2}]}),
    ("no items", {**GOOD, "items": []}),
    ("blank menu item", {**GOOD, "menu_item": "  "}),
    ("dish not on the menu", {**GOOD, "menu_item": "Pizza Margherita"}),
    ("5,000-character instructions", {**GOOD, "instructions": "x" * 5000}),
    ("quantity as text", {**GOOD, "items": [{"name": "Salmon", "quantity": "2"}]}),
]


class StrictItem(BaseModel):
    model_config = {"strict": True}
    name: str = Field(min_length=1)
    quantity: int = Field(ge=1, le=20)


class StrictOrderInput(BaseModel):
    """The API's OrderInput plus the checks it doesn't make."""
    model_config = {"strict": True, "str_strip_whitespace": True}
    restaurant_id: str = Field(min_length=1)
    restaurant_name: str = Field(min_length=1)
    menu_item: Literal["Salmon Poke Bowl", "Miso Salmon Bowl", "Salmon Edamame Bowl"]
    items: List[StrictItem] = Field(min_length=1)
    instructions: Optional[str] = Field(default=None, max_length=500)
    customer_id: Optional[str] = None


def main():
    logging.disable(logging.CRITICAL)
    work = tempfile.mkdtemp(prefix="order-validation-")
    os.chdir(work)
    from fastapi.testclient import TestClient
    import order_api
    import robot_trigger

    client = TestClient(order_api.app)
    print(f"{'request':30} {'API (as is)':>12} {'robot prepared':>15} {'stricter schema':>16}")
    counts = [0, 0, 0]
    for label, body in CASES:
        with contextlib.redirect_stdout(io.StringIO()):
            response = client.post("/api/orders", json=body)
        accepted = response.status_code == 200
        prepared = "-"
        if accepted:
            order_file = os.path.join("orders", response.json()["order_id"] + ".json")
            prepared = "yes" if robot_trigger.process_order(order_file) else "no"
        try:
            StrictOrderInput.model_validate(body)
            strict = "accepted"
        except ValidationError:
            strict = "rejected"
        counts[0] += accepted
        counts[1] += prepared == "yes"
        counts[2] += strict == "accepted"
        api = "accepted" if accepted else f"HTTP {response.status_code}"
        print(f"{label:30} {api:>12} {prepared:>15} {strict:>16}")
    n = len(CASES)
    print(f"\n{'total accepted':30} {counts[0]:>7} of {n} {counts[1]:>10} of {n} {counts[2]:>11} of {n}")
    print(f"\nRobot preparation logs written: {len(os.listdir('robot_logs'))}")
    os.chdir(ROOT)
    shutil.rmtree(work)


if __name__ == "__main__":
    main()
